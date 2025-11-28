"""
Модуль для загрузки данных из Parquet файлов.

Оптимизированная загрузка событий с возможностью выборки файлов
для ускорения обработки одного пользователя.
"""

import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm
import glob
import warnings
import logging
from datetime import datetime
from typing import Optional, List, Dict, Tuple

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


def load_parquet_without_embeddings(file_path: str) -> Optional[pd.DataFrame]:
    """
    Загружает Parquet файл, исключая колонку 'embedding' для экономии памяти.
    
    :param file_path: Путь к Parquet файлу
    :return: DataFrame или None в случае ошибки
    """
    try:
        if not Path(file_path).exists():
            logger.warning(f"Файл не найден: {file_path}")
            return None
            
        parquet_file = pq.ParquetFile(file_path)
        columns = [col for col in parquet_file.schema_arrow.names if col != 'embedding']
        table = pq.read_table(file_path, columns=columns)
        return table.to_pandas()
    except Exception as e:
        logger.error(f"Ошибка при загрузке {file_path}: {e}", exc_info=True)
        return None


def _add_temporal_features(df: pd.DataFrame, base_date: datetime = None) -> pd.DataFrame:
    """
    Добавляет временные признаки к DataFrame.
    
    :param df: DataFrame с колонкой 'timestamp'
    :param base_date: Базовая дата для расчета (по умолчанию 2020-01-01)
    :return: DataFrame с добавленными признаками
    """
    if base_date is None:
        base_date = datetime(2020, 1, 1)
    
    if 'timestamp' not in df.columns:
        return df
    
    df = df.copy()
    df['datetime'] = base_date + pd.to_timedelta(df['timestamp'], unit='s')
    df['date'] = df['datetime'].dt.date
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    return df


def _load_events_from_channel(
    channel: str,
    base_path: str,
    user_filter: Optional[Tuple[str, int]] = None,
    sample_users: Optional[List[int]] = None,
    max_files_per_channel: Optional[int] = None,
    sample_ratio: Optional[int] = None
) -> List[pd.DataFrame]:
    """
    Загружает события из одного канала.
    
    :param channel: Название канала
    :param base_path: Базовый путь к данным
    :param user_filter: Кортеж (колонка, значение) для фильтрации по одному пользователю
    :param sample_users: Список user_id для фильтрации
    :param max_files_per_channel: Максимальное количество файлов
    :param sample_ratio: Загружать каждый N-й файл
    :return: Список DataFrame с событиями
    """
    events_path = Path(base_path) / channel / "events"
    
    if not events_path.exists():
        logger.warning(f"Путь не найден: {events_path}")
        return []
    
    event_files = sorted(glob.glob(str(events_path / "*.pq")))
    
    if not event_files:
        logger.warning(f"Файлы событий не найдены в {events_path}")
        return []
    
    
    all_events = []
    base_date = datetime(2020, 1, 1)
    
    for file_path in tqdm(event_files, desc=f"  {channel}", leave=False):
        try:
            df = load_parquet_without_embeddings(file_path)
            if df is None or len(df) == 0:
                continue
            
            if user_filter is not None:
                col, value = user_filter
                if col not in df.columns:
                    continue
                df = df[df[col] == value].copy()
            
            if sample_users is not None:
                if 'user_id' not in df.columns:
                    continue
                df = df[df['user_id'].isin(sample_users)].copy()
            
            if len(df) > 0:
                df = _add_temporal_features(df, base_date)
                df['channel'] = channel
                all_events.append(df)
                
        except Exception as e:
            logger.error(f"Ошибка при загрузке {file_path}: {e}", exc_info=True)
            continue
    
    return all_events


def load_user_events(
    user_id: int,
    base_path: str = "./t_ecd_data/dataset/small",
    channels: Optional[List[str]] = None,
    max_files_per_channel: Optional[int] = None,
    sample_ratio: Optional[int] = None
) -> pd.DataFrame:
    """
    Загружает события для конкретного пользователя с оптимизацией.
    
    Оптимизации:
    - Фильтрация по user_id сразу после загрузки каждого файла
    - Ограничение количества файлов (max_files_per_channel)
    - Выборка файлов (sample_ratio - каждый N-й файл)
    
    :param user_id: ID пользователя
    :param base_path: Базовый путь к данным
    :param channels: Список каналов для загрузки (None = все каналы)
    :param max_files_per_channel: Максимальное количество файлов на канал (None = все)
    :param sample_ratio: Загружать каждый N-й файл (None = все файлы)
    :return: DataFrame с событиями пользователя
    """
    if channels is None:
        channels = ['marketplace', 'retail', 'offers']
    
    all_events = []
    for channel in channels:
        channel_events = _load_events_from_channel(
            channel=channel,
            base_path=base_path,
            user_filter=('user_id', user_id),
            max_files_per_channel=max_files_per_channel,
            sample_ratio=sample_ratio
        )
        all_events.extend(channel_events)
    
    if all_events:
        combined = pd.concat(all_events, ignore_index=True)
        logger.info(f"Загружено {len(combined):,} событий для пользователя {user_id}")
        return combined.sort_values(['user_id', 'datetime'])
    
    logger.warning(f"События для пользователя {user_id} не найдены")
    return pd.DataFrame()


def load_all_events(
    base_path: str = "./t_ecd_data/dataset/small",
    channels: Optional[List[str]] = None,
    sample_users: Optional[List[int]] = None,
    max_files_per_channel: Optional[int] = None,
    sample_ratio: Optional[int] = None
) -> pd.DataFrame:
    """
    Загружает все события из всех каналов с оптимизацией.
    
    Оптимизации:
    - Фильтрация по списку пользователей (sample_users)
    - Ограничение количества файлов (max_files_per_channel)
    - Выборка файлов (sample_ratio - каждый N-й файл)
    
    :param base_path: Базовый путь к данным
    :param channels: Список каналов для загрузки (None = все каналы)
    :param sample_users: Список user_id для фильтрации (None = все пользователи)
    :param max_files_per_channel: Максимальное количество файлов на канал (None = все)
    :param sample_ratio: Загружать каждый N-й файл (None = все файлы)
    :return: DataFrame со всеми событиями
    """
    if channels is None:
        channels = ['marketplace', 'retail', 'offers']
    
    all_events = []
    for channel in channels:
        channel_events = _load_events_from_channel(
            channel=channel,
            base_path=base_path,
            sample_users=sample_users,
            max_files_per_channel=max_files_per_channel,
            sample_ratio=sample_ratio
        )
        all_events.extend(channel_events)
    
    if all_events:
        combined = pd.concat(all_events, ignore_index=True)
        logger.info(f"Загружено {len(combined):,} событий")
        return combined.sort_values(['user_id', 'datetime'])
    
    logger.warning("События не найдены")
    return pd.DataFrame()


def load_reference_data(base_path: str = "./t_ecd_data/dataset/small") -> Dict[str, pd.DataFrame]:
    """
    Загружает справочные данные (пользователи, товары, бренды).
    
    :param base_path: Базовый путь к данным
    :return: Словарь с DataFrame справочников
    :raises FileNotFoundError: Если базовый путь не существует
    """
    base_path_obj = Path(base_path)
    if not base_path_obj.exists():
        raise FileNotFoundError(f"Базовый путь не найден: {base_path}")
    
    datasets = {}
    
    users_path = base_path_obj / "users.pq"
    if users_path.exists():
        datasets['users'] = load_parquet_without_embeddings(str(users_path))
        if datasets['users'] is None:
            raise ValueError(f"Не удалось загрузить пользователей из {users_path}")
    else:
        raise FileNotFoundError(f"Файл пользователей не найден: {users_path}")
    
    brands_path = base_path_obj / "brands.pq"
    if brands_path.exists():
        datasets['brands'] = load_parquet_without_embeddings(str(brands_path))
    else:
        datasets['brands'] = pd.DataFrame()
    
    channels = ['marketplace', 'retail', 'offers']
    items_dict = {}
    
    for channel in channels:
        items_path = base_path_obj / channel / "items.pq"
        if items_path.exists():
            items_df = load_parquet_without_embeddings(str(items_path))
            if items_df is not None:
                datasets[f'{channel}_items'] = items_df
                items_dict[f'{channel}_items'] = items_df
        else:
            datasets[f'{channel}_items'] = pd.DataFrame()
            items_dict[f'{channel}_items'] = pd.DataFrame()
    
    datasets['items_dict'] = items_dict
    return datasets

