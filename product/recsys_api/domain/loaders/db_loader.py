import pandas as pd
import psycopg2
from datetime import datetime
from typing import Optional, List, Dict, Union, Any
import logging

from domain.features.feature_engineering import create_user_features
from domain.portraits.user_portrait import create_user_portrait_from_features, create_default_portrait

logger = logging.getLogger(__name__)


def _convert_db_config_to_dict(db_config: Union[Dict[str, Union[str, int]], Any]) -> Dict[str, Union[str, int]]:
    """
    Преобразует db_config в словарь.
    
    Поддерживает:
    - Словари (dict)
    - Pydantic модели (DatabaseConfig) через model_dump() или dict()
    - Объекты с атрибутами (host, port, database, user, password)
    
    :param db_config: Конфигурация БД в любом формате
    :return: Словарь с параметрами подключения к БД
    :raises: AttributeError если не удалось преобразовать
    """
    if isinstance(db_config, dict):
        return db_config
    elif hasattr(db_config, 'model_dump'):
        # Pydantic v2
        return db_config.model_dump()
    elif hasattr(db_config, 'dict'):
        # Pydantic v1
        return db_config.dict()
    else:
        # Объект с атрибутами
        return {
            'host': db_config.host,
            'port': db_config.port,
            'database': db_config.database,
            'user': db_config.user,
            'password': db_config.password
        }


class DatabaseLoader:
    """Класс для загрузки данных из PostgreSQL."""

    def __init__(self, host: str, port: int, database: str, user: str, password: str):
        """
        Инициализирует подключение к базе данных.

        :param host: Хост базы данных
        :param port: Порт базы данных
        :param database: Имя базы данных
        :param user: Имя пользователя
        :param password: Пароль
        """
        self.connection_params = {
            'host': host,
            'port': port,
            'database': database,
            'user': user,
            'password': password
        }

    def _get_connection(self):
        """
        Создает подключение к базе данных.

        :return: Подключение к БД
        """
        return psycopg2.connect(**self.connection_params)

    def _add_temporal_features(self, df: pd.DataFrame, base_date: datetime = None) -> pd.DataFrame:
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

    def load_user_events(
        self,
        user_id: int,
        channels: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Загружает события для конкретного пользователя из БД.
        Цены и категории подтягиваются из соответствующих таблиц товаров по каналам.

        :param user_id: ID пользователя
        :param channels: Список каналов для загрузки (None = все каналы)
        :return: DataFrame с событиями пользователя
        """
        if channels is None:
            channels = ['marketplace', 'retail', 'offers']

        conn = self._get_connection()
        try:
            union_queries = []
            for channel in channels:
                union_queries.append(f"""
                    SELECT 
                        e.user_id,
                        e.item_id,
                        e.channel,
                        e.action_type,
                        e.timestamp,
                        i.price,
                        i.category,
                        i.brand_id
                    FROM public.event e
                    LEFT JOIN public.{channel}_item i ON e.item_id = i.item_id
                    WHERE e.user_id = %s AND e.channel = %s
                """)
            
            query = " UNION ALL ".join(union_queries) + " ORDER BY timestamp"
            
            # Параметры: для каждого канала нужны user_id и channel
            params = []
            for channel in channels:
                params.extend([user_id, channel])
            
            df = pd.read_sql_query(query, conn, params=params)
            
            if len(df) > 0:
                df = self._add_temporal_features(df)
                logger.info(f"Загружено {len(df):,} событий для пользователя {user_id}")
                return df.sort_values(['user_id', 'datetime'])
            else:
                logger.warning(f"События для пользователя {user_id} не найдены")
                return pd.DataFrame()
        finally:
            conn.close()

    def load_all_events(
        self,
        channels: Optional[List[str]] = None,
        sample_users: Optional[List[int]] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Загружает все события из БД с возможностью фильтрации.
        Цены и категории подтягиваются из соответствующих таблиц товаров по каналам.

        :param channels: Список каналов для загрузки (None = все каналы)
        :param sample_users: Список user_id для фильтрации (None = все пользователи)
        :param limit: Ограничение количества записей (None = без ограничений)
        :return: DataFrame со всеми событиями
        """
        if channels is None:
            channels = ['marketplace', 'retail', 'offers']

        conn = self._get_connection()
        try:
            # Собираем события из всех каналов с JOIN к соответствующим таблицам товаров
            union_queries = []
            for channel in channels:
                query_parts = [
                    f"""SELECT 
                        e.user_id,
                        e.item_id,
                        e.channel,
                        e.action_type,
                        e.timestamp,
                        i.price,
                        i.category,
                        i.brand_id
                    FROM public.event e
                    LEFT JOIN public.{channel}_item i ON e.item_id = i.item_id
                    WHERE e.channel = %s"""
                ]
                union_queries.append(" ".join(query_parts))
            
            query = " UNION ALL ".join(union_queries)
            params = list(channels)
            
            if sample_users:
                # Оборачиваем UNION ALL в подзапрос и фильтруем
                query = f"""
                    SELECT * FROM ({query}) AS combined_events
                    WHERE user_id IN ({','.join(['%s'] * len(sample_users))})
                """
                params.extend(sample_users)
            
            query += " ORDER BY timestamp"
            
            if limit:
                query += " LIMIT %s"
                params.append(limit)

            df = pd.read_sql_query(query, conn, params=params)

            if len(df) > 0:
                df = self._add_temporal_features(df)
                logger.info(f"Загружено {len(df):,} событий из БД")
                return df.sort_values(['user_id', 'datetime'])
            else:
                logger.warning("События не найдены в БД")
                return pd.DataFrame()
        finally:
            conn.close()

    def load_reference_data(self) -> Dict[str, pd.DataFrame]:
        """
        Загружает справочные данные (пользователи, товары, бренды, кластеры) из БД.

        :return: Словарь с DataFrame справочников
        """
        conn = self._get_connection()
        datasets = {}
        
        try:
            users_query = "SELECT user_id, name, cluster_id FROM public.user"
            datasets['users'] = pd.read_sql_query(users_query, conn)
            
            if datasets['users'] is None or len(datasets['users']) == 0:
                raise ValueError("Справочник пользователей пуст")
            
            brands_query = "SELECT brand_id, name FROM public.brand"
            datasets['brands'] = pd.read_sql_query(brands_query, conn)
            if datasets['brands'] is None or len(datasets['brands']) == 0:
                datasets['brands'] = pd.DataFrame()
            
            clusters_query = "SELECT cluster_id, description FROM public.cluster"
            datasets['clusters'] = pd.read_sql_query(clusters_query, conn)
            if datasets['clusters'] is None or len(datasets['clusters']) == 0:
                datasets['clusters'] = pd.DataFrame()
            
            channels = ['marketplace', 'retail', 'offers']
            items_dict = {}
            
            for channel in channels:
                items_query = f"""
                    SELECT 
                        item_id,
                        category,
                        brand_id,
                        price
                    FROM public.{channel}_item
                """
                items_df = pd.read_sql_query(items_query, conn)
                if items_df is not None and len(items_df) > 0:
                    datasets[f'{channel}_items'] = items_df
                    items_dict[f'{channel}_items'] = items_df
                else:
                    datasets[f'{channel}_items'] = pd.DataFrame()
                    items_dict[f'{channel}_items'] = pd.DataFrame()
            
            datasets['items_dict'] = items_dict
            return datasets
            
        finally:
            conn.close()


def load_user_events(
    user_id: int,
    db_config: Union[Dict[str, Union[str, int]], Any],
    channels: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Загружает события для конкретного пользователя из БД.

    :param user_id: ID пользователя
    :param db_config: Словарь с параметрами подключения к БД (host, port, database, user, password)
                     или объект DatabaseConfig (Pydantic модель)
    :param channels: Список каналов для загрузки (None = все каналы)
    :return: DataFrame с событиями пользователя
    """
    db_config_dict = _convert_db_config_to_dict(db_config)
    loader = DatabaseLoader(**db_config_dict)
    return loader.load_user_events(user_id, channels)


def load_all_events(
    db_config: Union[Dict[str, Union[str, int]], Any],
    channels: Optional[List[str]] = None,
    sample_users: Optional[List[int]] = None,
    limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Загружает все события из БД с возможностью фильтрации.

    :param db_config: Словарь с параметрами подключения к БД (host, port, database, user, password)
                     или объект DatabaseConfig (Pydantic модель)
    :param channels: Список каналов для загрузки (None = все каналы)
    :param sample_users: Список user_id для фильтрации (None = все пользователи)
    :param limit: Ограничение количества записей (None = без ограничений)
    :return: DataFrame со всеми событиями
    """
    db_config_dict = _convert_db_config_to_dict(db_config)
    loader = DatabaseLoader(**db_config_dict)
    return loader.load_all_events(channels, sample_users, limit)


def load_reference_data(db_config: Union[Dict[str, Union[str, int]], Any]) -> Dict[str, pd.DataFrame]:
    """
    Загружает справочные данные (пользователи, товары, бренды) из БД.

    :param db_config: Словарь с параметрами подключения к БД (host, port, database, user, password)
                     или объект DatabaseConfig (Pydantic модель)
    :return: Словарь с DataFrame справочников
    """
    db_config_dict = _convert_db_config_to_dict(db_config)
    loader = DatabaseLoader(**db_config_dict)
    return loader.load_reference_data()


def get_user_portrait_from_db(
    user_id: int,
    db_config: Optional[Union[Dict[str, Union[str, int]], Any]] = None
) -> Dict:
    """
    Получает портрет пользователя из БД или создает дефолтный портрет, если пользователь не найден.
    
    :param user_id: ID пользователя
    :param db_config: Словарь с параметрами подключения к БД (host, port, database, user, password)
                     или объект DatabaseConfig (Pydantic модель)
    :return: Словарь с портретом пользователя (всегда возвращает портрет, даже если пользователь не найден)
    """
    if db_config is None:
        logger.warning(f"db_config не предоставлен, создается дефолтный портрет для пользователя {user_id}")
        return create_default_portrait(user_id)
    
    try:
        try:
            db_config_dict = _convert_db_config_to_dict(db_config)
        except (AttributeError, TypeError) as e:
            logger.error(f"Не удалось преобразовать db_config в словарь. Тип: {type(db_config)}, ошибка: {e}")
            return create_default_portrait(user_id)
        
        loader = DatabaseLoader(**db_config_dict)
        
        events_df = loader.load_user_events(user_id)
        
        try:
            reference_data = loader.load_reference_data()
            users_df = reference_data.get('users', pd.DataFrame())
            clusters_df = reference_data.get('clusters', pd.DataFrame())
        except Exception as e:
            logger.warning(f"Не удалось загрузить справочные данные: {e}. Создается дефолтный портрет")
            return create_default_portrait(user_id)
        
        user_exists = len(users_df) > 0 and user_id in users_df['user_id'].values
        
        if not user_exists:
            logger.warning(f"Пользователь {user_id} не найден в справочнике пользователей. Создается дефолтный портрет")
            return create_default_portrait(user_id)
        
        user_row = users_df[users_df['user_id'] == user_id]
        if len(user_row) == 0:
            logger.warning(f"Пользователь {user_id} не найден после фильтрации. Создается дефолтный портрет")
            return create_default_portrait(user_id)
        
        user_features_df = create_user_features(events_df, user_row)
        
        portrait = create_user_portrait_from_features(user_id, user_features_df)
        
        if portrait is None:
            logger.warning(f"Не удалось создать портрет для пользователя {user_id} из признаков. Создается дефолтный портрет")
            return create_default_portrait(user_id)
        
        # Добавляем описание кластера из справочника
        cluster_id = user_row.iloc[0].get('cluster_id')
        if cluster_id is not None and len(clusters_df) > 0:
            cluster_row = clusters_df[clusters_df['cluster_id'] == cluster_id]
            if len(cluster_row) > 0:
                portrait['cluster_description'] = cluster_row.iloc[0]['description']
                logger.info(f"Добавлено описание кластера {cluster_id} для пользователя {user_id}")
            else:
                portrait['cluster_description'] = None
                logger.warning(f"Описание кластера {cluster_id} не найдено")
        else:
            portrait['cluster_description'] = None
        
        logger.info(f"Портрет пользователя {user_id} успешно создан")
        return portrait
        
    except Exception as e:
        logger.error(f"Ошибка при получении портрета пользователя {user_id}: {e}", exc_info=True)
        logger.info(f"Создается дефолтный портрет для пользователя {user_id}")
        return create_default_portrait(user_id)

