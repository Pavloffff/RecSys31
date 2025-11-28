"""
Главный скрипт для генерации портрета одного пользователя.

Использование:
    python main.py --user_id 25770580
    python main.py --user_id 25770580 --max_files 10 --sample_ratio 5
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import os

load_dotenv()
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.loaders import load_user_events, load_reference_data
from src.preprocessing import merge_events_with_items
from src.features import create_user_features
from src.portraits import create_user_portrait_from_features, print_user_portrait, save_portrait_to_json
from src.recommendations import generate_recommendations_with_llm, print_recommendations, save_recommendations_to_json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def validate_args(args) -> bool:
    """
    Валидирует аргументы командной строки.
    
    :param args: Аргументы парсера
    :return: True если валидация прошла успешно
    """
    if args.user_id <= 0:
        logger.error(f"❌ Некорректный user_id: {args.user_id}")
        return False
    
    if args.max_files is not None and args.max_files <= 0:
        logger.error(f"❌ max_files должен быть положительным числом: {args.max_files}")
        return False
    
    if args.sample_ratio is not None and args.sample_ratio <= 0:
        logger.error(f"❌ sample_ratio должен быть положительным числом: {args.sample_ratio}")
        return False
    
    base_path = Path(args.base_path)
    if not base_path.exists():
        logger.error(f"❌ Базовый путь не существует: {args.base_path}")
        return False
    
    return True


def load_and_validate_user_data(user_id: int, base_path: str) -> tuple:
    """
    Загружает и валидирует данные пользователя.
    
    :param user_id: ID пользователя
    :param base_path: Базовый путь к данным
    :return: Кортеж (users_df, items_dict) или (None, None) при ошибке
    """
    try:
        datasets = load_reference_data(base_path=base_path)
        
        users_df = datasets['users']
        if users_df is None or len(users_df) == 0:
            logger.error("❌ Справочник пользователей пуст")
            return None, None
        
        if user_id not in users_df['user_id'].values:
            logger.error(f"❌ Пользователь {user_id} не найден в справочнике")
            return None, None
        
        users_df = users_df[users_df['user_id'] == user_id].copy()
        items_dict = datasets['items_dict']
        
        return users_df, items_dict
        
    except Exception as e:
        logger.error(f"❌ Ошибка при загрузке справочных данных: {e}", exc_info=True)
        return None, None


def process_user_portrait(
    user_id: int,
    base_path: str,
    channels: list,
    max_files: Optional[int],
    sample_ratio: Optional[int],
    output_path: Optional[str] = None,
    generate_recommendations: bool = True,
    llm_api_key: Optional[str] = None,
    llm_model: str = "google/gemini-2.5-flash",
    base_url: str = "https://openrouter.ai/api/v1",
    products_path: Optional[str] = None
) -> int:
    """
    Обрабатывает портрет пользователя и генерирует рекомендации.
    
    :param user_id: ID пользователя
    :param base_path: Базовый путь к данным
    :param channels: Список каналов
    :param max_files: Максимальное количество файлов
    :param sample_ratio: Коэффициент выборки файлов
    :param output_path: Путь для сохранения портрета
    :param generate_recommendations: Генерировать ли рекомендации через LLM
    :param llm_api_key: API ключ для LLM
    :param llm_model: Модель LLM для использования
    :param base_url: Базовый URL для LLM API
    :param products_path: Путь к файлу с информацией о продуктах
    :return: Код возврата (0 = успех, 1 = ошибка)
    """
    try:
        users_df, items_dict = load_and_validate_user_data(user_id, base_path)
        if users_df is None:
            return 1
        
        user_events = load_user_events(
            user_id=user_id,
            base_path=base_path,
            channels=channels,
            max_files_per_channel=max_files,
            sample_ratio=sample_ratio
        )
        
        if len(user_events) == 0:
            logger.warning(f"Пользователь {user_id} не имеет событий")
        
        events_merged = merge_events_with_items(user_events, items_dict)
        
        user_features_df = create_user_features(
            events_df=events_merged,
            users_df=users_df,
            items_dict=items_dict
        )
        portrait = create_user_portrait_from_features(
            user_id=user_id,
            user_features_df=user_features_df
        )
        
        if portrait is None:
            logger.error(f"❌ Не удалось создать портрет для пользователя {user_id}")
            return 1
        
        print_user_portrait(portrait)
        
        if output_path:
            save_portrait_to_json(portrait, output_path)
        else:
            output_dir = project_root / "output"
            output_dir.mkdir(exist_ok=True)
            default_output = output_dir / f"user_portrait_{user_id}.json"
            save_portrait_to_json(portrait, str(default_output))
        
        if generate_recommendations:
            try:
                recommendations = generate_recommendations_with_llm(
                    portrait=portrait,
                    products_path=products_path,
                    api_key=llm_api_key,
                    model=llm_model,
                    base_url=base_url
                )
                
                if recommendations:
                    print_recommendations(recommendations)
                    
                    output_dir = project_root / "output"
                    output_dir.mkdir(exist_ok=True)
                    rec_output = output_dir / f"user_recommendations_{user_id}.json"
                    save_recommendations_to_json(recommendations, str(rec_output))
                else:
                    logger.warning("Не удалось сгенерировать рекомендации")
                    
            except Exception as e:
                logger.error(f"❌ Ошибка при генерации рекомендаций: {e}", exc_info=True)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}", exc_info=True)
        return 1


def main():
    """Главная функция для генерации портрета пользователя."""
    parser = argparse.ArgumentParser(
        description='Генерация портрета пользователя на основе событий',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  # Базовый запуск для одного пользователя
  python main.py --user_id 25770580
  
  # С ограничением количества файлов (быстрее)
  python main.py --user_id 25770580 --max_files 10
  
  # С выборкой файлов (каждый 5-й файл)
  python main.py --user_id 25770580 --sample_ratio 5
  
  # С указанием каналов
  python main.py --user_id 25770580 --channels marketplace retail
  
  # С сохранением результата
  python main.py --user_id 25770580 --output user_portrait.json
  
  # С генерацией рекомендаций через LLM
  python main.py --user_id 25770580 --generate_recommendations
  
  # С указанием модели LLM
  python main.py --user_id 25770580 --generate_recommendations --llm_model gpt-4
        """
    )
    
    parser.add_argument(
        '--user_id',
        type=int,
        required=True,
        help='ID пользователя для генерации портрета'
    )
    
    parser.add_argument(
        '--base_path',
        type=str,
        default='./data/t_ecd_data/dataset/small',
        help='Базовый путь к данным (по умолчанию: ./data/t_ecd_data/dataset/small)'
    )
    
    parser.add_argument(
        '--channels',
        nargs='+',
        default=['marketplace', 'retail', 'offers'],
        help='Каналы для загрузки (по умолчанию: все каналы)'
    )
    
    parser.add_argument(
        '--max_files',
        type=int,
        default=None,
        help='Максимальное количество файлов на канал (для ускорения)'
    )
    
    parser.add_argument(
        '--sample_ratio',
        type=int,
        default=None,
        help='Загружать каждый N-й файл (для ускорения, например: 5 = каждый 5-й файл)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Путь для сохранения портрета в JSON (опционально)'
    )
    
    parser.add_argument(
        '--log_level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Уровень логирования (по умолчанию: INFO)'
    )
    
    parser.add_argument(
        '--generate_recommendations',
        action='store_true',
        help='Генерировать рекомендации продуктов с использованием LLM'
    )
    
    parser.add_argument(
        '--no_recommendations',
        action='store_false',
        dest='generate_recommendations',
        help='Отключить генерацию рекомендаций (по умолчанию: включено)'
    )
    
    # Устанавливаем значение по умолчанию после парсинга
    parser.set_defaults(generate_recommendations=True)
    
    parser.add_argument(
        '--llm_api_key',
        type=str,
        default=None,
        help='API ключ для LLM (если не указан, берется из переменной окружения OPENAI_API_KEY или OPENROUTER_API_KEY)'
    )
    
    parser.add_argument(
        '--llm_model',
        type=str,
        default='google/gemini-2.5-flash',
        help='Модель LLM для использования (по умолчанию: google/gemini-2.5-flash)'
    )
    
    parser.add_argument(
        '--llm_base_url',
        type=str,
        default='https://openrouter.ai/api/v1',
        help='Базовый URL для LLM API (по умолчанию: https://openrouter.ai/api/v1)'
    )
    
    parser.add_argument(
        '--products_path',
        type=str,
        default=None,
        help='Путь к файлу с информацией о продуктах (по умолчанию: research/psb_products.md)'
    )
    
    args = parser.parse_args()
    
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    logger.info(f"Генерация портрета пользователя {args.user_id}")
    
    if not validate_args(args):
        return 1
    
    exit_code = process_user_portrait(
        user_id=args.user_id,
        base_path=args.base_path,
        channels=args.channels,
        max_files=args.max_files,
        sample_ratio=args.sample_ratio,
        output_path=args.output,
        generate_recommendations=args.generate_recommendations,
        llm_api_key=args.llm_api_key,
        llm_model=args.llm_model,
        base_url=args.llm_base_url,
        products_path=args.products_path
    )
    
    return exit_code


if __name__ == '__main__':
    exit(main())

