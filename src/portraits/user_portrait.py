"""
Модуль для генерации портрета пользователя на основе признаков.
"""

import pandas as pd
from datetime import datetime
from typing import Dict, Optional


def create_user_portrait_from_features(
    user_id: int,
    user_features_df: pd.DataFrame
) -> Optional[Dict]:
    """
    Создает портрет пользователя на основе предобработанных признаков.
    
    :param user_id: ID пользователя
    :param user_features_df: DataFrame с признаками пользователей
    :return: Словарь с портретом пользователя или None, если пользователь не найден
    """
    print(f"\n{'='*60}")
    print(f"⚡ ГЕНЕРАЦИЯ ПОРТРЕТА ПОЛЬЗОВАТЕЛЯ: {user_id}")
    print(f"{'='*60}")
    
    # Извлекаем признаки пользователя
    user_features = user_features_df[user_features_df['user_id'] == user_id]
    
    if len(user_features) == 0:
        print(f"❌ Пользователь {user_id} не найден в признаках")
        return None
    
    user_features = user_features.iloc[0]
    
    # Формируем портрет из предобработанных признаков
    portrait = {
        'user_id': int(user_id),
        'socdem_cluster': user_features.get('socdem_cluster'),
        'region': user_features.get('region'),
        
        # Базовая статистика
        'total_events': int(user_features.get('total_events', 0)),
        'first_event': user_features.get('first_event'),
        'last_event': user_features.get('last_event'),
        'activity_days': float(user_features.get('activity_days', 0)),
        'events_per_day': float(user_features.get('events_per_day', 0)),
        
        # Воронка конверсии
        'view_count': int(user_features.get('view_count', 0)),
        'click_count': int(user_features.get('click_count', 0)),
        'purchase_count': int(user_features.get('purchase_count', 0)),
        'view_to_click_rate': float(user_features.get('view_to_click_rate', 0)),
        'click_to_purchase_rate': float(user_features.get('click_to_purchase_rate', 0)),
        'purchase_rate': float(user_features.get('purchase_rate', 0)),
        
        # Финансовые показатели
        'total_spent': float(user_features.get('total_spent', 0)),
        'avg_purchase': float(user_features.get('avg_purchase', 0)),
        'std_purchase': float(user_features.get('std_purchase', 0)),
        
        # Разнообразие
        'unique_categories': int(user_features.get('unique_categories', 0)),
        'unique_brands': int(user_features.get('unique_brands', 0)),
        'unique_channels': int(user_features.get('unique_channels', 0)),
        'is_multi_channel': bool(user_features.get('is_multi_channel', False)),
        'preferred_channel': user_features.get('preferred_channel', 'unknown'),
        'top_category': user_features.get('top_category', 'unknown'),
        
        # Временные паттерны
        'avg_hour': float(user_features.get('avg_hour', 12)) if pd.notna(user_features.get('avg_hour')) else 12,
        'hour_std': float(user_features.get('hour_std', 0)),
        'night_activity_ratio': float(user_features.get('night_activity_ratio', 0)),
        
        # Ценовые предпочтения
        'avg_price_interest': float(user_features.get('avg_price_interest', 0)),
        'price_std': float(user_features.get('price_std', 0)),
        'min_price_interest': float(user_features.get('min_price_interest', 0)),
        'max_price_interest': float(user_features.get('max_price_interest', 0)),
        'price_range': float(user_features.get('price_range', 0)),
    }
    
    print("✅ Портрет создан на основе предобработанных признаков")
    
    return portrait


def print_user_portrait(portrait: Optional[Dict]) -> None:
    """
    Красиво выводит портрет пользователя.
    
    :param portrait: Словарь с портретом пользователя
    """
    if portrait is None:
        print("❌ Портрет не найден")
        return
    
    print(f"\n{'='*60}")
    print(f"👤 ПОРТРЕТ ПОЛЬЗОВАТЕЛЯ: {portrait['user_id']}")
    print(f"{'='*60}")
    
    print(f"\n📋 БАЗОВАЯ ИНФОРМАЦИЯ:")
    print(f"  Социально-демографический кластер: {portrait.get('socdem_cluster', 'N/A')}")
    print(f"  Регион: {portrait.get('region', 'N/A')}")
    
    print(f"\n📊 АКТИВНОСТЬ:")
    print(f"  Всего событий: {portrait.get('total_events', 0):,}")
    print(f"  Первое событие: {portrait.get('first_event', 'N/A')}")
    print(f"  Последнее событие: {portrait.get('last_event', 'N/A')}")
    print(f"  Дней активности: {portrait.get('activity_days', 0):.1f}")
    print(f"  Событий в день: {portrait.get('events_per_day', 0):.2f}")
    
    print(f"\n🔄 ВОРОНКА КОНВЕРСИИ:")
    print(f"  Просмотров: {portrait.get('view_count', 0):,}")
    print(f"  Кликов: {portrait.get('click_count', 0):,}")
    print(f"  Покупок: {portrait.get('purchase_count', 0):,}")
    print(f"  Конверсия view→click: {portrait.get('view_to_click_rate', 0):.4f}")
    print(f"  Конверсия click→purchase: {portrait.get('click_to_purchase_rate', 0):.4f}")
    print(f"  Общая конверсия: {portrait.get('purchase_rate', 0):.4f}")
    
    print(f"\n💰 ФИНАНСОВЫЕ ПОКАЗАТЕЛИ:")
    print(f"  Общие траты: {portrait.get('total_spent', 0):.2f}")
    print(f"  Средний чек: {portrait.get('avg_purchase', 0):.2f}")
    print(f"  Стд. отклонение: {portrait.get('std_purchase', 0):.2f}")
    
    print(f"\n🎯 РАЗНООБРАЗИЕ:")
    print(f"  Уникальных категорий: {portrait.get('unique_categories', 0)}")
    print(f"  Топ категория: {portrait.get('top_category', 'N/A')}")
    print(f"  Уникальных брендов: {portrait.get('unique_brands', 0)}")
    print(f"  Уникальных каналов: {portrait.get('unique_channels', 0)}")
    print(f"  Мультиканальность: {'Да' if portrait.get('is_multi_channel', False) else 'Нет'}")
    print(f"  Предпочитаемый канал: {portrait.get('preferred_channel', 'N/A')}")
    
    print(f"\n⏰ ВРЕМЕННЫЕ ПАТТЕРНЫ:")
    print(f"  Средний час активности: {portrait.get('avg_hour', 'N/A'):.1f}")
    print(f"  Ночная активность: {portrait.get('night_activity_ratio', 0):.2%}")
    
    print(f"\n💵 ЦЕНОВЫЕ ПРЕДПОЧТЕНИЯ:")
    print(f"  Средний интерес к цене: {portrait.get('avg_price_interest', 0):.2f}")
    print(f"  Диапазон цен: {portrait.get('price_range', 0):.2f}")
    
    print(f"\n{'='*60}\n")


def save_portrait_to_json(portrait: Optional[Dict], output_path: str) -> None:
    """
    Сохраняет портрет пользователя в JSON файл.
    
    :param portrait: Словарь с портретом пользователя
    :param output_path: Путь для сохранения файла
    """
    if portrait is None:
        print("❌ Нечего сохранять: портрет не найден")
        return
    
    import json
    
    # Конвертируем datetime и другие типы для JSON
    portrait_json = {}
    for k, v in portrait.items():
        if isinstance(v, (pd.Timestamp, datetime)):
            portrait_json[k] = str(v)
        elif pd.isna(v):
            portrait_json[k] = None
        else:
            portrait_json[k] = v
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(portrait_json, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"💾 Портрет сохранен в {output_path}")

