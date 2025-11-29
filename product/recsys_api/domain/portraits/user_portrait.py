"""
Модуль для генерации портрета пользователя на основе признаков.
"""

import pandas as pd
import logging
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)


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
    user_features = user_features_df[user_features_df['user_id'] == user_id]
    
    if len(user_features) == 0:
        logger.error(f"❌ Пользователь {user_id} не найден в признаках")
        return None
    
    user_features = user_features.iloc[0]
    
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
    
    return portrait


