"""
Модуль для создания признаков пользователей на основе их поведения.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List

# Константы для признаков
NUMERIC_FEATURES = [
    'total_events', 'activity_days', 'events_per_day',
    'view_count', 'click_count', 'purchase_count',
    'view_to_click_rate', 'click_to_purchase_rate', 'purchase_rate',
    'total_spent', 'avg_purchase', 'std_purchase',
    'unique_categories', 'unique_brands', 'unique_channels',
    'avg_hour', 'hour_std', 'night_activity_ratio',
    'avg_price_interest', 'price_std', 'min_price_interest', 
    'max_price_interest', 'price_range'
]

BOOL_FEATURES = ['is_multi_channel']

CATEGORICAL_FEATURES = ['preferred_channel', 'top_category']

NIGHT_HOURS_START = 22
NIGHT_HOURS_END = 6


def _initialize_empty_features(user_base: pd.DataFrame) -> pd.DataFrame:
    """
    Инициализирует пустые признаки для пользователей без событий.
    
    :param user_base: DataFrame с базовой информацией о пользователях
    :return: DataFrame с инициализированными признаками
    """
    empty_features = user_base.copy()
    
    for col in NUMERIC_FEATURES:
        empty_features[col] = 0
    
    for col in BOOL_FEATURES:
        empty_features[col] = False
    
    for col in CATEGORICAL_FEATURES:
        empty_features[col] = 'unknown'
    
    empty_features['first_event'] = pd.NaT
    empty_features['last_event'] = pd.NaT
    
    return empty_features


def _calculate_basic_stats(events_sorted: pd.DataFrame) -> pd.DataFrame:
    """
    Вычисляет базовую статистику по событиям.
    
    :param events_sorted: Отсортированный DataFrame с событиями
    :return: DataFrame с базовой статистикой
    """
    all_events_stats = events_sorted.groupby('user_id').agg({
        'item_id': 'count',
        'datetime': ['min', 'max'],
    }).reset_index()
    all_events_stats.columns = ['user_id', 'total_events', 'first_event', 'last_event']
    
    # Период активности
    all_events_stats['activity_days'] = (
        (all_events_stats['last_event'] - all_events_stats['first_event']).dt.total_seconds() / 86400
    ).fillna(0)
    all_events_stats['events_per_day'] = (
        all_events_stats['total_events'] / (all_events_stats['activity_days'] + 1)
    )
    
    return all_events_stats


def _calculate_funnel_stats(events_sorted: pd.DataFrame, all_events_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Вычисляет статистику воронки конверсии.
    
    :param events_sorted: Отсортированный DataFrame с событиями
    :param all_events_stats: DataFrame с базовой статистикой
    :return: DataFrame со статистикой воронки
    """
    funnel_stats = events_sorted.groupby(['user_id', 'action_type']).size().unstack(fill_value=0)
    funnel_stats = funnel_stats.reset_index()
    
    # Безопасное извлечение счетчиков
    action_types = ['view', 'click', 'clickout']
    for action in action_types:
        col_name = f'{action}_count' if action != 'clickout' else 'purchase_count'
        funnel_stats[col_name] = funnel_stats.get(action, 0)
    
    # Объединяем с базовой статистикой
    funnel_stats = funnel_stats.merge(
        all_events_stats[['user_id', 'total_events']], 
        on='user_id', 
        how='left'
    )
    
    # Вычисляем конверсии
    funnel_stats['view_to_click_rate'] = (
        funnel_stats['click_count'] / (funnel_stats['view_count'] + 1)
    )
    funnel_stats['click_to_purchase_rate'] = (
        funnel_stats['purchase_count'] / (funnel_stats['click_count'] + 1)
    )
    funnel_stats['purchase_rate'] = (
        funnel_stats['purchase_count'] / (funnel_stats['total_events'] + 1)
    )
    
    return funnel_stats[['user_id', 'view_count', 'click_count', 'purchase_count',
                         'view_to_click_rate', 'click_to_purchase_rate', 'purchase_rate']]


def _calculate_purchase_stats(events_sorted: pd.DataFrame) -> pd.DataFrame:
    """
    Вычисляет статистику по покупкам.
    
    :param events_sorted: Отсортированный DataFrame с событиями
    :return: DataFrame со статистикой покупок
    """
    purchases = events_sorted[events_sorted['action_type'] == 'clickout'].copy()
    
    if len(purchases) == 0:
        return pd.DataFrame({'user_id': []})
    
    purchases_with_price = purchases[purchases['price'].notna()].copy()
    
    if len(purchases_with_price) == 0:
        return pd.DataFrame({'user_id': []})
    
    purchase_stats = purchases_with_price.groupby('user_id').agg({
        'price': ['sum', 'mean', 'std'],
        'item_id': 'count'
    }).reset_index()
    purchase_stats.columns = [
        'user_id', 'total_spent', 'avg_purchase', 'std_purchase', 'purchase_count'
    ]
    purchase_stats['std_purchase'] = purchase_stats['std_purchase'].fillna(0)
    
    return purchase_stats


def _calculate_diversity_stats(events_sorted: pd.DataFrame) -> pd.DataFrame:
    """
    Вычисляет признаки разнообразия.
    
    :param events_sorted: Отсортированный DataFrame с событиями
    :return: DataFrame с признаками разнообразия
    """
    diversity_stats = events_sorted.groupby('user_id').agg({
        'category': lambda x: x.nunique() if 'category' in events_sorted.columns and x.notna().any() else 0,
        'brand_id': lambda x: x.nunique() if 'brand_id' in events_sorted.columns and x.notna().any() else 0,
        'channel': lambda x: x.nunique() if 'channel' in events_sorted.columns and x.notna().any() else 0,
    }).reset_index()
    diversity_stats.columns = ['user_id', 'unique_categories', 'unique_brands', 'unique_channels']
    
    # Мультиканальность
    diversity_stats['is_multi_channel'] = diversity_stats['unique_channels'] > 1
    
    # Предпочитаемый канал
    if 'channel' in events_sorted.columns:
        preferred_channels = events_sorted.groupby('user_id')['channel'].agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'unknown'
        ).reset_index()
        preferred_channels.columns = ['user_id', 'preferred_channel']
        diversity_stats = diversity_stats.merge(preferred_channels, on='user_id', how='left')
    else:
        diversity_stats['preferred_channel'] = 'unknown'
    
    return diversity_stats


def _calculate_temporal_stats(events_sorted: pd.DataFrame) -> pd.DataFrame:
    """
    Вычисляет временные признаки.
    
    :param events_sorted: Отсортированный DataFrame с событиями
    :return: DataFrame с временными признаками
    """
    if 'hour' not in events_sorted.columns:
        return pd.DataFrame({
            'user_id': events_sorted['user_id'].unique(),
            'avg_hour': 12,
            'hour_std': 0,
            'night_activity_ratio': 0
        })
    
    temporal_stats = events_sorted.groupby('user_id')['hour'].agg(['mean', 'std']).reset_index()
    temporal_stats.columns = ['user_id', 'avg_hour', 'hour_std']
    temporal_stats['hour_std'] = temporal_stats['hour_std'].fillna(0)
    
    # Ночной покупатель (активность с 22:00 до 6:00)
    night_activity = events_sorted.groupby('user_id')['hour'].apply(
        lambda x: ((x < NIGHT_HOURS_END) | (x >= NIGHT_HOURS_START)).sum() / len(x)
    ).reset_index()
    night_activity.columns = ['user_id', 'night_activity_ratio']
    temporal_stats = temporal_stats.merge(night_activity, on='user_id', how='left')
    
    return temporal_stats


def _calculate_price_stats(events_sorted: pd.DataFrame) -> pd.DataFrame:
    """
    Вычисляет ценовые признаки.
    
    :param events_sorted: Отсортированный DataFrame с событиями
    :return: DataFrame с ценовыми признаками
    """
    if 'price' not in events_sorted.columns:
        return pd.DataFrame({'user_id': []})
    
    price_events = events_sorted[events_sorted['price'].notna()]
    if len(price_events) == 0:
        return pd.DataFrame({'user_id': []})
    
    price_stats = price_events.groupby('user_id')['price'].agg(['mean', 'std', 'min', 'max']).reset_index()
    price_stats.columns = ['user_id', 'avg_price_interest', 'price_std', 'min_price_interest', 'max_price_interest']
    price_stats['price_std'] = price_stats['price_std'].fillna(0)
    price_stats['price_range'] = price_stats['max_price_interest'] - price_stats['min_price_interest']
    
    return price_stats


def _calculate_top_category(events_sorted: pd.DataFrame) -> pd.DataFrame:
    """
    Вычисляет топ категорию пользователя.
    
    :param events_sorted: Отсортированный DataFrame с событиями
    :return: DataFrame с топ категорией
    """
    if 'category' not in events_sorted.columns:
        return pd.DataFrame({'user_id': [], 'top_category': []})
    
    events_with_category = events_sorted[events_sorted['category'].notna()]
    if len(events_with_category) == 0:
        return pd.DataFrame({'user_id': [], 'top_category': []})
    
    top_categories = events_with_category.groupby(
        ['user_id', 'category']
    ).size().reset_index(name='category_count')
    
    if len(top_categories) == 0:
        return pd.DataFrame({'user_id': [], 'top_category': []})
    
    top_category = top_categories.loc[top_categories.groupby('user_id')['category_count'].idxmax()]
    top_category = top_category[['user_id', 'category']].rename(columns={'category': 'top_category'})
    
    return top_category


def create_user_features(
    events_df: pd.DataFrame,
    users_df: pd.DataFrame,
    items_dict: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Создает ключевые признаки для пользователей на основе их поведения.
    
    :param events_df: DataFrame с событиями
    :param users_df: DataFrame с пользователями
    :param items_dict: Словарь с данными о товарах по каналам (не используется, но оставлен для совместимости)
    :return: DataFrame с признаками пользователей
    """
    user_base = users_df.copy()
    
    if len(events_df) == 0:
        return _initialize_empty_features(user_base)
    
    events_sorted = events_df.sort_values(['user_id', 'datetime']).copy()
    
    # Вычисляем все признаки
    all_events_stats = _calculate_basic_stats(events_sorted)
    funnel_stats = _calculate_funnel_stats(events_sorted, all_events_stats)
    purchase_stats = _calculate_purchase_stats(events_sorted)
    diversity_stats = _calculate_diversity_stats(events_sorted)
    temporal_stats = _calculate_temporal_stats(events_sorted)
    price_stats = _calculate_price_stats(events_sorted)
    top_category = _calculate_top_category(events_sorted)
    
    # Объединяем все признаки
    user_features_df = user_base.copy()
    user_features_df = user_features_df.merge(all_events_stats, on='user_id', how='left')
    user_features_df = user_features_df.merge(funnel_stats, on='user_id', how='left')
    
    if len(purchase_stats) > 0:
        user_features_df = user_features_df.merge(purchase_stats, on='user_id', how='left')
    
    user_features_df = user_features_df.merge(diversity_stats, on='user_id', how='left')
    user_features_df = user_features_df.merge(temporal_stats, on='user_id', how='left')
    
    if len(price_stats) > 0:
        user_features_df = user_features_df.merge(price_stats, on='user_id', how='left')
    
    if len(top_category) > 0:
        user_features_df = user_features_df.merge(top_category, on='user_id', how='left')
    
    # Заполняем пустые значения
    for col in NUMERIC_FEATURES:
        if col in user_features_df.columns:
            user_features_df[col] = user_features_df[col].fillna(0)
    
    for col in BOOL_FEATURES:
        if col in user_features_df.columns:
            user_features_df[col] = user_features_df[col].fillna(False)
    
    for col in CATEGORICAL_FEATURES:
        if col in user_features_df.columns:
            user_features_df[col] = user_features_df[col].fillna('unknown')
    
    return user_features_df

