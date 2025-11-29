"""
Модуль для предобработки данных и объединения событий с товарами.
"""

import pandas as pd
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def merge_events_with_items(
    events_df: pd.DataFrame,
    items_dict: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Объединяет события с данными о товарах по каналам.
    
    :param events_df: DataFrame с событиями
    :param items_dict: Словарь с данными о товарах по каналам
    :return: Объединенный DataFrame
    :raises ValueError: Если events_df не содержит колонку 'channel'
    """
    if len(events_df) == 0:
        return events_df
    
    if 'channel' not in events_df.columns:
        return events_df
    
    merged_events_list = []
    channels = events_df['channel'].unique()
    
    for channel in channels:
        channel_events = events_df[events_df['channel'] == channel].copy()
        items_key = f'{channel}_items'
        items_df = items_dict.get(items_key)
        
        if items_df is not None and len(items_df) > 0 and len(channel_events) > 0:
            try:
                channel_merged = channel_events.merge(
                    items_df,
                    on='item_id',
                    how='left',
                    suffixes=('', '_item')
                )
                merged_events_list.append(channel_merged)
            except KeyError:
                merged_events_list.append(channel_events)
        elif len(channel_events) > 0:
            merged_events_list.append(channel_events)
    
    if merged_events_list:
        events_merged = pd.concat(merged_events_list, ignore_index=True)
        return events_merged
    
    return events_df

