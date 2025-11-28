"""
Модули для загрузки данных из Parquet файлов.
"""

from .data_loader import (
    load_parquet_without_embeddings,
    load_user_events,
    load_all_events,
    load_reference_data
)

__all__ = [
    'load_parquet_without_embeddings',
    'load_user_events',
    'load_all_events',
    'load_reference_data'
]

