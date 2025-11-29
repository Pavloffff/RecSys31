"""
Модули для загрузки данных из Parquet файлов и базы данных.
"""

from .data_loader import (
    load_parquet_without_embeddings,
    load_user_events as load_user_events_from_files,
    load_all_events as load_all_events_from_files,
    load_reference_data as load_reference_data_from_files
)

from .db_loader import (
    load_user_events,
    load_all_events,
    load_reference_data,
    get_user_portrait_from_db
)

__all__ = [
    'load_parquet_without_embeddings',
    'load_user_events',
    'load_user_events_from_files',
    'load_all_events',
    'load_all_events_from_files',
    'load_reference_data',
    'load_reference_data_from_files',
    'get_user_portrait_from_db'
]

