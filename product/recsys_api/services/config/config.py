import os
from typing import Any, Type

from pydantic import BaseModel

from services.config.exceptions import ImproperlyConfigured
from services.logger.logger import logger


class KafkaConfig(BaseModel):
    """Конфигурация для Kafka."""

    host: str
    port: int
    topic: str
    initial_timeout: int
    group_id: str
    auto_offset_reset: str
    enable_auto_commit: bool
    retry_timeout: int


class DatabaseConfig(BaseModel):
    """Конфигурация для базы данных."""

    host: str
    port: int
    database: str
    user: str
    password: str


class AppConfig(BaseModel):
    """Конфигурация приложения."""

    title: str
    host: str
    port: int


class Config(BaseModel):
    """Основная конфигурация приложения."""

    recsys_kafka: KafkaConfig
    out_kafka: KafkaConfig
    app: AppConfig
    database: DatabaseConfig

    @classmethod
    def from_env(cls) -> 'Config':
        """
        Создает конфигурацию из переменных окружения.

        :return: Экземпляр Config
        """
        logger.info(cls._getenv('KAFKA_PORT', int))
        return Config(
            recsys_kafka=KafkaConfig(
                host=cls._getenv('KAFKA_HOST'),
                port=cls._getenv('KAFKA_PORT', int),
                topic=cls._getenv('KAFKA_REC_TOPIC'),
                auto_offset_reset=cls._getenv('KAFKA_AUTO_OFFSET_RESET'),
                enable_auto_commit=bool(cls._getenv('KAFKA_ENABLE_AUTO_COMMIT', int)),
                group_id=cls._getenv('KAFKA_REC_GROUP_ID'),
                initial_timeout=cls._getenv('KAFKA_INITIAL_TIMEOUT', int),
                retry_timeout=cls._getenv('KAFKA_RETRY_TIMEOUT', int, default=10)
            ),
            out_kafka=KafkaConfig(
                host=cls._getenv('KAFKA_HOST'),
                port=cls._getenv('KAFKA_PORT', int),
                topic=cls._getenv('KAFKA_OUT_TOPIC'),
                auto_offset_reset=cls._getenv('KAFKA_AUTO_OFFSET_RESET'),
                enable_auto_commit=bool(cls._getenv('KAFKA_ENABLE_AUTO_COMMIT', int)),
                group_id=cls._getenv('KAFKA_OUT_GROUP_ID'),
                initial_timeout=cls._getenv('KAFKA_INITIAL_TIMEOUT', int),
                retry_timeout=cls._getenv('KAFKA_RETRY_TIMEOUT', int, default=10)
            ),
            app=AppConfig(
                title='RecSys API',
                host='0.0.0.0',
                port=8002
            ),
            database=DatabaseConfig(
                host=cls._getenv('POSTGRES_HOST', default='recsys-database'),
                port=cls._getenv('POSTGRES_PORT', int, default=5432),
                database=cls._getenv('POSTGRES_DB'),
                user=cls._getenv('POSTGRES_USER'),
                password=cls._getenv('POSTGRES_PASSWORD')
            )
        )

    @staticmethod
    def _getenv(var_name: str, cast_to: Type = str, default: Any = None) -> Any:
        """
        Получает переменную окружения с приведением типа.

        :param var_name: Имя переменной окружения
        :param cast_to: Тип для приведения
        :param default: Значение по умолчанию, если переменная не найдена
        :return: Значение переменной окружения
        """
        try:
            value = os.environ[var_name]
            return cast_to(value)
        except KeyError:
            if default is not None:
                return default
            raise ImproperlyConfigured(var_name)
        except ValueError:
            raise ValueError(f"The value {var_name} can't be cast to {cast_to}.")

