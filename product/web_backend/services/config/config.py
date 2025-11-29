import os

from typing import Any, Type

from pydantic import BaseModel
from sqlalchemy.engine import URL

from services.config.exceptions import ImproperlyConfigured
from services.logger.logger import logger

class DatabaseConfig(BaseModel):
    host: str
    port: int
    user: str
    password: str
    database: str
    query_cache_size: int = 1200
    pool_size: int = 10
    max_overflow: int = 200
    future: bool = True
    echo: bool = False
    driver: str = 'postgresql+asyncpg'
    
    @property
    def url(self) -> URL:
        return URL.create(
            drivername=self.driver,
            host=self.host,
            port=self.port,
            username=self.user,
            password=self.password,
            database=self.database
        )

class KafkaConfig(BaseModel):
    host: str
    port: int
    topic: str
    initial_timeout: int
    group_id: str
    auto_offset_reset: str
    enable_auto_commit: bool
    retry_timeout: int


class AppConfig(BaseModel):
    title: str
    openapi_str: str
    host: str
    port: int


class Config(BaseModel):
    kafka: KafkaConfig
    database: DatabaseConfig
    app: AppConfig
        
    @classmethod
    def from_env(cls) -> 'Config':
        logger.info(cls._getenv('KAFKA_PORT', int))
        return Config(
            kafka=KafkaConfig(
                host=cls._getenv('KAFKA_HOST'),
                port=cls._getenv('KAFKA_PORT', int),
                topic=cls._getenv('KAFKA_REC_TOPIC'),
                auto_offset_reset=cls._getenv('KAFKA_AUTO_OFFSET_RESET'),
                enable_auto_commit=bool(cls._getenv('KAFKA_ENABLE_AUTO_COMMIT', int)),
                group_id=cls._getenv('KAFKA_REC_GROUP_ID'),
                initial_timeout=cls._getenv('KAFKA_INITIAL_TIMEOUT', int),
                retry_timeout=cls._getenv('KAFKA_RETRY_TIMEOUT', int)
            ),
            app=AppConfig(
                title='RecSys',
                openapi_str='/recsysv1',
                host='0.0.0.0',
                port=8000
            ),
            database=DatabaseConfig(
                database=cls._getenv('POSTGRES_DB'),
                host=cls._getenv('POSTGRES_HOST'),
                port=cls._getenv('POSTGRES_PORT', int),
                user=cls._getenv('POSTGRES_USER'),
                password=cls._getenv('POSTGRES_PASSWORD'),
            )
        )
    
    @staticmethod
    def _getenv(var_name: str, cast_to: Type = str) -> Any:
        try:
            value = os.environ[var_name]
            return cast_to(value)
        except KeyError:
            raise ImproperlyConfigured(var_name)
        except ValueError:
            raise ValueError(f"The value {var_name} can't be cast to {cast_to}.")
    