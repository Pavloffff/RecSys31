import os

from typing import Any, Type

from pydantic import BaseModel

from services.config.exceptions import ImproperlyConfigured
from services.logger.logger import logger


class KafkaConfig(BaseModel):
    host: str
    port: int
    topic: str
    initial_timeout: int
    group_id: str
    auto_offset_reset: str
    enable_auto_commit: bool


class Config(BaseModel):
    kafka: KafkaConfig
    
    @classmethod
    def from_env(cls) -> 'Config':
        return Config(
            kafka=KafkaConfig(
                host=cls._getenv('KAFKA_HOST'),
                port=cls._getenv('KAFKA_PORT', int),
                topic=cls._getenv('KAFKA_OUT_TOPIC'),
                auto_offset_reset=cls._getenv('KAFKA_AUTO_OFFSET_RESET'),
                enable_auto_commit=bool(cls._getenv('KAFKA_ENABLE_AUTO_COMMIT', int)),
                group_id=cls._getenv('KAFKA_OUT_GROUP_ID'),
                initial_timeout=cls._getenv('KAFKA_INITIAL_TIMEOUT', int)
            ),
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
    