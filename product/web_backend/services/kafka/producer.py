import time

from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable, KafkaError

from services.codec.abstract_codec import AbstractCodec
from services.config.config import KafkaConfig
from services.logger.logger import logger


class Producer:
    def __init__(self, config: KafkaConfig, codec: AbstractCodec):
        self._config = config
        self._codec = codec
        self._connect()
    
    def produce(self, message: dict, encoding: str):
        raw_message = self._codec.encode(message, encoding)
        logger.critical(f'RAW_MSG:{raw_message}')
        while True:
            try:
                future = self._producer.send(self._config.topic, raw_message)
                result = future.get(timeout=self._config.retry_timeout)
                logger.info(f"Send message: {result}")
                break
            except KafkaError as e:
                logger.error(f"Kafka send error: {e}")
                self._connect()
                time.sleep(self._config.retry_timeout)
    
    def _connect(self):
        while True:
            try:
                logger.debug(f"Attempt to connect to Kafka {self._config.host}:{self._config.port} ...")
                self._producer = KafkaProducer(
                    bootstrap_servers=[f'{self._config.host}:{self._config.port}']
                )
                logger.debug("Connected to Kafka")
                break
            except NoBrokersAvailable:
                logger.warning(f"Kafka is not available. Retry at {self._config.initial_timeout} seconds")
                time.sleep(self._config.initial_timeout)
    