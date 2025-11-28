import json
import time

from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable

from services.codec.abstract_codec import AbstractCodec
from services.config.config import KafkaConfig
from services.logger.logger import logger


class Consumer:
    def __init__(self, config: KafkaConfig, codec: AbstractCodec):
        self._config: KafkaConfig = config
        self._codec = AbstractCodec
        self._consumer = self._connect()
    
    def listen(self):
        for message in self._consumer:
            yield self._codec.decode(message.value, 'utf-8')
    
    def _connect(self):
        while True:
            try:
                logger.info(f'Attempt to connect to Kafka {self._config.host}:{self._config.port} ...')
                consumer = KafkaConsumer(
                    self._config.topic,
                    bootstrap_servers=[f'{self._config.host}:{self._config.port}'],
                    auto_offset_reset=self._config.auto_offset_reset,
                    enable_auto_commit=self._config.enable_auto_commit,
                    group_id=self._config.group_id
                )
                logger.info('Connected to kafka')
                return consumer
            except NoBrokersAvailable:
                logger.warning('Kafka is not available. Retry at {self._initial_timeout} seconds')
                time.sleep(self._config.initial_timeout)
    
