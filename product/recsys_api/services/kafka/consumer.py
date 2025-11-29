import json
import time

from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable

from ..codec.abstract_codec import AbstractCodec
from ..config.config import KafkaConfig
from ..logger.logger import logger


class Consumer:
    """Kafka Consumer для получения сообщений."""

    def __init__(self, config: KafkaConfig, codec: AbstractCodec):
        """
        Инициализирует Consumer.

        :param config: Конфигурация Kafka
        :param codec: Кодец для декодирования сообщений
        """
        self._config: KafkaConfig = config
        self._codec = codec
        self._consumer = self._connect()

    def listen(self):
        """
        Генератор для получения сообщений из Kafka.

        :yield: Декодированное сообщение
        """
        for message in self._consumer:
            yield self._codec.decode(message.value, 'utf-8')

    def _connect(self):
        """
        Подключается к Kafka.

        :return: KafkaConsumer
        """
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
                logger.warning(f'Kafka is not available. Retry at {self._config.initial_timeout} seconds')
                time.sleep(self._config.initial_timeout)

