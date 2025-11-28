import asyncio
import time

from aiokafka import AIOKafkaConsumer
from aiokafka.errors import KafkaError

from services.codec.abstract_codec import AbstractCodec
from services.config.config import KafkaConfig
from services.logger.logger import logger


class Consumer:
    def __init__(self, config: KafkaConfig, codec: AbstractCodec):
        self._config: KafkaConfig = config
        self._codec = codec
        self._consumer: AIOKafkaConsumer = None
    
    async def listen(self):
        await self._connect()
        try:
            async for message in self._consumer:
                yield self._codec.decode(message.value, 'utf-8')
        finally:
            await self._consumer.stop()
    
    async def _connect(self):
        while True:
            try:
                logger.info(f'Attempt to connect to Kafka {self._config.host}:{self._config.port} ...')
                self._consumer = AIOKafkaConsumer(
                    self._config.topic,
                    bootstrap_servers=[f'{self._config.host}:{self._config.port}'],
                    auto_offset_reset=self._config.auto_offset_reset,
                    enable_auto_commit=self._config.enable_auto_commit,
                    group_id=self._config.group_id
                )
                await self._consumer.start()
                logger.info('Connected to kafka')
                return
            except KafkaError:
                logger.warning(f'Kafka is not available. Retry at {self._config.initial_timeout} seconds')
                await asyncio.sleep(self._config.initial_timeout)
    
