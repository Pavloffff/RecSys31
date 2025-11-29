import asyncio
from typing import Optional, Dict, Any
from services.config.config import Config
from services.codec.json_codec import JsonCodec
from services.kafka.consumer import Consumer
from services.logger.logger import logger


class KafkaManager:
    def __init__(self):
        self.latest_message: Optional[Dict[str, Any]] = None
        self._lock = asyncio.Lock()
    
    async def start_consuming(self):
        config = Config.from_env()
        codec = JsonCodec()
        consumer = Consumer(config.kafka, codec)

        async for message in consumer.listen():
            try:
                logger.info(f'Received message: {message}')
                
                async with self._lock:
                    self.latest_message = message
                    
            except Exception as ex:
                logger.error(f'Error processing message: {ex}')
                continue
    
    def get_latest_message(self) -> Optional[Dict[str, Any]]:
        return self.latest_message


data_manager = KafkaManager()


async def run_kafka_consumer():
    """Запуск Kafka consumer"""
    await data_manager.start_consuming()