import asyncio

from services.codec.json_codec import JsonCodec
from services.config.config import Config
from services.kafka.consumer import Consumer
from services.logger.logger import logger
from usecases.processor.processor import MessagesProcessor


async def main():
    config = Config.from_env()
    codec = JsonCodec()
    consumer = Consumer(config.kafka, codec)
    processor = MessagesProcessor()
    
    for message in consumer.listen():
        try:
            logger.debug(f'received message: {message}')
            await processor.process(message)
        except Exception as e:
            logger.error(e)
            continue


if __name__ == '__main__':
    asyncio.run(main())
