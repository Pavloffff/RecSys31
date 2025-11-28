import asyncio

from domain.answer.answer import Answer
from services.config.config import Config
from services.codec.json_codec import JsonCodec
from services.kafka.consumer import Consumer
from services.logger.logger import logger


async def main():
    config = Config.from_env()
    
    codec = JsonCodec()
    consumer = Consumer(config.kafka, codec)

    kafka_task = asyncio.create_task(_listen_kafka(consumer))
    
    try:
        await asyncio.gather(kafka_task)
    finally:
        kafka_task.cancel()


async def _listen_kafka(consumer: Consumer):
    async for message in consumer.listen():
        try:
            logger.info(f'received message: {message}')
            logger.info(f'WEB received answer: {Answer(text=message["text"])}')
        except Exception as ex:
            logger.error(ex)
            continue


if __name__ == '__main__':
    asyncio.run(main())
