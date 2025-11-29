import asyncio

from domain.messages.message import RecommendationRequest
from services.codec.json_codec import JsonCodec
from services.config.config import Config
from services.kafka.consumer import Consumer
from services.kafka.producer import Producer
from services.logger.logger import logger
from usecases.processor.processor import RecommendationProcessor


async def main():
    config = Config.from_env()
    
    processor = RecommendationProcessor(db_config=config.database)
    
    codec = JsonCodec()
    consumer = Consumer(config.recsys_kafka, codec)
    producer = Producer(config.out_kafka, codec)
    
    logger.info("RecSys API сервис запущен и слушает Kafka")
    
    for message in consumer.listen():
        try:
            logger.info(f'Получено сообщение: {message}')
            
            request = RecommendationRequest(
                user_id=message['user_id'],
                context=message.get('context', {})
            )
            
            response = await processor.process(request)
            
            logger.info(f'Обработан ответ: {response}')
            producer.produce(response.dict(), 'utf-8')
            
        except Exception as e:
            logger.error(f'Ошибка при обработке сообщения: {e}', exc_info=True)
            continue


if __name__ == '__main__':
    asyncio.run(main())

