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
            logger.debug(f'Тип сообщения: {type(message)}, Структура: {message if isinstance(message, dict) else "не словарь"}')
            
            if not isinstance(message, dict):
                logger.error(f'Сообщение не является словарем. Тип: {type(message)}, Значение: {message}')
                continue
            
            if 'user_id' not in message:
                logger.error(f'В сообщении отсутствует обязательное поле "user_id". Доступные ключи: {list(message.keys())}')
                logger.error(f'Полное содержимое сообщения: {message}')
                continue
            
            user_id = message['user_id']
            if not isinstance(user_id, int):
                try:
                    user_id = int(user_id)
                    logger.warning(f'user_id преобразован из {type(message["user_id"])} в int: {user_id}')
                except (ValueError, TypeError) as e:
                    logger.error(f'user_id не может быть преобразован в int. Значение: {user_id}, Тип: {type(user_id)}')
                    continue
            
            request = RecommendationRequest(
                user_id=user_id,
                context=message.get('context', {})
            )
            
            response = await processor.process(request)
            
            logger.info(f'Обработан ответ: {response}')
            producer.produce(response.dict(), 'utf-8')
            
        except KeyError as e:
            logger.error(f'Отсутствует обязательное поле в сообщении: {e}. Доступные ключи: {list(message.keys()) if isinstance(message, dict) else "N/A"}', exc_info=True)
            continue
        except Exception as e:
            logger.error(f'Ошибка при обработке сообщения: {e}', exc_info=True)
            continue


if __name__ == '__main__':
    asyncio.run(main())

