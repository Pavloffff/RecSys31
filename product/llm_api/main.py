import asyncio

from domain.llm.yandex_api_llm import YandexApiLlm
from domain.messages.message import Message
from services.codec.json_codec import JsonCodec
from services.config.config import Config
from services.kafka.consumer import Consumer
from services.kafka.producer import Producer
from services.logger.logger import logger
from usecases.processor.processor import MessagesProcessor


async def main():
    config = Config.from_env()
    
    llm = YandexApiLlm(config.yandex_api)
    processor = MessagesProcessor(llm)
    
    codec = JsonCodec()
    consumer = Consumer(config.llm_kafka, codec)
    producer = Producer(config.out_kafka, codec)
    
    for message in consumer.listen():
        try:
            logger.info(f'received message: {message}')
            answer = await processor.process(
                Message(
                    context=message['context'],
                    text=message['text']
                )
            )
            logger.info(f'processed answer: {answer}')
            producer.produce(answer.dict(), 'utf-8')
        except Exception as e:
            logger.error(e)
            continue


if __name__ == '__main__':
    asyncio.run(main())
