import threading
import time
import uuid
from typing import Optional, Dict, Any

from ..config.config import Config
from ..kafka.producer import Producer
from ..kafka.consumer import Consumer
from ..codec.json_codec import JsonCodec
from ..logger.logger import logger


class LlmClient:
    """
    Клиент для отправки запросов в llm_api через Kafka и получения ответов.
    
    Использует паттерн request-reply через Kafka топики.
    """

    def __init__(self, config: Config):
        """
        Инициализирует LlmClient.

        :param config: Конфигурация приложения
        """
        self._config = config
        self._codec = JsonCodec()
        self._producer = Producer(config.llm_kafka, self._codec)
        self._consumer = Consumer(config.out_kafka, self._codec)
        
        self._pending_requests: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        
        self._response_thread = threading.Thread(target=self._listen_for_responses, daemon=True)
        self._response_thread.start()

    def invoke(self, context: str, text: str, timeout: int = 30) -> Optional[str]:
        """
        Отправляет запрос в llm_api и ожидает ответ.

        :param context: Контекст для LLM (инструкции)
        :param text: Текст запроса
        :param timeout: Таймаут ожидания ответа в секундах
        :return: Ответ от LLM или None в случае ошибки/таймаута
        """
        request_id = str(uuid.uuid4())
        
        message = {
            'context': context,
            'text': text
        }
        
        event = threading.Event()
        response_data = {'response': None, 'error': None}
        
        with self._lock:
            self._pending_requests[request_id] = {
                'event': event,
                'response_data': response_data
            }
        
        try:
            logger.info(f"Sending request to llm_api: {message}")
            self._producer.produce(message, 'utf-8')
            
            if event.wait(timeout=timeout):
                with self._lock:
                    if request_id in self._pending_requests:
                        result = self._pending_requests[request_id]['response_data']
                        if result['error']:
                            logger.error(f"Error in LLM response: {result['error']}")
                            return None
                        return result['response']
            else:
                logger.warning(f"Timeout waiting for LLM response (request_id: {request_id})")
                with self._lock:
                    if request_id in self._pending_requests:
                        del self._pending_requests[request_id]
                return None
                
        except Exception as e:
            logger.error(f"Error sending request to llm_api: {e}", exc_info=True)
            with self._lock:
                if request_id in self._pending_requests:
                    del self._pending_requests[request_id]
            return None

    def _listen_for_responses(self):
        """
        Слушает ответы из Kafka и сопоставляет их с ожидающими запросами.
        """
        logger.info("Starting to listen for LLM responses")
        for message in self._consumer.listen():
            try:
                logger.debug(f"Received response from llm_api: {message}")
                
                # Ответ от llm_api имеет формат: {'text': '...'}
                if isinstance(message, dict) and 'text' in message:
                    response_text = message['text']
                    
                    # Находим ожидающий запрос (берем первый, так как нет request_id в текущей реализации)
                    # В идеале нужно добавить request_id в протокол
                    with self._lock:
                        if self._pending_requests:
                            # Берем первый ожидающий запрос
                            request_id = next(iter(self._pending_requests))
                            request_data = self._pending_requests.pop(request_id)
                            request_data['response_data']['response'] = response_text
                            request_data['event'].set()
                            logger.info(f"Matched response to request {request_id}")
                        else:
                            logger.warning("Received response but no pending requests")
                            
            except Exception as e:
                logger.error(f"Error processing response: {e}", exc_info=True)
                continue

