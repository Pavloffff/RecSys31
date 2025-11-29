from openai import OpenAI
import httpx

from domain.llm.base_llm import BaseLlm
from services.logger.logger import logger


class YandexApiConfig:
    """
    Конфигурация для Yandex API.
    
    :param folder_id: ID папки в Yandex Cloud
    :param api_key: API ключ для доступа к Yandex GPT
    """
    
    def __init__(self, folder_id: str, api_key: str):
        self.folder_id = folder_id
        self.api_key = api_key


class YandexApiLlm(BaseLlm):
    """
    Класс для работы с Yandex GPT API.
    
    Использует OpenAI-совместимый интерфейс для взаимодействия с Yandex Assistant API.
    """
    
    def __init__(self, config: YandexApiConfig, timeout: int = 120):
        """
        Инициализирует YandexApiLlm.
        
        :param config: Конфигурация Yandex API
        :param timeout: Таймаут для API запросов в секундах (по умолчанию 120s)
        """
        self._config = config
        self._timeout = timeout
        
        # Создаем HTTP клиент с таймаутом
        http_client = httpx.Client(
            timeout=httpx.Timeout(timeout=timeout, connect=10.0)
        )
        
        self._client = OpenAI(
            base_url='https://rest-assistant.api.cloud.yandex.net/v1',
            api_key=config.api_key,
            project=config.folder_id,
            http_client=http_client
        )
        
    def invoke(self, context: str, question: str) -> str:
        """
        Отправляет запрос к Yandex GPT API.
        
        :param context: Контекст (инструкции) для модели
        :param question: Вопрос пользователя
        :return: Ответ от модели
        :raises: Exception при ошибке API или таймауте
        """
        try:
            logger.info(f"RecSys LLM: Отправка запроса в Yandex API (timeout={self._timeout}s)")
            logger.debug(f"RecSys LLM: Context length: {len(context)}, Question length: {len(question)}")
            
            response = self._client.responses.create(
                model=f'gpt://{self._config.folder_id}/qwen3-235b-a22b-fp8/latest',
                instructions=context,
                input=question
            )
            
            output_text = response.output_text
            logger.info(f"RecSys LLM: Получен ответ от Yandex API (length: {len(output_text)} символов)")
            logger.debug(f"RecSys LLM: Response preview: {output_text[:500]}{'...' if len(output_text) > 500 else ''}")
            
            return output_text
            
        except httpx.TimeoutException as e:
            logger.error(f"RecSys LLM: Таймаут при запросе к Yandex API (timeout={self._timeout}s): {e}")
            raise
        except Exception as e:
            logger.error(f"RecSys LLM: Ошибка при запросе к Yandex API: {e}", exc_info=True)
            raise

