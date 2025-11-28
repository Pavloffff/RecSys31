from openai import OpenAI


from domain.llm.base_llm import BaseLlm
from services.config.config import YandexApiConfig


# TODO: убрать хардкод, переписать на асинхронную
class YandexApiLlm(BaseLlm):
    def __init__(self, config: YandexApiConfig):
        self._config = config
        self._client = OpenAI(
            base_url='https://rest-assistant.api.cloud.yandex.net/v1',
            api_key=config.api_key,
            project=config.folder_id
        )
        
    def invoke(self, context: str, question: str) -> str:
        return self._client.responses.create(
            model=f'gpt://{self._config.folder_id}/qwen3-235b-a22b-fp8/latest',
            instructions=context,
            input=question
        ).output_text
        