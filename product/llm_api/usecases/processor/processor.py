from typing import Any

from services.logger.logger import logger

from domain.messages.answer import Answer
from domain.llm.base_llm import BaseLlm
from domain.messages.message import Message


class MessagesProcessor:
    def __init__(self, llm: BaseLlm):
        self._llm = llm
        
    async def process(self, message: Message) -> Answer:
        return Answer(
            text=self._llm.invoke(context=message.context, question=message.text)
        )
