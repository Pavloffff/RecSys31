from typing import Any

from services.logger.logger import logger


class MessagesProcessor:
    async def process(self, message: dict) -> Any:
        logger.info(message)
        