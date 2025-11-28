from abc import ABC, abstractmethod


class AbstractCodec(ABC):
    @abstractmethod
    def decode(message: str, encoding: str) -> dict:
        pass
