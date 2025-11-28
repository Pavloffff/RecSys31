from abc import ABC, abstractmethod


class AbstractCodec(ABC):
    @abstractmethod
    def decode(self, message: str, encoding: str) -> dict:
        pass
