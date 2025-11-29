from abc import ABC, abstractmethod


class AbstractCodec(ABC):

    @abstractmethod
    def decode(self, message: bytes, encoding: str) -> dict:
        pass

    @abstractmethod
    def encode(self, message: dict, encoding: str) -> bytes:
        pass

