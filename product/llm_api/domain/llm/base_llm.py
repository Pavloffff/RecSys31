from abc import abstractmethod, ABC


class BaseLlm(ABC):
    @abstractmethod
    def invoke(self, context: str, question: str) -> str:
        pass
