from abc import abstractmethod, ABC


class BaseLlm(ABC):
    """
    Абстрактный базовый класс для работы с LLM.
    """
    
    @abstractmethod
    def invoke(self, context: str, question: str) -> str:
        """
        Выполняет запрос к LLM.
        
        :param context: Контекст (системные инструкции) для модели
        :param question: Вопрос пользователя
        :return: Ответ от модели
        """
        pass

