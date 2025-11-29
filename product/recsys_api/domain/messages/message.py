from typing import Dict, Any
from pydantic import BaseModel


class RecommendationRequest(BaseModel):
    """
    Запрос на рекомендации продуктов.
    
    :param user_id: ID пользователя
    :param context: Дополнительный контекст для рекомендаций
    """
    user_id: int
    context: Dict[str, Any] = {}

