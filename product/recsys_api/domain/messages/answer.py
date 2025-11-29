from typing import Optional
from pydantic import BaseModel
from domain.recommendations.llm_recommender import RecommendationsResponse


class RecommendationResponse(BaseModel):
    """
    Ответ с рекомендациями продуктов.
    
    :param user_id: ID пользователя
    :param recommendations: Типизированный объект с рекомендациями от LLM
    :param success: Флаг успешности
    :param error: Сообщение об ошибке (если есть)
    """
    user_id: int
    recommendations: Optional[RecommendationsResponse] = None
    success: bool = True
    error: Optional[str] = None

