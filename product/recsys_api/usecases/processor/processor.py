from typing import Dict, Any, Optional
from pathlib import Path

from services.logger.logger import logger
from domain.messages.message import RecommendationRequest
from domain.messages.answer import RecommendationResponse
from domain.loaders.db_loader import get_user_portrait_from_db
from domain.recommendations.llm_recommender import generate_recommendations_with_llm


class RecommendationProcessor:
    """
    Процессор для генерации рекомендаций продуктов.
    
    Обрабатывает запросы на рекомендации, получает портрет пользователя
    из БД и генерирует рекомендации с помощью LLM.
    """
    
    def __init__(self, db_config: Optional[Dict[str, Any]] = None):
        """
        Инициализирует процессор.
        
        :param db_config: Конфигурация базы данных (если None, используется конфиг по умолчанию)
        """
        self._db_config = db_config
        
    async def process(self, request: RecommendationRequest) -> RecommendationResponse:
        """
        Обрабатывает запрос на рекомендации.
        
        :param request: Запрос на рекомендации
        :return: Ответ с рекомендациями
        """
        try:
            portrait = get_user_portrait_from_db(request.user_id, self._db_config)
            
            if portrait is None:
                logger.error(f"Критическая ошибка: get_user_portrait_from_db вернул None для пользователя {request.user_id}")
                return RecommendationResponse(
                    user_id=request.user_id,
                    success=False,
                    error="Критическая ошибка при получении портрета пользователя"
                )
            
            recommendations = generate_recommendations_with_llm(portrait)
            
            if recommendations is None:
                logger.error(f"Не удалось сгенерировать рекомендации для пользователя {request.user_id}")
                return RecommendationResponse(
                    user_id=request.user_id,
                    success=False,
                    error="Ошибка при генерации рекомендаций"
                )
            
            return RecommendationResponse(
                user_id=request.user_id,
                recommendations=recommendations,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Ошибка при обработке запроса для пользователя {request.user_id}: {e}", exc_info=True)
            return RecommendationResponse(
                user_id=request.user_id,
                success=False,
                error=str(e)
            )

