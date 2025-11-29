"""
Модули для генерации рекомендаций продуктов с использованием LLM.
"""

from .llm_recommender import (
    load_products_info,
    generate_recommendations_with_llm,
    format_portrait_for_prompt,
    save_recommendations_to_json,
    print_recommendations,
    ProductRecommendation,
    RecommendationsResponse,
    RecommendationPromptData
)

__all__ = [
    'load_products_info',
    'generate_recommendations_with_llm',
    'format_portrait_for_prompt',
    'save_recommendations_to_json',
    'print_recommendations',
    'ProductRecommendation',
    'RecommendationsResponse',
    'RecommendationPromptData'
]

