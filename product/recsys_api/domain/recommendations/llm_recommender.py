"""
Модуль для генерации рекомендаций продуктов с использованием LLM.

Использует портрет пользователя и информацию о продуктах для генерации
персонализированных рекомендаций через прямой вызов Yandex GPT API.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field

from domain.llm.yandex_api_llm import YandexApiLlm, YandexApiConfig
from services.logger.logger import logger


class ProductRecommendation(BaseModel):
    """
    Схема для одной рекомендации продукта.
    
    :param product_name: Название продукта
    :param priority: Приоритет рекомендации (1 - наивысший)
    :param reasoning: Обоснование, почему продукт подходит клиенту
    :param key_benefits: Список ключевых преимуществ для клиента
    :param match_score: Оценка соответствия продукта клиенту (0.0 - 1.0)
    """
    product_name: str = Field(..., description="Название продукта")
    priority: int = Field(..., ge=1, description="Приоритет рекомендации (1 - наивысший)")
    reasoning: str = Field(..., description="Обоснование, почему продукт подходит клиенту")
    key_benefits: List[str] = Field(default_factory=list, description="Список ключевых преимуществ")
    marketing_strategy: str = Field(..., description="Маркетинговая стратегия банка")
    consequences: str = Field(..., description="Ожидаемые результаты")
    match_score: float = Field(..., ge=0.0, le=1.0, description="Оценка соответствия продукта (0.0 - 1.0)")


class RecommendationsResponse(BaseModel):
    """
    Схема ответа с рекомендациями продуктов от LLM.
    
    :param recommendations: Список рекомендаций продуктов
    :param summary: Краткое резюме рекомендаций
    """
    recommendations: List[ProductRecommendation] = Field(..., description="Список рекомендаций продуктов")
    summary: str = Field(..., description="Краткое резюме рекомендаций")


class RecommendationPromptData(BaseModel):
    """
    Схема данных для создания промпта рекомендаций.
    
    :param portrait_text: Отформатированный текст портрета пользователя
    :param products_text: Текст с информацией о продуктах
    """
    portrait_text: str = Field(..., description="Отформатированный текст портрета пользователя")
    products_text: str = Field(..., description="Текст с информацией о продуктах")

_llm: Optional[YandexApiLlm] = None


def load_products_info(products_path: Optional[str] = None) -> str:
    """
    Загружает информацию о продуктах из файла.
    
    :param products_path: Путь к файлу с информацией о продуктах
    :return: Текст с информацией о продуктах
    """
    if products_path is None:
        project_root = Path(__file__).parent.parent.parent.parent
        products_path = project_root / "research" / "psb_products.md"
    
    products_path = Path(products_path)
    
    if not products_path.exists():
        logger.warning(f"Файл с продуктами не найден: {products_path}")
        return "Информация о продуктах недоступна."
    
    try:
        with open(products_path, 'r', encoding='utf-8') as f:
            products_text = f.read()
        return products_text
    except Exception as e:
        logger.error(f"Ошибка при загрузке продуктов: {e}")
        return "Информация о продуктах недоступна."


def format_portrait_for_prompt(portrait: Dict[str, Any]) -> str:
    """
    Форматирует портрет пользователя для включения в промпт LLM.
    
    :param portrait: Словарь с портретом пользователя
    :return: Отформатированная строка с информацией о пользователе
    """
    if portrait is None:
        return "Информация о пользователе недоступна."
    
    # Форматируем основные характеристики
    lines = [
        "=== ПОРТРЕТ ПОЛЬЗОВАТЕЛЯ ===",
        f"ID пользователя: {portrait.get('user_id', 'N/A')}",
        f"Регион: {portrait.get('region', 'N/A')}",
        f"Социально-демографический кластер: {portrait.get('socdem_cluster', 'N/A')}",
        "",
        "--- АКТИВНОСТЬ ---",
        f"Всего событий: {portrait.get('total_events', 0):,}",
        f"Дней активности: {portrait.get('activity_days', 0):.1f}",
        f"Событий в день: {portrait.get('events_per_day', 0):.2f}",
        "",
        "--- ВОРОНКА КОНВЕРСИИ ---",
        f"Просмотров: {portrait.get('view_count', 0):,}",
        f"Кликов: {portrait.get('click_count', 0):,}",
        f"Покупок: {portrait.get('purchase_count', 0):,}",
        f"Конверсия view→click: {portrait.get('view_to_click_rate', 0):.4f}",
        f"Конверсия click→purchase: {portrait.get('click_to_purchase_rate', 0):.4f}",
        "",
        "--- ФИНАНСОВЫЕ ПОКАЗАТЕЛИ ---",
        f"Общие траты: {portrait.get('total_spent', 0):.2f} ₽",
        f"Средний чек: {portrait.get('avg_purchase', 0):.2f} ₽",
        f"Стд. отклонение чека: {portrait.get('std_purchase', 0):.2f} ₽",
        "",
        "--- РАЗНООБРАЗИЕ ---",
        f"Уникальных категорий: {portrait.get('unique_categories', 0)}",
        f"Топ категория: {portrait.get('top_category', 'N/A')}",
        f"Уникальных брендов: {portrait.get('unique_brands', 0)}",
        f"Уникальных каналов: {portrait.get('unique_channels', 0)}",
        f"Мультиканальность: {'Да' if portrait.get('is_multi_channel', False) else 'Нет'}",
        f"Предпочитаемый канал: {portrait.get('preferred_channel', 'N/A')}",
        "",
        "--- ВРЕМЕННЫЕ ПАТТЕРНЫ ---",
        f"Средний час активности: {portrait.get('avg_hour', 12):.1f}",
        f"Ночная активность: {portrait.get('night_activity_ratio', 0):.2%}",
        "",
        "--- ЦЕНОВЫЕ ПРЕДПОЧТЕНИЯ ---",
        f"Средний интерес к цене: {portrait.get('avg_price_interest', 0):.2f} ₽",
        f"Диапазон цен: {portrait.get('price_range', 0):.2f} ₽",
        f"Мин. цена интереса: {portrait.get('min_price_interest', 0):.2f} ₽",
        f"Макс. цена интереса: {portrait.get('max_price_interest', 0):.2f} ₽",
    ]
    
    return "\n".join(lines)


def create_recommendation_prompt(portrait_text: str, products_text: str, cluster_text: str="") -> str:
    """
    Создает промпт для LLM с портретом пользователя и информацией о продуктах.
    
    :param portrait_text: Отформатированный текст портрета пользователя
    :param products_text: Текст с информацией о продуктах
    :return: Полный промпт для LLM
    """
    prompt = f"""Ты - эксперт по банковским продуктам и персональным рекомендациям.

Твоя задача: проанализировать портрет клиента и рекомендовать ему наиболее подходящие банковские продукты из доступного списка.

ПОРТРЕТ КЛИЕНТА:
{portrait_text}

ДОСТУПНЫЕ ПРОДУКТЫ:
{products_text}

ПОРТРЕТ СОЦИАЛЬНО-ДЕМОГРАФИЧЕСКОЙ ГРУППЫ, В КОТОРУЮ ВХОДИТ КЛИЕНТ:
{cluster_text}

ИНСТРУКЦИИ:
1. Проанализируй портрет клиента и его социально-демографическую группу, определи его финансовые потребности, предпочтения, сформулируй ожидаемые выводы и оптимальную маркетинговую стратегию взаимодействия для банка
2. Выбери ТОП-5 наиболее подходящих продуктов из списка доступных
3. Для каждого продукта укажи:
   - Название продукта
   - Краткое обоснование, почему этот продукт подходит клиенту
   - Ключевые преимущества для данного клиента
   - Маркетинговую стратегию банка, которая обеспечит максимальную эффективность продажи этого продукта для этого человека
   - Ожидаемые последствия продажи продукта: приблизительная количественная оценка потенциального финансового эффекта в российских реалиях и роста лояльности клиента
4. Расположи рекомендации по приоритету (от наиболее подходящего к менее подходящему)
5. Будь конкретным и используй данные из портрета для обоснования
6. Избегай нелогичных или противоречивых предложений

ФОРМАТ ОТВЕТА (JSON):
{{
  "recommendations": [
    {{
      "product_name": "Название продукта",
      "priority": 1,
      "reasoning": "Почему продукт подходит клиенту",
      "key_benefits": ["Преимущество 1", "Преимущество 2"],
      "marketing_strategy": "Маркетинговая стратегия",
      "consequences": "Ожидаемые результаты",
      "match_score": 0.85
    }}
  ],
  "summary": "Краткое резюме рекомендаций"
}}

Важно: верни ТОЛЬКО валидный JSON, без дополнительных комментариев до или после JSON."""
    
    return prompt


def _get_llm() -> Optional[YandexApiLlm]:
    """
    Получает или создает глобальный экземпляр YandexApiLlm.
    
    :return: YandexApiLlm или None в случае ошибки
    """
    global _llm
    
    if _llm is None:
        try:
            # Получаем конфигурацию из переменных окружения
            folder_id = os.environ.get('YANDEX_GPT_FOLDER_ID')
            api_key = os.environ.get('YANDEX_GPT_API_KEY')
            
            if not folder_id or not api_key:
                logger.error("YANDEX_GPT_FOLDER_ID или YANDEX_GPT_API_KEY не установлены")
                return None
            
            config = YandexApiConfig(folder_id=folder_id, api_key=api_key)
            _llm = YandexApiLlm(config, timeout=120)
            logger.info("YandexApiLlm initialized")
        except Exception as e:
            logger.error(f"Failed to initialize YandexApiLlm: {e}", exc_info=True)
            return None
    
    return _llm


def call_llm_api(prompt: str) -> Optional[str]:
    """
    Вызывает LLM напрямую через Yandex GPT API.
    
    :param prompt: Промпт для LLM
    :return: Ответ от LLM или None в случае ошибки
    """
    llm = _get_llm()
    if llm is None:
        logger.error("Failed to get YandexApiLlm")
        return None
    
    # Контекст для LLM (инструкции)
    context = "Ты - эксперт по банковским продуктам. Твоя задача - давать персонализированные рекомендации на основе анализа портрета клиента. Всегда отвечай ТОЛЬКО в формате JSON, без дополнительного текста."
    
    try:
        logger.info(f"RecSys: Отправка запроса в Yandex GPT API (длина промпта: {len(prompt)} символов)")
        response = llm.invoke(context=context, question=prompt)
        if response:
            logger.info(f"RecSys: Получен ответ от модели LLM (длина: {len(response)} символов)")
            logger.debug(f"RecSys: Полный ответ от модели: {response}")
        else:
            logger.error(f"RecSys: llm.invoke вернул None")
        return response
    except Exception as e:
        logger.error(f"RecSys: Исключение при вызове LLM: {e}", exc_info=True)
        return None


def parse_llm_response(response: str) -> Optional[RecommendationsResponse]:
    """
    Парсит ответ от LLM и валидирует его по схеме.
    
    :param response: Ответ от LLM (должен быть JSON)
    :return: Валидированный объект RecommendationsResponse или None в случае ошибки
    """
    if not response:
        return None
    
    try:
        response = response.strip()
        
        if not response.startswith('{'):
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                response = response[start_idx:end_idx]
        
        response_dict = json.loads(response)
        
        # Валидируем по схеме Pydantic
        recommendations = RecommendationsResponse(**response_dict)
        return recommendations
        
    except json.JSONDecodeError as e:
        logger.error(f"Ошибка при парсинге JSON ответа: {e}")
        logger.debug(f"Ответ от LLM: {response[:500]}...")
        return None
    except Exception as e:
        logger.error(f"Ошибка при валидации ответа по схеме: {e}")
        logger.debug(f"Ответ от LLM: {response[:500]}...")
        return None


def generate_recommendations_with_llm(
    portrait: Dict[str, Any],
    products_path: Optional[str] = None
) -> Optional[RecommendationsResponse]:
    """
    Генерирует рекомендации продуктов с использованием LLM.
    
    :param portrait: Портрет пользователя
    :param products_path: Путь к файлу с информацией о продуктах
    :return: Валидированный объект RecommendationsResponse или None в случае ошибки
    """
    products_text = load_products_info(products_path)
    
    portrait_text = format_portrait_for_prompt(portrait)
    

    cluster_text = portrait.get('cluster_description', '')
    if not cluster_text:
        cluster_text = "Информация о социально-демографическом кластере недоступна."
    
    prompt = create_recommendation_prompt(portrait_text, products_text, cluster_text)
    
    logger.info(f"RecSys: Вызов LLM для генерации рекомендаций (длина промпта: {len(prompt)} символов)")
    response = call_llm_api(prompt)
    
    if not response:
        logger.error("RecSys: ❌ Не удалось получить ответ от LLM")
        return None
    
    logger.info(f"RecSys: Ответ от модели успешно получен, начинаем парсинг (длина ответа: {len(response)} символов)")
    recommendations = parse_llm_response(response)
    
    if recommendations:
        logger.info(f"RecSys: Рекомендации успешно распарсены. Количество рекомендаций: {len(recommendations.recommendations)}")
    else:
        logger.warning("RecSys: Не удалось распарсить ответ от модели. Возможно, ответ не в формате JSON или не соответствует схеме.")
        logger.debug(f"RecSys: Сырой ответ для анализа: {response[:500]}...")
    
    return recommendations


def print_recommendations(recommendations: Optional[RecommendationsResponse]) -> None:
    """
    Красиво выводит рекомендации в консоль.
    
    :param recommendations: Объект RecommendationsResponse с рекомендациями
    """
    if not recommendations:
        print("❌ Рекомендации не найдены")
        return
    
    print(f"\n{'='*60}")
    print("💡 РЕКОМЕНДАЦИИ ПРОДУКТОВ (LLM)")
    print(f"{'='*60}\n")
    
    if not recommendations.recommendations:
        print("⚠️  Рекомендации отсутствуют")
        return
    
    for i, rec in enumerate(recommendations.recommendations, 1):
        print(f"{i}. {rec.product_name}")
        print(f"   Приоритет: {rec.priority} | Совпадение: {rec.match_score:.2%}")
        print(f"   Обоснование: {rec.reasoning}")
        
        if rec.key_benefits:
            print(f"   Ключевые преимущества:")
            for benefit in rec.key_benefits:
                print(f"     • {benefit}")
        print("")
    
    if recommendations.summary:
        print(f"📋 Резюме: {recommendations.summary}")
        print("")


def save_recommendations_to_json(
    recommendations: Optional[RecommendationsResponse],
    output_path: str
) -> None:
    """
    Сохраняет рекомендации в JSON файл.
    
    :param recommendations: Объект RecommendationsResponse с рекомендациями
    :param output_path: Путь для сохранения файла
    """
    if not recommendations:
        return
    
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Используем model_dump для сериализации Pydantic модели
            json.dump(recommendations.model_dump(), f, indent=2, ensure_ascii=False, default=str)
        
    except Exception as e:
        logger.error(f"Ошибка при сохранении рекомендаций: {e}")

