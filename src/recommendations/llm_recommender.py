"""
Модуль для генерации рекомендаций продуктов с использованием LLM.

Использует портрет пользователя и информацию о продуктах для генерации
персонализированных рекомендаций через LLM API.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import os

logger = logging.getLogger(__name__)


def load_products_info(products_path: Optional[str] = None) -> str:
    """
    Загружает информацию о продуктах из файла.
    
    :param products_path: Путь к файлу с информацией о продуктах
    :return: Текст с информацией о продуктах
    """
    if products_path is None:
        # Путь по умолчанию
        project_root = Path(__file__).parent.parent.parent
        products_path = project_root / "research" / "psb_products.md"
    
    products_path = Path(products_path)
    
    if not products_path.exists():
        logger.warning(f"Файл с продуктами не найден: {products_path}")
        return "Информация о продуктах недоступна."
    
    try:
        with open(products_path, 'r', encoding='utf-8') as f:
            products_text = f.read()
        logger.info(f"✅ Загружена информация о продуктах из {products_path}")
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


def create_recommendation_prompt(portrait_text: str, products_text: str) -> str:
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

ИНСТРУКЦИИ:
1. Проанализируй портрет клиента и определи его финансовые потребности и предпочтения
2. Выбери ТОП-5 наиболее подходящих продуктов из списка доступных
3. Для каждого продукта укажи:
   - Название продукта
   - Краткое обоснование, почему этот продукт подходит клиенту
   - Ключевые преимущества для данного клиента
4. Расположи рекомендации по приоритету (от наиболее подходящего к менее подходящему)
5. Будь конкретным и используй данные из портрета для обоснования

ФОРМАТ ОТВЕТА (JSON):
{{
  "recommendations": [
    {{
      "product_name": "Название продукта",
      "priority": 1,
      "reasoning": "Почему продукт подходит клиенту",
      "key_benefits": ["Преимущество 1", "Преимущество 2"],
      "match_score": 0.85
    }}
  ],
  "summary": "Краткое резюме рекомендаций"
}}

Важно: верни ТОЛЬКО валидный JSON, без дополнительных комментариев до или после JSON."""
    
    return prompt


def call_llm_api(prompt: str, api_key: Optional[str] = None, model: str = "gpt-4o-mini", base_url: str = "https://api.openai.com/v1") -> Optional[str]:
    """
    Вызывает LLM API для генерации рекомендаций.
    
    Поддерживает OpenAI API и OpenRouter API. Для других провайдеров можно расширить.
    
    :param prompt: Промпт для LLM
    :param api_key: API ключ (если None, берется из переменной окружения)
    :param model: Модель для использования
    :param base_url: Базовый URL для API (по умолчанию OpenAI, можно указать OpenRouter)
    :return: Ответ от LLM или None в случае ошибки
    """
    try:
        from openai import OpenAI
        
        if not api_key:
            if "openrouter" in base_url.lower():
                api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
            else:
                api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        
        if not api_key:
            logger.error("API ключ не найден. Установите OPENAI_API_KEY или OPENROUTER_API_KEY в переменных окружения.")
            return None
        
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        logger.info(f"🤖 Отправка запроса в LLM (модель: {model}, base_url: {base_url})...")
        
        messages = [
            {
                "role": "system",
                "content": "Ты - эксперт по банковским продуктам. Твоя задача - давать персонализированные рекомендации на основе анализа портрета клиента. Всегда отвечай ТОЛЬКО в формате JSON, без дополнительного текста."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        is_openrouter = "openrouter" in base_url.lower()
        
        try:
            if is_openrouter:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.7
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.7,
                    response_format={"type": "json_object"}
                )
        except Exception as e:
            if not is_openrouter:
                logger.warning(f"Попытка без response_format из-за ошибки: {e}")
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.7
                )
            else:
                raise
        
        result = response.choices[0].message.content
        logger.info("✅ Получен ответ от LLM")
        return result
        
    except ImportError:
        logger.error("Библиотека openai не установлена. Установите: pip install openai")
        return None
    except Exception as e:
        logger.error(f"Ошибка при вызове LLM API: {e}", exc_info=True)
        return None


def parse_llm_response(response: str) -> Optional[Dict[str, Any]]:
    """
    Парсит ответ от LLM и извлекает рекомендации.
    
    :param response: Ответ от LLM (должен быть JSON)
    :return: Словарь с рекомендациями или None в случае ошибки
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
        
        recommendations = json.loads(response)
        logger.info("✅ Рекомендации успешно распарсены")
        return recommendations
        
    except json.JSONDecodeError as e:
        logger.error(f"Ошибка при парсинге JSON ответа: {e}")
        logger.debug(f"Ответ LLM: {response[:500]}...")
        return None
    except Exception as e:
        logger.error(f"Неожиданная ошибка при парсинге ответа: {e}")
        return None


def generate_recommendations_with_llm(
    portrait: Dict[str, Any],
    products_path: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    base_url: str = "https://api.openai.com/v1"
) -> Optional[Dict[str, Any]]:
    """
    Генерирует рекомендации продуктов с использованием LLM.
    
    :param portrait: Портрет пользователя
    :param products_path: Путь к файлу с информацией о продуктах
    :param api_key: API ключ для LLM (если None, берется из переменной окружения)
    :param model: Модель LLM для использования
    :return: Словарь с рекомендациями или None в случае ошибки
    """
    logger.info("🎯 Генерация рекомендаций с использованием LLM...")
    
    products_text = load_products_info(products_path)
    
    portrait_text = format_portrait_for_prompt(portrait)
    
    prompt = create_recommendation_prompt(portrait_text, products_text)
    
    response = call_llm_api(prompt, api_key, model, base_url)
    
    if not response:
        logger.error("❌ Не удалось получить ответ от LLM")
        return None
    
    recommendations = parse_llm_response(response)
    
    if recommendations:
        logger.info(f"✅ Сгенерировано {len(recommendations.get('recommendations', []))} рекомендаций")
    
    return recommendations


def print_recommendations(recommendations: Optional[Dict[str, Any]]) -> None:
    """
    Красиво выводит рекомендации в консоль.
    
    :param recommendations: Словарь с рекомендациями
    """
    if not recommendations:
        print("❌ Рекомендации не найдены")
        return
    
    print(f"\n{'='*60}")
    print("💡 РЕКОМЕНДАЦИИ ПРОДУКТОВ (LLM)")
    print(f"{'='*60}\n")
    
    recs = recommendations.get('recommendations', [])
    
    if not recs:
        print("⚠️  Рекомендации отсутствуют")
        return
    
    for i, rec in enumerate(recs, 1):
        product_name = rec.get('product_name', 'Неизвестный продукт')
        priority = rec.get('priority', i)
        reasoning = rec.get('reasoning', 'Обоснование не указано')
        benefits = rec.get('key_benefits', [])
        match_score = rec.get('match_score', 0)
        
        print(f"{i}. {product_name}")
        print(f"   Приоритет: {priority} | Совпадение: {match_score:.2%}")
        print(f"   Обоснование: {reasoning}")
        
        if benefits:
            print(f"   Ключевые преимущества:")
            for benefit in benefits:
                print(f"     • {benefit}")
        print()
    
    summary = recommendations.get('summary', '')
    if summary:
        print(f"📋 Резюме: {summary}")
        print()


def save_recommendations_to_json(
    recommendations: Optional[Dict[str, Any]],
    output_path: str
) -> None:
    """
    Сохраняет рекомендации в JSON файл.
    
    :param recommendations: Словарь с рекомендациями
    :param output_path: Путь для сохранения файла
    """
    if not recommendations:
        logger.warning("Нечего сохранять: рекомендации отсутствуют")
        return
    
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(recommendations, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"💾 Рекомендации сохранены в {output_path}")
        print(f"💾 Рекомендации сохранены в {output_path}")
        
    except Exception as e:
        logger.error(f"Ошибка при сохранении рекомендаций: {e}")
        print(f"❌ Ошибка при сохранении рекомендаций: {e}")

