import re
import string
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextProcessor:
    """
    Утилиты для обработки текста
    """

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Очищает текст от лишних символов и нормализует
        """
        if not text:
            return ""

        # Удаляем лишние пробелы и переносы строк
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        # Удаляем специальные символы, но оставляем знаки препинания
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\"\']+', '', text)

        return text

    @staticmethod
    def extract_keywords(text: str, min_length: int = 3) -> List[str]:
        """
        Извлекает ключевые слова из текста
        """
        if not text:
            return []

        # Приводим к нижнему регистру и разбиваем на слова
        words = re.findall(r'\b\w+\b', text.lower())

        # Фильтруем по длине
        keywords = [word for word in words if len(word) >= min_length]

        # Удаляем дубликаты, сохраняя порядок
        seen = set()
        unique_keywords = []
        for word in keywords:
            if word not in seen:
                seen.add(word)
                unique_keywords.append(word)

        return unique_keywords

    @staticmethod
    def get_text_stats(text: str) -> Dict[str, int]:
        """
        Получает статистику текста
        """
        if not text:
            return {
                "characters": 0,
                "words": 0,
                "sentences": 0,
                "paragraphs": 0
            }

        # Подсчет символов
        characters = len(text)

        # Подсчет слов
        words = len(re.findall(r'\b\w+\b', text))

        # Подсчет предложений
        sentences = len(re.findall(r'[.!?]+', text))

        # Подсчет абзацев
        paragraphs = len([p for p in text.split('\n\n') if p.strip()])

        return {
            "characters": characters,
            "words": words,
            "sentences": sentences,
            "paragraphs": paragraphs
        }

    @staticmethod
    def truncate_text(text: str, max_length: int = 1000, add_ellipsis: bool = True) -> str:
        """
        Обрезает текст до указанной длины
        """
        if not text or len(text) <= max_length:
            return text

        truncated = text[:max_length]

        if add_ellipsis:
            # Находим последний пробел, чтобы не разрывать слова
            last_space = truncated.rfind(' ')
            if last_space > max_length * 0.8:  # Если пробел не слишком далеко
                truncated = truncated[:last_space]
            truncated += "..."

        return truncated

    @staticmethod
    def highlight_keywords(text: str, keywords: List[str],
                           start_tag: str = "**", end_tag: str = "**") -> str:
        """
        Выделяет ключевые слова в тексте
        """
        if not text or not keywords:
            return text

        result = text
        for keyword in keywords:
            # Создаем паттерн для поиска слова как отдельного слова
            pattern = r'\b' + re.escape(keyword) + r'\b'
            replacement = f"{start_tag}{keyword}{end_tag}"
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        return result

    @staticmethod
    def similarity_score(text1: str, text2: str) -> float:
        """
        Простой расчет схожести текстов на основе общих слов
        """
        if not text1 or not text2:
            return 0.0

        # Получаем множества слов для каждого текста
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))

        if not words1 or not words2:
            return 0.0

        # Вычисляем коэффициент Жаккара
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0