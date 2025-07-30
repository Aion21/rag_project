import re
import string
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextProcessor:
    """
    Utils for text processing
    """

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean text
        """
        if not text:
            return ""

        # Delete newlines
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        # Delete special characters
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\"\']+', '', text)

        return text

    @staticmethod
    def extract_keywords(text: str, min_length: int = 3) -> List[str]:
        """
        Get keywords from text
        """
        if not text:
            return []

        # Change to lowercase
        words = re.findall(r'\b\w+\b', text.lower())

        # Filter by length
        keywords = [word for word in words if len(word) >= min_length]

        # Delete duplicates
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
        Get text stats
        """
        if not text:
            return {
                "characters": 0,
                "words": 0,
                "sentences": 0,
                "paragraphs": 0
            }

        # Symbol count
        characters = len(text)

        # Word count
        words = len(re.findall(r'\b\w+\b', text))

        # Sentence count
        sentences = len(re.findall(r'[.!?]+', text))

        # Paragraph count
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
        Cut text to max length
        """
        if not text or len(text) <= max_length:
            return text

        truncated = text[:max_length]

        if add_ellipsis:
            # Find the last space before max_length
            last_space = truncated.rfind(' ')
            if last_space > max_length * 0.8:
                truncated = truncated[:last_space]
            truncated += "..."

        return truncated

    @staticmethod
    def highlight_keywords(text: str, keywords: List[str],
                           start_tag: str = "**", end_tag: str = "**") -> str:
        """
        Highlight keywords in text
        """
        if not text or not keywords:
            return text

        result = text
        for keyword in keywords:
            # Create a regex pattern to match the keyword
            pattern = r'\b' + re.escape(keyword) + r'\b'
            replacement = f"{start_tag}{keyword}{end_tag}"
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        return result

    @staticmethod
    def similarity_score(text1: str, text2: str) -> float:
        """
        Simple similarity score between two texts
        """
        if not text1 or not text2:
            return 0.0

        # Get unique words
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))

        if not words1 or not words2:
            return 0.0

        # Calculate intersection and union
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0