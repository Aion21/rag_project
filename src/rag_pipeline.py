import logging
from typing import List, Optional
from pathlib import Path

from src.document_loader import DocumentLoader
from src.vector_store import VectorStore
from src.llm_handler import LLMHandler
from config.settings import DATA_PATH, SEARCH_K, SIMILARITY_THRESHOLD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self):
        """
        Инициализация RAG пайплайна
        """
        self.document_loader = DocumentLoader()
        self.vector_store = VectorStore()
        self.llm_handler = LLMHandler()

        logger.info("RAG Pipeline инициализирован")

    def load_documents(self, data_path: str = DATA_PATH) -> dict:
        """
        Загружает документы из указанной папки в векторную базу данных
        """
        try:
            logger.info(f"Начинаем загрузку документов из: {data_path}")

            # Проверяем существование папки
            if not Path(data_path).exists():
                return {
                    "success": False,
                    "message": f"Папка {data_path} не существует",
                    "documents_loaded": 0
                }

            # Загружаем документы
            documents = self.document_loader.load_documents_from_directory(data_path)

            if not documents:
                return {
                    "success": False,
                    "message": "Не найдено документов для загрузки",
                    "documents_loaded": 0
                }

            # Добавляем в векторное хранилище
            self.vector_store.add_documents(documents)

            # Получаем информацию о коллекции
            collection_info = self.vector_store.get_collection_info()

            return {
                "success": True,
                "message": f"Успешно загружено {len(documents)} документов/чанков",
                "documents_loaded": len(documents),
                "total_in_collection": collection_info.get("document_count", 0)
            }

        except Exception as e:
            logger.error(f"Ошибка при загрузке документов: {e}")
            return {
                "success": False,
                "message": f"Ошибка при загрузке: {str(e)}",
                "documents_loaded": 0
            }

    def query(self, user_question: str, k: int = SEARCH_K) -> str:
        """
        Обрабатывает пользовательский запрос через RAG пайплайн
        """
        try:
            if not user_question.strip():
                return "Пожалуйста, задайте вопрос."

            logger.info(f"Обрабатываем запрос: {user_question}")

            # Проверяем, является ли это общим приветствием или вопросом
            general_queries = self._is_general_query(user_question.lower())

            # Поиск релевантных документов
            relevant_docs = self.vector_store.search_similar_documents(
                query=user_question,
                k=k
            )

            # ОТЛАДКА - логируем найденные документы
            logger.info(f"Найдено документов: {len(relevant_docs)}")
            for i, (doc, score) in enumerate(relevant_docs):
                logger.info(f"Документ {i + 1}: {doc.metadata.get('filename', 'unknown')} (score: {score:.3f})")

            # Если это общий вопрос и нет релевантных документов, отвечаем дружелюбно
            if general_queries and (
                    not relevant_docs or not any(score >= SIMILARITY_THRESHOLD for _, score in relevant_docs)):
                return self._handle_general_query(user_question)

            if not relevant_docs:
                return "Я специализируюсь на ответах по документам в вашей базе знаний. Попробуйте задать вопрос о содержимом ваших документов."

            # Фильтруем документы по порогу релевантности
            filtered_docs = [
                (doc, score) for doc, score in relevant_docs
                if score >= SIMILARITY_THRESHOLD
            ]

            # ОТЛАДКА - логируем фильтрацию
            logger.info(f"Порог релевантности: {SIMILARITY_THRESHOLD}")
            logger.info(f"Документов после фильтрации: {len(filtered_docs)}")
            for i, (doc, score) in enumerate(filtered_docs):
                logger.info(f"Прошел фильтр {i + 1}: {doc.metadata.get('filename', 'unknown')} (score: {score:.3f})")

            if not filtered_docs:
                # Если документы есть, но не релевантны - даем дружелюбный ответ
                logger.warning(
                    f"Все документы отфильтрованы! Лучший score: {relevant_docs[0][1] if relevant_docs else 'N/A'}")
                return "Я не нашел достаточно релевантной информации в ваших документах для этого вопроса. Попробуйте переформулировать запрос или задать более конкретный вопрос по содержимому ваших файлов."

            # ОТЛАДКА - показываем какие документы передаем в LLM
            logger.info(f"Передаем в LLM {len(filtered_docs)} отфильтрованных документов:")
            for i, (doc, score) in enumerate(filtered_docs, 1):
                filename = doc.metadata.get('filename', 'unknown')
                logger.info(f"  {i}. {filename} (score: {score:.3f})")

            # ВАЖНО: передаем ТОЛЬКО отфильтрованные документы, НЕ все relevant_docs
            response = self.llm_handler.generate_response(
                query=user_question,
                relevant_docs=filtered_docs  # ТОЛЬКО релевантные!
            )

            return response

        except Exception as e:
            logger.error(f"Ошибка при обработке запроса: {e}")
            return "Извините, произошла ошибка при обработке вашего запроса. Попробуйте еще раз."

    def query_stream(self, user_question: str, k: int = SEARCH_K):
        """
        Обрабатывает пользовательский запрос через RAG пайплайн в потоковом режиме
        """
        try:
            if not user_question.strip():
                yield "Пожалуйста, задайте вопрос."
                return

            logger.info(f"Обрабатываем потоковый запрос: {user_question}")

            # Проверяем, является ли это общим приветствием или вопросом
            general_queries = self._is_general_query(user_question.lower())

            # Поиск релевантных документов
            relevant_docs = self.vector_store.search_similar_documents(
                query=user_question,
                k=k
            )

            # Если это общий вопрос и нет релевантных документов, отвечаем дружелюбно
            if general_queries and (
                    not relevant_docs or not any(score >= SIMILARITY_THRESHOLD for _, score in relevant_docs)):
                yield self._handle_general_query(user_question)
                return

            if not relevant_docs:
                yield "Я специализируюсь на ответах по документам в вашей базе знаний. Попробуйте задать вопрос о содержимом ваших документов."
                return

            # Фильтруем документы по порогу релевантности
            filtered_docs = [
                (doc, score) for doc, score in relevant_docs
                if score >= SIMILARITY_THRESHOLD
            ]

            if not filtered_docs:
                yield "Я не нашел достаточно релевантной информации в ваших документах для этого вопроса. Попробуйте переформулировать запрос или задать более конкретный вопрос по содержимому ваших файлов."
                return

            # Генерируем ответ с помощью LLM в потоковом режиме
            for response_chunk in self.llm_handler.generate_response_stream(
                    query=user_question,
                    relevant_docs=filtered_docs
            ):
                yield response_chunk

        except Exception as e:
            logger.error(f"Ошибка при потоковой обработке запроса: {e}")
            yield "Извините, произошла ошибка при обработке вашего запроса. Попробуйте еще раз."

    def get_system_status(self) -> dict:
        """
        Получает статус системы
        """
        try:
            # Информация о коллекции
            collection_info = self.vector_store.get_collection_info()

            # Проверка подключения к OpenAI
            openai_status = self.llm_handler.check_connection()

            # Проверка существования папки с данными
            data_path_exists = Path(DATA_PATH).exists()

            return {
                "vector_store": {
                    "status": "✅ Работает",
                    "collection_name": collection_info.get("name", "Неизвестно"),
                    "documents_count": collection_info.get("document_count", 0),
                    "embedding_model": collection_info.get("embedding_model", "Неизвестно")
                },
                "llm": {
                    "status": "✅ Работает" if openai_status else "❌ Ошибка подключения",
                    "model": self.llm_handler.llm.model_name if hasattr(self.llm_handler.llm,
                                                                        'model_name') else "Неизвестно"
                },
                "data_path": {
                    "path": DATA_PATH,
                    "exists": "✅ Существует" if data_path_exists else "❌ Не найдена",
                    "files_count": len(list(Path(DATA_PATH).rglob("*"))) if data_path_exists else 0
                }
            }

        except Exception as e:
            logger.error(f"Ошибка при получении статуса системы: {e}")
            return {
                "error": f"Ошибка при получении статуса: {str(e)}"
            }

    def clear_vector_store(self) -> dict:
        """
        Очищает векторное хранилище
        """
        try:
            self.vector_store.clear_collection()
            return {
                "success": True,
                "message": "Векторное хранилище очищено"
            }
        except Exception as e:
            logger.error(f"Ошибка при очистке векторного хранилища: {e}")
            return {
                "success": False,
                "message": f"Ошибка при очистке: {str(e)}"
            }

    def search_documents(self, query: str, k: int = SEARCH_K) -> List[dict]:
        """
        Поиск документов без генерации ответа (для отладки)
        """
        try:
            relevant_docs = self.vector_store.search_similar_documents(query, k)

            results = []
            for doc, score in relevant_docs:
                results.append({
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "score": round(score, 3),
                    "filename": doc.metadata.get("filename", "Неизвестно"),
                    "directory": doc.metadata.get("directory", ""),
                    "chunk_index": doc.metadata.get("chunk_index", 0)
                })

            return results

        except Exception as e:
            logger.error(f"Ошибка при поиске документов: {e}")
            return []

    def _is_general_query(self, query: str) -> bool:
        """
        Определяет, является ли запрос общим приветствием или вопросом
        """
        general_patterns = [
            'привет', 'hello', 'hi', 'здравствуй', 'добрый день', 'добрый вечер',
            'как дела', 'что умеешь', 'помоги', 'кто ты', 'что ты',
            'спасибо', 'thanks', 'thank you', 'пока', 'bye', 'до свидания'
        ]

        return any(pattern in query for pattern in general_patterns)

    def _handle_general_query(self, query: str) -> str:
        """
        Обрабатывает общие запросы без поиска в документах
        """
        query_lower = query.lower()

        if any(word in query_lower for word in ['привет', 'hello', 'hi', 'здравствуй', 'добрый']):
            return """Привет! 👋 

Я ваш ИИ-ассистент для работы с документами. Я могу помочь вам найти информацию в загруженных файлах и ответить на вопросы по их содержимому.

Что я умею:
• Отвечать на вопросы по содержимому ваших документов
• Искать конкретную информацию в файлах
• Анализировать и сравнивать данные из разных документов

Задайте мне вопрос о содержимом ваших файлов, и я постараюсь помочь!"""

        elif any(word in query_lower for word in ['что умеешь', 'что ты', 'кто ты', 'помоги']):
            return """Я ваш персональный ассистент для работы с документами! 🤖

Мои возможности:
• 📚 Анализ содержимого ваших документов (TXT, PDF, DOCX, CSV, MD)
• 🔍 Быстрый поиск информации по всей базе знаний
• 💡 Ответы на вопросы на основе загруженных файлов
• 📊 Сравнение и анализ данных из разных документов

Просто задайте вопрос о содержимом ваших файлов, например:
"Расскажи об Александре Струнникове" или "Какие продукты у нас есть?" """

        elif any(word in query_lower for word in ['спасибо', 'thanks']):
            return "Пожалуйста! Рад был помочь. Если у вас есть еще вопросы по документам, обращайтесь! 😊"

        elif any(word in query_lower for word in ['пока', 'bye', 'до свидания']):
            return "До свидания! Удачи в работе с документами! 👋"

        else:
            return "Я готов помочь вам с поиском информации в ваших документах. Задайте конкретный вопрос по содержимому файлов!"