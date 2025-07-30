import logging
from typing import List, Tuple
import chromadb
from chromadb.config import Settings
from langchain.schema import Document
from sentence_transformers import SentenceTransformer

from config.settings import CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL, SEARCH_K

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self):
        """
        Инициализация векторного хранилища ChromaDB
        """
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        # Настройка ChromaDB
        self.client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Создание или получение коллекции
        try:
            self.collection = self.client.get_collection(COLLECTION_NAME)
            logger.info(f"Коллекция '{COLLECTION_NAME}' найдена")
        except:
            self.collection = self.client.create_collection(
                name=COLLECTION_NAME,
                metadata={"description": "RAG документы"}
            )
            logger.info(f"Создана новая коллекция '{COLLECTION_NAME}'")

    def add_documents(self, documents: List[Document]) -> None:
        """
        Добавляет документы в векторное хранилище
        """
        if not documents:
            logger.warning("Нет документов для добавления")
            return

        logger.info(f"Добавляем {len(documents)} документов в векторное хранилище...")

        # Подготовка данных для ChromaDB
        texts = []
        metadatas = []
        ids = []

        for i, doc in enumerate(documents):
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
            ids.append(f"doc_{i}_{hash(doc.page_content)}")

        # Создание эмбеддингов
        logger.info("Создание эмбеддингов...")
        embeddings = self.embedding_model.encode(texts).tolist()

        # Добавление в ChromaDB
        try:
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )
            logger.info(f"✅ Успешно добавлено {len(documents)} документов")
        except Exception as e:
            logger.error(f"Ошибка при добавлении документов: {e}")
            raise

    def search_similar_documents(self, query: str, k: int = SEARCH_K) -> List[Tuple[Document, float]]:
        """
        Поиск похожих документов по запросу
        """
        try:
            # Создание эмбеддинга для запроса
            query_embedding = self.embedding_model.encode([query]).tolist()

            # Поиск в ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=k,
                include=['documents', 'metadatas', 'distances']
            )

            # Преобразование результатов
            similar_docs = []
            if results['documents'] and results['documents'][0]:
                logger.info(f"Обрабатываем {len(results['documents'][0])} результатов поиска")

                for i in range(len(results['documents'][0])):
                    doc = Document(
                        page_content=results['documents'][0][i],
                        metadata=results['metadatas'][0][i]
                    )
                    # ChromaDB возвращает cosine distance
                    distance = results['distances'][0][i]

                    # Отладка - логируем расстояния
                    filename = results['metadatas'][0][i].get('filename', 'unknown')
                    logger.info(f"Документ {filename}: distance={distance:.4f}")

                    # Исправленное преобразование cosine distance в similarity
                    # Для cosine distance обычно от 0 (идентичные) до 2 (противоположные)
                    # Но ChromaDB может возвращать нормализованные значения от 0 до 1

                    if distance <= 1.0:
                        # Если distance от 0 до 1, то similarity = 1 - distance
                        similarity = 1.0 - distance
                    else:
                        # Если distance больше 1, нормализуем: similarity = 1 - (distance / 2)
                        similarity = max(0.0, 1.0 - (distance / 2.0))

                    # Дополнительная проверка: если similarity все еще 0, используем инвертированный distance
                    if similarity == 0.0 and distance > 0:
                        similarity = 1.0 / (1.0 + distance)

                    logger.info(f"Документ {filename}: distance={distance:.4f} -> similarity={similarity:.4f}")

                    similar_docs.append((doc, similarity))

            logger.info(f"Найдено {len(similar_docs)} похожих документов")
            return similar_docs

        except Exception as e:
            logger.error(f"Ошибка при поиске документов: {e}")
            return []

    def get_collection_info(self) -> dict:
        """
        Получение информации о коллекции
        """
        try:
            count = self.collection.count()
            return {
                "name": COLLECTION_NAME,
                "document_count": count,
                "embedding_model": EMBEDDING_MODEL
            }
        except Exception as e:
            logger.error(f"Ошибка при получении информации о коллекции: {e}")
            return {}

    def clear_collection(self) -> None:
        """
        Очистка коллекции
        """
        try:
            # Удаляем текущую коллекцию
            self.client.delete_collection(COLLECTION_NAME)
            # Создаем новую пустую коллекцию
            self.collection = self.client.create_collection(
                name=COLLECTION_NAME,
                metadata={"description": "RAG документы"}
            )
            logger.info("Коллекция очищена")
        except Exception as e:
            logger.error(f"Ошибка при очистке коллекции: {e}")

    def document_exists(self, doc_hash: str) -> bool:
        """
        Проверка существования документа по хешу
        """
        try:
            results = self.collection.get(ids=[f"doc_hash_{doc_hash}"])
            return len(results['ids']) > 0
        except:
            return False