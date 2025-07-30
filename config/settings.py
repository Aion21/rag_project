import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI настройки
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # Изменено с gpt-3.5-turbo на gpt-4o-mini

# ChromaDB настройки
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "documents"

# Настройки обработки документов
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Настройки поиска
SEARCH_K = 5  # Количество похожих документов для поиска
SIMILARITY_THRESHOLD = 0.15  # Повышен до 0.25 - только действительно релевантные документы

# Настройки эмбеддингов
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Поддерживаемые форматы файлов
SUPPORTED_EXTENSIONS = ['.txt', '.pdf', '.docx', '.md', '.csv']

# Путь к данным
DATA_PATH = "./data"

# Настройки Gradio
GRADIO_SHARE = False
GRADIO_PORT = 7860