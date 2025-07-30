import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# ChromaDB settings
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "documents"

# Document processing settings
CHUNK_SIZE = 1000  # Text chunk size
CHUNK_OVERLAP = 200  # Overlap between chunks

# Search settings
SEARCH_K = 5  # Number of similar documents to retrieve
SIMILARITY_THRESHOLD = 0.20

# Embedding settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Supported file formats
SUPPORTED_EXTENSIONS = ['.txt', '.pdf', '.docx', '.md', '.csv']

# Path to data
DATA_PATH = "./data"

# Gradio settings
GRADIO_SHARE = False
GRADIO_PORT = 7860

# Logging settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")