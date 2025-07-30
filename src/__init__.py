from .document_loader import DocumentLoader
from .vector_store import VectorStore
from .llm_handler import LLMHandler
from .rag_pipeline import RAGPipeline

__version__ = "1.0.0"
__author__ = "RAG System"

__all__ = [
    'DocumentLoader',
    'VectorStore',
    'LLMHandler',
    'RAGPipeline'
]

COMPONENT_VERSIONS = {
    'document_loader': '1.0.0',
    'vector_store': '1.0.0',
    'llm_handler': '1.0.0',
    'rag_pipeline': '1.0.0'
}