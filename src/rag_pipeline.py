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
        Initialize the RAG pipeline
        """
        self.document_loader = DocumentLoader()
        self.vector_store = VectorStore()
        self.llm_handler = LLMHandler()

        logger.info("RAG Pipeline initialized")

    def load_documents(self, data_path: str = DATA_PATH) -> dict:
        """
        Load documents from specified folder into vector database
        """
        try:
            logger.info(f"Starting document loading from: {data_path}")

            # Check if folder exists
            if not Path(data_path).exists():
                return {
                    "success": False,
                    "message": f"Folder {data_path} does not exist",
                    "documents_loaded": 0
                }

            # Load documents
            documents = self.document_loader.load_documents_from_directory(data_path)

            if not documents:
                return {
                    "success": False,
                    "message": "No documents found for loading",
                    "documents_loaded": 0
                }

            # Add to vector store
            self.vector_store.add_documents(documents)

            # Get collection info
            collection_info = self.vector_store.get_collection_info()

            return {
                "success": True,
                "message": f"Successfully loaded {len(documents)} documents/chunks",
                "documents_loaded": len(documents),
                "total_in_collection": collection_info.get("document_count", 0)
            }

        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            return {
                "success": False,
                "message": f"Loading error: {str(e)}",
                "documents_loaded": 0
            }

    def query(self, user_question: str, k: int = SEARCH_K) -> str:
        """
        Process user query through RAG pipeline
        """
        try:
            if not user_question.strip():
                return "Please ask a question."

            logger.info(f"Processing query: {user_question}")

            # Check if this is a general greeting or question
            general_queries = self._is_general_query(user_question.lower())

            # Search for relevant documents
            relevant_docs = self.vector_store.search_similar_documents(
                query=user_question,
                k=k
            )

            # DEBUG - log found documents
            logger.info(f"Found documents: {len(relevant_docs)}")
            for i, (doc, score) in enumerate(relevant_docs):
                logger.info(f"Document {i + 1}: {doc.metadata.get('filename', 'unknown')} (score: {score:.3f})")

            # If this is a general question and no relevant documents, respond friendly
            if general_queries and (
                    not relevant_docs or not any(score >= SIMILARITY_THRESHOLD for _, score in relevant_docs)):
                return self._handle_general_query(user_question)

            if not relevant_docs:
                return "I specialize in answering questions based on documents in your knowledge base. Try asking a question about the content of your documents."

            # Filter documents by relevance threshold
            filtered_docs = [
                (doc, score) for doc, score in relevant_docs
                if score >= SIMILARITY_THRESHOLD
            ]

            # DEBUG - log filtering
            logger.info(f"Relevance threshold: {SIMILARITY_THRESHOLD}")
            logger.info(f"Documents after filtering: {len(filtered_docs)}")
            for i, (doc, score) in enumerate(filtered_docs):
                logger.info(f"Passed filter {i + 1}: {doc.metadata.get('filename', 'unknown')} (score: {score:.3f})")

            if not filtered_docs:
                # If documents exist but not relevant - give friendly response
                logger.warning(
                    f"All documents filtered out! Best score: {relevant_docs[0][1] if relevant_docs else 'N/A'}")
                return "I didn't find sufficiently relevant information in your documents for this question. Try rephrasing your query or ask a more specific question about the content of your files."

            # DEBUG - show which documents we're passing to LLM
            logger.info(f"Passing to LLM {len(filtered_docs)} filtered documents:")
            for i, (doc, score) in enumerate(filtered_docs, 1):
                filename = doc.metadata.get('filename', 'unknown')
                logger.info(f"  {i}. {filename} (score: {score:.3f})")

            # IMPORTANT: pass ONLY filtered documents, NOT all relevant_docs
            response = self.llm_handler.generate_response(
                query=user_question,
                relevant_docs=filtered_docs  # ONLY relevant!
            )

            return response

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return "Sorry, an error occurred while processing your request. Please try again."

    def query_stream(self, user_question: str, k: int = SEARCH_K):
        """
        Process user query through RAG pipeline in streaming mode
        """
        try:
            if not user_question.strip():
                yield "Please ask a question."
                return

            logger.info(f"Processing streaming query: {user_question}")

            # Check if this is a general greeting or question
            general_queries = self._is_general_query(user_question.lower())

            # Search for relevant documents
            relevant_docs = self.vector_store.search_similar_documents(
                query=user_question,
                k=k
            )

            # If this is a general question and no relevant documents, respond friendly
            if general_queries and (
                    not relevant_docs or not any(score >= SIMILARITY_THRESHOLD for _, score in relevant_docs)):
                yield self._handle_general_query(user_question)
                return

            if not relevant_docs:
                yield "I specialize in answering questions based on documents in your knowledge base. Try asking about the content of your documents."
                return

            # Filter documents by relevance threshold
            filtered_docs = [
                (doc, score) for doc, score in relevant_docs
                if score >= SIMILARITY_THRESHOLD
            ]

            if not filtered_docs:
                yield "I didn't find sufficiently relevant information in your documents for this question. Try rephrasing your query or ask a more specific question about the content of your files."
                return

            # Generate response using LLM in streaming mode
            for response_chunk in self.llm_handler.generate_response_stream(
                    query=user_question,
                    relevant_docs=filtered_docs
            ):
                yield response_chunk

        except Exception as e:
            logger.error(f"Error during streaming query processing: {e}")
            yield "Sorry, an error occurred while processing your request. Please try again."

    def get_system_status(self) -> dict:
        """
        Get system status
        """
        try:
            # Collection information
            collection_info = self.vector_store.get_collection_info()

            # Check OpenAI connection
            openai_status = self.llm_handler.check_connection()

            # Check data folder existence
            data_path_exists = Path(DATA_PATH).exists()

            return {
                "vector_store": {
                    "status": "✅ Working",
                    "collection_name": collection_info.get("name", "Unknown"),
                    "documents_count": collection_info.get("document_count", 0),
                    "embedding_model": collection_info.get("embedding_model", "Unknown")
                },
                "llm": {
                    "status": "✅ Working" if openai_status else "❌ Connection error",
                    "model": self.llm_handler.llm.model_name if hasattr(self.llm_handler.llm,
                                                                        'model_name') else "Unknown"
                },
                "data_path": {
                    "path": DATA_PATH,
                    "exists": "✅ Exists" if data_path_exists else "❌ Not found",
                    "files_count": len(list(Path(DATA_PATH).rglob("*"))) if data_path_exists else 0
                }
            }

        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {
                "error": f"Status error: {str(e)}"
            }

    def clear_vector_store(self) -> dict:
        """
        Clear vector store
        """
        try:
            self.vector_store.clear_collection()
            return {
                "success": True,
                "message": "Vector store cleared"
            }
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            return {
                "success": False,
                "message": f"Clearing error: {str(e)}"
            }

    def search_documents(self, query: str, k: int = SEARCH_K) -> List[dict]:
        """
        Search documents without generating response (for debugging)
        """
        try:
            relevant_docs = self.vector_store.search_similar_documents(query, k)

            results = []
            for doc, score in relevant_docs:
                results.append({
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "score": round(score, 3),
                    "filename": doc.metadata.get("filename", "Unknown"),
                    "directory": doc.metadata.get("directory", ""),
                    "chunk_index": doc.metadata.get("chunk_index", 0)
                })

            return results

        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []

    def _is_general_query(self, query: str) -> bool:
        """
        Determine if query is a general greeting or question
        """
        general_patterns = [
            'hello', 'hi', 'hey', 'good morning', 'good evening', 'good day',
            'how are you', 'what can you do', 'help', 'who are you', 'what are you',
            'thank you', 'thanks', 'bye', 'goodbye', 'see you',
            'привет', 'здравствуй', 'добрый день', 'добрый вечер',
            'как дела', 'что умеешь', 'помоги', 'кто ты', 'что ты',
            'спасибо', 'пока', 'до свидания'
        ]

        return any(pattern in query for pattern in general_patterns)

    def _handle_general_query(self, query: str) -> str:
        """
        Handle general queries without document search
        """
        query_lower = query.lower()

        if any(word in query_lower for word in
               ['hello', 'hi', 'hey', 'good morning', 'good evening', 'good day', 'привет', 'здравствуй', 'добрый']):
            return """Hello! 👋 

I'm your AI assistant for working with documents. I can help you find information in uploaded files and answer questions based on their content.

What I can do:
• Answer questions about the content of your documents
• Search for specific information in files
• Analyze and compare data from different documents

Ask me a question about the content of your files, and I'll try to help!"""

        elif any(word in query_lower for word in
                 ['what can you do', 'what are you', 'who are you', 'help', 'что умеешь', 'что ты', 'кто ты',
                  'помоги']):
            return """I'm your personal assistant for working with documents! 🤖

My capabilities:
• 📚 Analyze content of your documents (TXT, PDF, DOCX, CSV, MD)
• 🔍 Fast information search across your knowledge base
• 💡 Answer questions based on uploaded files
• 📊 Compare and analyze data from different documents

Just ask me a question about the content of your files, for example:
"Tell me about John Smith" or "What products do we have?" """

        elif any(word in query_lower for word in ['thank you', 'thanks', 'спасибо']):
            return "You're welcome! Happy to help. If you have more questions about your documents, feel free to ask! 😊"

        elif any(word in query_lower for word in ['bye', 'goodbye', 'see you', 'пока', 'до свидания']):
            return "Goodbye! Good luck working with your documents! 👋"

        else:
            return "I'm ready to help you search for information in your documents. Ask a specific question about the content of your files!"