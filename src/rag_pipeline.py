from typing import List
from pathlib import Path

from src.document_loader import DocumentLoader
from src.vector_store import VectorStore
from src.llm_handler import LLMHandler
from config.settings import DATA_PATH, SEARCH_K, SIMILARITY_THRESHOLD


class RAGPipeline:
    def __init__(self):
        """Initialize the RAG pipeline"""
        self.document_loader = DocumentLoader()
        self.vector_store = VectorStore()
        self.llm_handler = LLMHandler()

    def load_documents(self, data_path: str = DATA_PATH) -> dict:
        """Load documents from folder into vector database"""
        if not Path(data_path).exists():
            return {
                "success": False,
                "message": f"Folder {data_path} does not exist",
                "documents_loaded": 0
            }

        documents = self.document_loader.load_documents_from_directory(data_path)

        if not documents:
            return {
                "success": False,
                "message": "No documents found for loading",
                "documents_loaded": 0
            }

        try:
            self.vector_store.add_documents(documents)
            collection_info = self.vector_store.get_collection_info()

            return {
                "success": True,
                "message": f"Successfully loaded {len(documents)} documents/chunks",
                "documents_loaded": len(documents),
                "total_in_collection": collection_info.get("document_count", 0)
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Loading error: {str(e)}",
                "documents_loaded": 0
            }

    def query(self, user_question: str, k: int = SEARCH_K) -> str:
        """Process user query through RAG pipeline"""
        if not user_question.strip():
            return "Please ask a question."

        # Handle general queries
        if self._is_general_query(user_question.lower()):
            return self._handle_general_query(user_question)

        # Search for relevant documents
        relevant_docs = self.vector_store.search_similar_documents(user_question, k)

        if not relevant_docs:
            return "I specialize in answering questions based on documents in your knowledge base. Try asking about the content of your documents."

        # Filter by relevance threshold
        filtered_docs = [(doc, score) for doc, score in relevant_docs if score >= SIMILARITY_THRESHOLD]

        if not filtered_docs:
            return "I didn't find sufficiently relevant information in your documents for this question. Try rephrasing your query or ask a more specific question."

        try:
            return self.llm_handler.generate_response(user_question, filtered_docs)
        except Exception as e:
            return "Sorry, an error occurred while processing your request. Please try again."

    def query_stream(self, user_question: str, k: int = SEARCH_K):
        """Process user query in streaming mode"""
        if not user_question.strip():
            yield "Please ask a question."
            return

        # Handle general queries
        if self._is_general_query(user_question.lower()):
            yield self._handle_general_query(user_question)
            return

        # Search for relevant documents
        relevant_docs = self.vector_store.search_similar_documents(user_question, k)

        if not relevant_docs:
            yield "I specialize in answering questions based on documents in your knowledge base. Try asking about the content of your documents."
            return

        # Filter by relevance threshold
        filtered_docs = [(doc, score) for doc, score in relevant_docs if score >= SIMILARITY_THRESHOLD]

        if not filtered_docs:
            yield "I didn't find sufficiently relevant information in your documents for this question. Try rephrasing your query or ask a more specific question."
            return

        try:
            for response_chunk in self.llm_handler.generate_response_stream(user_question, filtered_docs):
                yield response_chunk
        except Exception as e:
            yield "Sorry, an error occurred while processing your request. Please try again."

    def get_system_status(self) -> dict:
        """Get system status"""
        try:
            collection_info = self.vector_store.get_collection_info()
            openai_status = self.llm_handler.check_connection()
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
                    "model": getattr(self.llm_handler.llm, 'model_name', "Unknown")
                },
                "data_path": {
                    "path": DATA_PATH,
                    "exists": "✅ Exists" if data_path_exists else "❌ Not found",
                    "files_count": len(list(Path(DATA_PATH).rglob("*"))) if data_path_exists else 0
                }
            }
        except Exception as e:
            return {"error": f"Status error: {str(e)}"}

    def clear_vector_store(self) -> dict:
        """Clear vector store"""
        try:
            self.vector_store.clear_collection()
            return {"success": True, "message": "Vector store cleared"}
        except Exception as e:
            return {"success": False, "message": f"Clearing error: {str(e)}"}

    def search_documents(self, query: str, k: int = SEARCH_K) -> List[dict]:
        """Search documents for debugging"""
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
        except:
            return []

    def _is_general_query(self, query: str) -> bool:
        """Check if query is a general greeting"""
        general_patterns = [
            'hello', 'hi', 'hey', 'good morning', 'good evening', 'good day',
            'how are you', 'what can you do', 'help', 'who are you', 'what are you',
            'thank you', 'thanks', 'bye', 'goodbye', 'see you'
        ]
        return any(pattern in query for pattern in general_patterns)

    def _handle_general_query(self, query: str) -> str:
        """Handle general queries"""
        query_lower = query.lower()

        if any(word in query_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good evening']):
            return """Hello! 👋 

I'm your AI assistant for working with documents. I can help you find information in uploaded files and answer questions based on their content.

Ask me a question about the content of your files, and I'll try to help!"""

        elif any(word in query_lower for word in ['what can you do', 'what are you', 'who are you', 'help']):
            return """I'm your personal assistant for working with documents! 🤖

My capabilities:
• 📚 Analyze content of your documents (TXT, PDF, DOCX, CSV, MD)
• 🔍 Fast information search across your knowledge base
• 💡 Answer questions based on uploaded files
• 📊 Compare and analyze data from different documents

Just ask me a question about the content of your files!"""

        elif any(word in query_lower for word in ['thank you', 'thanks']):
            return "You're welcome! Happy to help. If you have more questions about your documents, feel free to ask! 😊"

        elif any(word in query_lower for word in ['bye', 'goodbye', 'see you']):
            return "Goodbye! Good luck working with your documents! 👋"

        return "I'm ready to help you search for information in your documents. Ask a specific question about the content of your files!"