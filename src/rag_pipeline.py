from typing import List, Dict
from pathlib import Path
from datetime import datetime

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

        # Conversation memory
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history_length = 10  # Maximum messages in memory

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
            new_docs_count = self.vector_store.add_documents(documents)
            if new_docs_count is None:
                new_docs_count = 0

            collection_info = self.vector_store.get_collection_info()

            if new_docs_count == 0:
                return {
                    "success": True,
                    "message": f"All {len(documents)} documents already exist in database",
                    "documents_loaded": 0,
                    "total_in_collection": collection_info.get("document_count", 0)
                }

            return {
                "success": True,
                "message": f"Successfully loaded {new_docs_count} new documents/chunks (skipped {len(documents) - new_docs_count} duplicates)",
                "documents_loaded": new_docs_count,
                "total_in_collection": collection_info.get("document_count", 0)
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Loading error: {str(e)}",
                "documents_loaded": 0
            }

    def query_with_history(self, user_question: str, k: int = SEARCH_K) -> str:
        """Process user query with conversation history"""
        if not user_question.strip():
            return "Please ask a question."

        if self._is_general_query(user_question.lower()):
            response = self._handle_general_query(user_question)
            self._add_to_history(user_question, response)
            return response

        relevant_docs = self.vector_store.search_similar_documents(user_question, k)

        if not relevant_docs:
            response = "I specialize in answering questions based on documents in your knowledge base. Try asking about the content of your documents."
            self._add_to_history(user_question, response)
            return response

        filtered_docs = [(doc, score) for doc, score in relevant_docs if score >= SIMILARITY_THRESHOLD]

        if not filtered_docs:
            response = "I didn't find sufficiently relevant information in your documents for this question. Try rephrasing your query or ask a more specific question."
            self._add_to_history(user_question, response)
            return response

        try:
            response = self.llm_handler.generate_response_with_history(
                query=user_question,
                relevant_docs=filtered_docs,
                conversation_history=self.conversation_history
            )

            # Add to history
            self._add_to_history(user_question, response)

            return response
        except Exception as e:
            error_response = "Sorry, an error occurred while processing your request. Please try again."
            self._add_to_history(user_question, error_response)
            return error_response

    def query_stream(self, user_question: str, k: int = SEARCH_K):
        """Process user query in streaming mode with history"""
        if not user_question.strip():
            yield "Please ask a question."
            return

        # Handle general queries
        if self._is_general_query(user_question.lower()):
            response = self._handle_general_query(user_question)
            self._add_to_history(user_question, response)
            yield response
            return

        relevant_docs = self.vector_store.search_similar_documents(user_question, k)

        if not relevant_docs:
            response = "I specialize in answering questions based on documents in your knowledge base. Try asking about the content of your documents."
            self._add_to_history(user_question, response)
            yield response
            return

        filtered_docs = [(doc, score) for doc, score in relevant_docs if score >= SIMILARITY_THRESHOLD]

        if not filtered_docs:
            response = "I didn't find sufficiently relevant information in your documents for this question. Try rephrasing your query or ask a more specific question."
            self._add_to_history(user_question, response)
            yield response
            return

        try:
            full_response = ""
            for response_chunk in self.llm_handler.generate_response_stream_with_history(
                    query=user_question,
                    relevant_docs=filtered_docs,
                    conversation_history=self.conversation_history
            ):
                full_response = response_chunk
                yield response_chunk

            # Add final response to history
            if full_response:
                self._add_to_history(user_question, full_response)

        except Exception as e:
            error_response = "Sorry, an error occurred while processing your request. Please try again."
            self._add_to_history(user_question, error_response)
            yield error_response

    def _add_to_history(self, question: str, answer: str):
        """Add conversation turn to history"""
        self.conversation_history.append({
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        # Limit history size
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        return self.conversation_history.copy()

    def clear_conversation_history(self) -> dict:
        """Clear conversation history"""
        self.conversation_history = []
        return {"success": True, "message": "Conversation history cleared"}

    def get_system_status(self) -> dict:
        """Get system status"""
        status = {
            "vector_store": {
                "status": "❌ Error",
                "collection_name": "Unknown",
                "documents_count": 0,
                "embedding_model": "Unknown"
            },
            "llm": {
                "status": "❌ Error",
                "model": "Unknown"
            },
            "data_path": {
                "path": DATA_PATH,
                "exists": "❌ Error",
                "files_count": 0
            },
            "conversation": {
                "history_length": len(self.conversation_history),
                "max_history": self.max_history_length
            }
        }

        # Check vector store
        try:
            collection_info = self.vector_store.get_collection_info()
            status["vector_store"] = {
                "status": "✅ Working",
                "collection_name": collection_info.get("name", "Unknown"),
                "documents_count": collection_info.get("document_count", 0),
                "embedding_model": collection_info.get("embedding_model", "Unknown")
            }
        except Exception as e:
            status["vector_store"]["status"] = f"❌ Error: {str(e)}"

        # Check LLM
        try:
            openai_status = self.llm_handler.check_connection()
            model_name = "Unknown"
            try:
                model_name = getattr(self.llm_handler.llm, 'model_name', "Unknown")
            except:
                pass

            status["llm"] = {
                "status": "✅ Working" if openai_status else "❌ Connection error",
                "model": model_name
            }
        except Exception as e:
            status["llm"]["status"] = f"❌ Error: {str(e)}"

        # Check data path
        try:
            data_path_exists = Path(DATA_PATH).exists()
            files_count = 0
            if data_path_exists:
                files_count = len(list(Path(DATA_PATH).rglob("*")))

            status["data_path"] = {
                "path": DATA_PATH,
                "exists": "✅ Exists" if data_path_exists else "❌ Not found",
                "files_count": files_count
            }
        except Exception as e:
            status["data_path"]["exists"] = f"❌ Error: {str(e)}"

        return status

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

I'll remember our conversation, so you can ask follow-up questions and I'll understand the context!

Ask me a question about the content of your files, and I'll try to help!"""

        elif any(word in query_lower for word in ['what can you do', 'what are you', 'who are you', 'help']):
            return """I'm your personal assistant for working with documents! 🤖

My capabilities:
• 📚 Analyze content of your documents (TXT, PDF, DOCX, CSV, MD)
• 🔍 Fast information search across your knowledge base
• 💡 Answer questions based on uploaded files
• 📊 Compare and analyze data from different documents
• 🗣️ Remember our conversation for context and follow-up questions

Just ask me a question about the content of your files!"""

        elif any(word in query_lower for word in ['thank you', 'thanks']):
            return "You're welcome! Happy to help. If you have more questions about your documents, feel free to ask! 😊"

        elif any(word in query_lower for word in ['bye', 'goodbye', 'see you']):
            return "Goodbye! Good luck working with your documents! 👋"

        return "I'm ready to help you search for information in your documents. Ask a specific question about the content of your files!"