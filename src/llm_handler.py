import logging
from typing import List, Tuple
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage
import os

from config.settings import OPENAI_API_KEY, OPENAI_MODEL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMHandler:
    def __init__(self):
        """
        Initialize LLM handler
        """
        if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_api_key_here":
            raise ValueError("Must set OPENAI_API_KEY in .env file")

        # Устанавливаем переменную окружения для OpenAI
        os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

        # МИНИМАЛЬНАЯ инициализация без ВСЕХ дополнительных параметров
        try:
            self.llm = ChatOpenAI(
                model=OPENAI_MODEL or "gpt-3.5-turbo"
                # ТОЛЬКО model - никаких других параметров!
            )
            logger.info(f"✅ LLM initialized successfully with model: {OPENAI_MODEL}")
        except Exception as e:
            logger.error(f"❌ Error initializing LLM: {e}")
            # Fallback с только базовыми настройками
            try:
                self.llm = ChatOpenAI()  # Вообще без параметров
                logger.info("✅ LLM initialized with default settings")
            except Exception as e2:
                logger.error(f"❌ Fallback also failed: {e2}")
                raise

        # System prompt for RAG
        self.system_prompt = """You are a friendly and helpful AI assistant that answers questions based on provided documents.

RULES:
1. Use ONLY information from the provided documents to answer
2. If information is insufficient, honestly say so, but remain friendly
3. Do NOT mention technical details (score, relevance, metadata)
4. Do NOT mention or cite source files, document names, or where information came from
5. Be accurate, but friendly and understandable
6. If there is contradictory information in the documents, mention it diplomatically
7. Give comprehensive and useful answers based on the found information

COMMUNICATION STYLE:
- Friendly and professional
- Understandable to regular users
- No technical jargon
- Use emojis for friendliness (in moderation)

RESPONSE FORMAT:
- Give a direct and useful answer to the question
- Add context and details from documents
- DO NOT mention sources, files, or document names
- Answer as if you naturally know this information
"""

    def generate_response(self, query: str, relevant_docs: List[Tuple[Document, float]]) -> str:
        """
        Generate response based on query and relevant documents
        """
        try:
            if not relevant_docs:
                return "Sorry, I couldn't find relevant documents to answer your question. Please try rephrasing your query or make sure documents are loaded into the database."

            # Prepare context from documents
            context = self._prepare_context(relevant_docs)

            # Create prompt
            user_prompt = f"""
CONTEXT FROM DOCUMENTS:
{context}

USER QUESTION:
{query}

Answer the question using only the provided information from the documents.
"""

            # Create messages
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=user_prompt)
            ]

            # Generate response
            logger.info("Generating response with LLM...")
            response = self.llm.invoke(messages)

            # Get main response WITHOUT adding sources
            response_text = response.content

            logger.info(f"Response generated, length: {len(response_text)} characters")

            return response_text

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"An error occurred while generating the response: {str(e)}"

    def generate_response_stream(self, query: str, relevant_docs: List[Tuple[Document, float]]):
        """
        Generate response in REAL streaming mode from OpenAI
        """
        try:
            logger.info(f"[LLM STREAM] Starting REAL streaming for {len(relevant_docs)} documents")

            if not relevant_docs:
                yield "Sorry, I couldn't find relevant documents to answer your question."
                return

            # Prepare context from documents
            context = self._prepare_context(relevant_docs)

            # Create prompt
            user_prompt = f"""
CONTEXT FROM DOCUMENTS:
{context}

USER QUESTION:
{query}

Answer the question using only the provided information from the documents.
"""

            # Create messages
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=user_prompt)
            ]

            logger.info("[LLM STREAM] Starting real streaming from OpenAI...")

            # НАСТОЯЩИЙ streaming от OpenAI
            accumulated_response = ""
            chunk_count = 0

            try:
                # Используем stream метод LangChain
                for chunk in self.llm.stream(messages):
                    chunk_count += 1
                    if hasattr(chunk, 'content') and chunk.content:
                        accumulated_response += chunk.content
                        yield accumulated_response

                    # Логируем каждые 10 чанков
                    if chunk_count % 10 == 0:
                        logger.info(f"[LLM STREAM] Received {chunk_count} chunks...")

            except Exception as stream_error:
                logger.error(f"[LLM STREAM] Streaming failed: {stream_error}")
                # Fallback на обычную генерацию
                logger.info("[LLM STREAM] Falling back to regular generation...")
                full_response = self.generate_response(query, relevant_docs)
                yield full_response
                return

            logger.info(f"[LLM STREAM] Streaming completed with {chunk_count} chunks")

            # DON'T add source information in streaming mode
            # Just return the clean response
            logger.info("[LLM STREAM] Generation completed successfully")

        except Exception as e:
            logger.error(f"[LLM STREAM] Error in streaming generation: {e}")
            yield f"An error occurred while generating the response: {str(e)}"

    def _prepare_context(self, relevant_docs: List[Tuple[Document, float]]) -> str:
        """
        Prepare context from relevant documents (already filtered)
        """
        context_parts = []

        for i, (doc, score) in enumerate(relevant_docs, 1):
            source = doc.metadata.get('filename', 'Unknown file')
            directory = doc.metadata.get('directory', '')

            # Simplify directory display
            dir_name = directory.split('/')[-1] if '/' in directory else directory

            context_part = f"""
--- DOCUMENT {i} ---
Source: {source}
Folder: {dir_name}
Relevance: {score:.3f}

Content:
{doc.page_content}
"""
            context_parts.append(context_part)

        if not context_parts:
            return "No relevant documents for this query."

        return "\n".join(context_parts)

    def _format_sources_info(self, relevant_docs: List[Tuple[Document, float]]) -> str:
        """
        Format source information (only documents above threshold)
        """
        from config.settings import SIMILARITY_THRESHOLD

        sources = []
        seen_sources = set()

        # STRICTLY filter documents by threshold
        truly_relevant_docs = [
            (doc, score) for doc, score in relevant_docs
            if score >= SIMILARITY_THRESHOLD
        ]

        # If too many documents, take only top-2 most relevant
        if len(truly_relevant_docs) > 2:
            truly_relevant_docs = truly_relevant_docs[:2]
            logger.info(f"[SOURCES] Limiting to top-2 documents from {len(relevant_docs)}")

        logger.info(f"[SOURCES] Original documents: {len(relevant_docs)}")
        logger.info(f"[SOURCES] Relevant (>={SIMILARITY_THRESHOLD}): {len(truly_relevant_docs)}")

        for doc, score in truly_relevant_docs:
            filename = doc.metadata.get('filename', 'Unknown file')
            directory = doc.metadata.get('directory', '')

            logger.info(f"[SOURCES] Adding to sources: {filename} (score: {score:.3f})")

            source_key = f"{filename}|{directory}"
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                if directory:
                    # Show only folder name without full path
                    dir_name = directory.split('/')[-1] if '/' in directory else directory
                    sources.append(f"{filename} (folder: {dir_name})")
                else:
                    sources.append(filename)

        if sources:
            logger.info(f"[SOURCES] Final sources: {sources}")
            return f"**Sources:** {', '.join(sources)}"
        else:
            logger.warning("[SOURCES] No relevant sources!")
            return "**Sources:** no relevant documents found"

    def check_connection(self) -> bool:
        """
        Check connection to OpenAI API
        """
        try:
            test_messages = [
                SystemMessage(content="You are a test assistant."),
                HumanMessage(content="Hello! This is a connection test.")
            ]

            response = self.llm.invoke(test_messages)
            logger.info("✅ OpenAI API connection successful")
            return True

        except Exception as e:
            logger.error(f"❌ OpenAI API connection error: {e}")
            return False