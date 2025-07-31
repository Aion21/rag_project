import os
from typing import List, Tuple
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage

from config.settings import OPENAI_API_KEY, OPENAI_MODEL


class LLMHandler:
    def __init__(self):
        """Initialize LLM handler"""
        if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_api_key_here":
            raise ValueError("Must set OPENAI_API_KEY in .env file")

        os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

        try:
            self.llm = ChatOpenAI(model=OPENAI_MODEL or "gpt-3.5-turbo")
        except:
            self.llm = ChatOpenAI()

        self.system_prompt = """You are a helpful AI assistant that answers questions based on provided documents.

RULES:
1. Use ONLY information from the provided documents
2. If information is insufficient, say so honestly
3. Don't mention technical details, scores, or file names
4. Be accurate and friendly
5. Answer as if you naturally know this information

Give direct, useful answers based on the document content."""

    def generate_response(self, query: str, relevant_docs: List[Tuple[Document, float]]) -> str:
        """Generate response based on query and documents"""
        if not relevant_docs:
            return "Sorry, I couldn't find relevant documents to answer your question. Please try rephrasing your query."

        context = self._prepare_context(relevant_docs)

        user_prompt = f"""
CONTEXT FROM DOCUMENTS:
{context}

USER QUESTION:
{query}

Answer the question using only the provided information.
"""

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt)
        ]

        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"An error occurred while generating the response: {str(e)}"

    def generate_response_stream(self, query: str, relevant_docs: List[Tuple[Document, float]]):
        """Generate streaming response"""
        if not relevant_docs:
            yield "Sorry, I couldn't find relevant documents to answer your question."
            return

        context = self._prepare_context(relevant_docs)

        user_prompt = f"""
CONTEXT FROM DOCUMENTS:
{context}

USER QUESTION:
{query}

Answer the question using only the provided information.
"""

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt)
        ]

        try:
            accumulated_response = ""
            for chunk in self.llm.stream(messages):
                if hasattr(chunk, 'content') and chunk.content:
                    accumulated_response += chunk.content
                    yield accumulated_response
        except:
            # Fallback to regular generation
            full_response = self.generate_response(query, relevant_docs)
            yield full_response

    def _prepare_context(self, relevant_docs: List[Tuple[Document, float]]) -> str:
        """Prepare context from relevant documents"""
        context_parts = []

        for i, (doc, score) in enumerate(relevant_docs, 1):
            source = doc.metadata.get('filename', 'Unknown file')
            directory = doc.metadata.get('directory', '')
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

        return "\n".join(context_parts) if context_parts else "No relevant documents for this query."

    def check_connection(self) -> bool:
        """Check OpenAI API connection"""
        try:
            test_messages = [
                SystemMessage(content="You are a test assistant."),
                HumanMessage(content="Hello! This is a connection test.")
            ]
            self.llm.invoke(test_messages)
            return True
        except:
            return False