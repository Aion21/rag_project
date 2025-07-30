import gradio as gr
import logging
import os
from pathlib import Path

from src.rag_pipeline import RAGPipeline
from config.settings import DATA_PATH, GRADIO_SHARE, GRADIO_PORT

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize RAG pipeline
rag_pipeline = None
try:
    from src.rag_pipeline import RAGPipeline

    rag_pipeline = RAGPipeline()
    logger.info("RAG Pipeline successfully initialized")
except ImportError as e:
    logger.error(f"Import error for RAG Pipeline: {e}")
    logger.error("Make sure all dependencies are installed:")
    logger.error("pip install sentence-transformers chromadb langchain langchain-openai")
except Exception as e:
    logger.error(f"Error initializing RAG Pipeline: {e}")
    import traceback

    traceback.print_exc()


def load_documents_interface():
    """
    Interface for loading documents
    """
    if not rag_pipeline:
        return "❌ RAG Pipeline not initialized"

    try:
        result = rag_pipeline.load_documents()

        if result["success"]:
            return f"✅ {result['message']}\n📊 Total in collection: {result['total_in_collection']} documents"
        else:
            return f"❌ {result['message']}"

    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        return f"❌ Error: {str(e)}"


def query_interface(question, history):
    """
    Interface for processing user questions - СОВМЕСТИМО С GRADIO 3.x
    """
    if not rag_pipeline:
        history = history or []
        history.append([question, "❌ RAG Pipeline not initialized"])
        return history, ""

    if not question.strip():
        history = history or []
        history.append([question, "Please ask a question."])
        return history, ""

    try:
        # Generate response
        response = rag_pipeline.query(question)

        # Add to history
        history = history or []
        history.append([question, response])

        return history, ""

    except Exception as e:
        logger.error(f"Error processing question: {e}")
        error_response = f"❌ An error occurred: {str(e)}"
        history = history or []
        history.append([question, error_response])
        return history, ""


def clear_chat():
    """
    Clear chat history
    """
    return []


def get_system_status_text():
    """
    Get text representation of system status
    """
    if not rag_pipeline:
        return "❌ RAG Pipeline not initialized"

    try:
        status = rag_pipeline.get_system_status()

        status_text = f"""
## 📊 System Status

### 🗄️ Vector Store
- Status: {status['vector_store']['status']}
- Collection: {status['vector_store']['collection_name']}
- Documents: {status['vector_store']['documents_count']}
- Embedding model: {status['vector_store']['embedding_model']}

### 🤖 LLM
- Status: {status['llm']['status']}
- Model: {status['llm']['model']}

### 📁 Data Folder
- Path: {status['data_path']['path']}
- Exists: {status['data_path']['exists']}
- Files found: {status['data_path']['files_count']}
"""
        return status_text

    except Exception as e:
        return f"❌ Error getting status: {str(e)}"


def clear_vector_store_interface():
    """
    Interface for clearing vector store
    """
    if not rag_pipeline:
        return "❌ RAG Pipeline not initialized"

    try:
        result = rag_pipeline.clear_vector_store()

        if result["success"]:
            return f"✅ {result['message']}"
        else:
            return f"❌ {result['message']}"

    except Exception as e:
        logger.error(f"Error clearing vector store: {e}")
        return f"❌ Error: {str(e)}"


def search_documents_interface(query, k):
    """
    Interface for document search (debugging)
    """
    if not rag_pipeline:
        return "❌ RAG Pipeline not initialized"

    if not query.strip():
        return "Enter a search query"

    try:
        results = rag_pipeline.search_documents(query, int(k))

        if not results:
            return "🔍 No documents found"

        search_results = f"🔍 Found {len(results)} documents:\n\n"

        for i, result in enumerate(results, 1):
            search_results += f"""
**{i}. {result['filename']}** (Score: {result['score']})
- Folder: {result['directory']}
- Chunk: {result['chunk_index']}

{result['content']}

---
"""

        return search_results

    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        return f"❌ Error: {str(e)}"


# Create Gradio interface - ВЕРСИЯ ДЛЯ GRADIO 3.x
def create_interface():
    """
    Create Gradio interface compatible with 3.x
    """

    # Простые функции для Gradio 3.x
    def chat_interface(question, history):
        return query_interface(question, history)

    def load_docs():
        return load_documents_interface()

    def clear_db():
        return clear_vector_store_interface()

    def get_status():
        return get_system_status_text()

    def search_docs(query, k):
        return search_documents_interface(query, k)

    # Интерфейс с вкладками для Gradio 3.x
    with gr.Blocks(title="RAG System") as interface:
        gr.Markdown("# 🤖 RAG System with ChromaDB")
        gr.Markdown("Intelligent question-answering system based on your documents")

        with gr.Tab("💬 Chat"):
            chatbot = gr.Chatbot(label="Chat History")
            question_input = gr.Textbox(label="Your Question", placeholder="Ask a question about your documents...")

            with gr.Row():
                submit_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear Chat")

            # События для Gradio 3.x
            submit_btn.click(
                chat_interface,
                inputs=[question_input, chatbot],
                outputs=[chatbot, question_input]
            )

            question_input.submit(
                chat_interface,
                inputs=[question_input, chatbot],
                outputs=[chatbot, question_input]
            )

            clear_btn.click(
                clear_chat,
                outputs=[chatbot]
            )

        with gr.Tab("📚 Documents"):
            gr.Markdown(f"### 📁 Documents Folder: `{DATA_PATH}`")
            gr.Markdown("Supported formats: TXT, PDF, DOCX, MD, CSV")

            with gr.Row():
                load_btn = gr.Button("📥 Load Documents", variant="primary")
                clear_db_btn = gr.Button("🗑️ Clear Database")

            load_status = gr.Textbox(label="Status", lines=5)

            load_btn.click(load_docs, outputs=[load_status])
            clear_db_btn.click(clear_db, outputs=[load_status])

        with gr.Tab("⚙️ Status"):
            status_display = gr.Markdown(get_system_status_text())
            refresh_btn = gr.Button("🔄 Refresh")

            refresh_btn.click(get_status, outputs=[status_display])

        with gr.Tab("🔍 Search"):
            gr.Markdown("### Search in Vector Database")

            search_input = gr.Textbox(label="Search Query")
            search_k = gr.Number(label="Number of Results", value=5)
            search_btn = gr.Button("🔍 Search", variant="primary")
            search_output = gr.Markdown(label="Results")

            search_btn.click(
                search_docs,
                inputs=[search_input, search_k],
                outputs=[search_output]
            )

    return interface


def main():
    """
    Main function to start the application
    """
    logger.info("Starting RAG system...")

    # Check for necessary folders
    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs("chroma_db", exist_ok=True)

    # Create and launch interface
    interface = create_interface()

    logger.info(f"Starting Gradio interface on port {GRADIO_PORT}")

    # Запуск с базовыми настройками для Gradio 3.x
    interface.launch(
        share=GRADIO_SHARE,
        server_port=GRADIO_PORT,
        show_error=True
    )


if __name__ == "__main__":
    main()