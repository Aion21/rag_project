import gradio as gr
import logging
import os
import time
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


def chat_with_bot(message, history):
    """
    Generator function for streaming chat response
    """
    if not rag_pipeline:
        history.append([message, "❌ RAG Pipeline not initialized"])
        yield history
        return

    if not message.strip():
        history.append([message, "Please ask a question."])
        yield history
        return

    try:
        # Сначала добавляем сообщение пользователя
        history.append([message, None])
        yield history

        # Небольшая задержка для визуального разделения
        time.sleep(0.1)

        # Генерируем ответ
        response = rag_pipeline.query(message)

        # Обновляем последнее сообщение с ответом
        history[-1][1] = response
        yield history

    except Exception as e:
        logger.error(f"Error processing question: {e}")
        error_response = f"❌ An error occurred: {str(e)}"
        history[-1][1] = error_response
        yield history


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


# Create Gradio interface
def create_interface():
    """
    Create Gradio interface with streaming chat
    """

    # Interface with tabs
    with gr.Blocks(title="RAG System", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🤖 RAG System with ChromaDB")
        gr.Markdown("Intelligent question-answering system based on your documents")

        with gr.Tab("💬 Chat"):
            # Chat interface с использованием ChatInterface для корректного отображения
            chatbot = gr.Chatbot(
                [],
                elem_id="chatbot",
                bubble_full_width=False,
                height=500,
                show_label=False
            )

            msg = gr.Textbox(
                label="Your message",
                placeholder="Ask a question about your documents...",
                container=False,
                scale=7
            )

            with gr.Row():
                submit = gr.Button("Send", variant="primary", scale=1)
                clear = gr.Button("Clear", scale=1)

            # Обработчики событий с использованием генератора
            def user_message(message, history):
                return "", history + [[message, None]]

            def bot_response(history):
                if not history or not history[-1][0]:
                    return history

                user_msg = history[-1][0]

                if not rag_pipeline:
                    history[-1][1] = "❌ RAG Pipeline not initialized"
                    return history

                try:
                    response = rag_pipeline.query(user_msg)
                    history[-1][1] = response
                except Exception as e:
                    logger.error(f"Error processing question: {e}")
                    history[-1][1] = f"❌ An error occurred: {str(e)}"

                return history

            # Подключение событий
            msg.submit(user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
                bot_response, chatbot, chatbot
            )
            submit.click(user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
                bot_response, chatbot, chatbot
            )
            clear.click(lambda: ([], ""), outputs=[chatbot, msg], queue=False)

        with gr.Tab("📚 Documents"):
            gr.Markdown(f"### 📁 Documents Folder: `{DATA_PATH}`")
            gr.Markdown("Supported formats: TXT, PDF, DOCX, MD, CSV")

            with gr.Row():
                load_btn = gr.Button("📥 Load Documents", variant="primary")
                clear_db_btn = gr.Button("🗑️ Clear Database")

            load_status = gr.Textbox(label="Status", lines=5)

            load_btn.click(
                load_documents_interface,
                outputs=[load_status]
            )
            clear_db_btn.click(
                clear_vector_store_interface,
                outputs=[load_status]
            )

        with gr.Tab("⚙️ Status"):
            status_display = gr.Markdown(get_system_status_text())
            refresh_btn = gr.Button("🔄 Refresh")

            refresh_btn.click(
                get_system_status_text,
                outputs=[status_display]
            )

        with gr.Tab("🔍 Search"):
            gr.Markdown("### Search in Vector Database")

            search_input = gr.Textbox(label="Search Query")
            search_k = gr.Number(label="Number of Results", value=5)
            search_btn = gr.Button("🔍 Search", variant="primary")
            search_output = gr.Markdown(label="Results")

            search_btn.click(
                search_documents_interface,
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

    # Launch Gradio interface
    interface.launch(
        share=GRADIO_SHARE,
        server_port=GRADIO_PORT,
        show_error=True,
        server_name="0.0.0.0"  # Allow external connections
    )


if __name__ == "__main__":
    main()