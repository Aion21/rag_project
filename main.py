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
    """Interface for loading documents"""
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


def get_system_status_text():
    """Get text representation of system status"""
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
    """Interface for clearing vector store"""
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
    """Interface for document search (debugging)"""
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


def respond_to_user(message, history):
    """Main chat function"""
    if not rag_pipeline:
        return "❌ RAG Pipeline not initialized"

    if not message.strip():
        return "Please ask a question."

    try:
        logger.info(f"Processing user query: {message}")
        response = rag_pipeline.query(message)
        logger.info(f"Generated response of length: {len(response)}")
        return response
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        return f"❌ An error occurred: {str(e)}"


def create_interface():
    """Create Gradio interface with maximum compatibility"""

    # CSS для красивого оформления и центрирования
    custom_css = """
    /* Центрирование основного контейнера */
    .gradio-container {
        max-width: 1200px !important;
        margin: 0 auto !important;
        padding: 20px !important;
    }

    /* Центрирование всего контента */
    .main {
        max-width: 1000px !important;
        margin: 0 auto !important;
    }

    /* Центрирование блоков */
    .block {
        max-width: 100% !important;
        margin: 0 auto !important;
    }

    /* Центрирование чатбота */
    .chatbot {
        margin: 0 auto !important;
        max-width: 800px !important;
    }

    /* Центрирование поля ввода */
    .input-container {
        max-width: 800px !important;
        margin: 0 auto !important;
    }

    /* ФИКСИРОВАННАЯ ширина шапки - НЕ зависит от контента */
    .header-container {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 40px 30px;
        border-radius: 15px;
        margin: 0 auto 30px auto;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.1);
        width: 900px !important;           /* ФИКСИРОВАННАЯ ширина */
        max-width: 900px !important;       /* Максимальная ширина */
        min-width: 900px !important;       /* Минимальная ширина */
        box-sizing: border-box !important; /* Учитывает padding и border */
        position: relative !important;     /* Для стабильного позиционирования */
    }

    /* Адаптивность для мобильных устройств */
    @media (max-width: 950px) {
        .header-container {
            width: calc(100vw - 40px) !important;
            min-width: auto !important;
            max-width: calc(100vw - 40px) !important;
        }
    }

    .header-title {
        color: white !important;
        font-size: 2.5em !important;
        font-weight: bold !important;
        margin-bottom: 10px !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        letter-spacing: 1px;
    }

    .header-subtitle {
        color: rgba(255, 255, 255, 0.9) !important;
        font-size: 1.2em !important;
        font-weight: 300 !important;
        margin-top: 10px !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
    }

    /* Стили для кнопок */
    .custom-button {
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }

    .custom-button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2) !important;
    }

    /* Центрирование вкладок */
    .tab-nav {
        display: flex !important;
        justify-content: center !important;
        margin-bottom: 20px !important;
    }

    /* Центрирование содержимого вкладок с фиксированной шириной */
    .tabitem {
        max-width: 900px !important;
        width: 900px !important;           /* ФИКСИРОВАННАЯ ширина контента */
        margin: 0 auto !important;
        padding: 20px !important;
        box-sizing: border-box !important;
    }

    /* Адаптивность для контента вкладок */
    @media (max-width: 950px) {
        .tabitem {
            width: calc(100vw - 40px) !important;
            max-width: calc(100vw - 40px) !important;
        }
    }

    /* Фиксированная ширина для основного блока интерфейса */
    .gradio-interface {
        width: 100% !important;
        max-width: 900px !important;
        margin: 0 auto !important;
    }
    """

    with gr.Blocks(
            title="🤖 RAG System",
            css=custom_css
    ) as interface:

        # Основной контейнер с фиксированной структурой
        with gr.Column(elem_classes="gradio-interface"):
            # Красивая шапка с фиксированной шириной
            gr.HTML("""
            <div class="header-container">
                <h1 class="header-title">🤖 RAG System with ChromaDB</h1>
                <p class="header-subtitle">Intelligent question-answering system based on your documents</p>
            </div>
            """)

        with gr.Tab("💬 Chat"):
            # Центрированный контейнер для чата
            with gr.Column(elem_classes="chat-container"):
                gr.Markdown("### 💬 Chat with your documents")
                gr.Markdown("*Ask questions about your uploaded documents and get intelligent responses*")

                # Классический интерфейс чата с центрированием
                chatbot = gr.Chatbot(
                    label="Chat History",
                    height=500,
                    elem_classes="chatbot"
                )

                # Центрированный контейнер для ввода
                with gr.Column(elem_classes="input-container"):
                    with gr.Row():
                        msg = gr.Textbox(
                            label="Your message",
                            placeholder="Ask a question about your documents...",
                            scale=4,
                            container=False
                        )
                        submit_btn = gr.Button("Send 📤", variant="primary", scale=1, elem_classes="custom-button")

                    clear_btn = gr.Button("Clear Chat 🗑️", variant="secondary", elem_classes="custom-button")

            # Функции для обработки чата
            def user_message(message, history):
                if not message.strip():
                    return "", history
                return "", history + [[message, None]]

            def bot_response(history):
                if not history or not history[-1][0]:
                    return history

                user_msg = history[-1][0]
                response = respond_to_user(user_msg, history)
                history[-1][1] = response
                return history

            def clear_chat():
                return [], ""

            # Подключение событий
            msg.submit(
                user_message,
                [msg, chatbot],
                [msg, chatbot],
                queue=False
            ).then(
                bot_response,
                chatbot,
                chatbot
            )

            submit_btn.click(
                user_message,
                [msg, chatbot],
                [msg, chatbot],
                queue=False
            ).then(
                bot_response,
                chatbot,
                chatbot
            )

            clear_btn.click(
                clear_chat,
                outputs=[chatbot, msg],
                queue=False
            )

        with gr.Tab("📚 Documents"):
            # Центрированный контейнер для документов
            with gr.Column(elem_classes="tabitem"):
                gr.Markdown(f"""
                ### 📁 Documents Folder: `{DATA_PATH}`
                **Supported formats:** 📄 TXT • 📋 PDF • 📝 DOCX • 📑 MD • 📊 CSV

                Upload your documents to the data folder and click "Load Documents" to start chatting with your files.
                """)

                with gr.Row():
                    load_btn = gr.Button("📥 Load Documents", variant="primary", size="lg", elem_classes="custom-button")
                    clear_db_btn = gr.Button("🗑️ Clear Database", variant="secondary", size="lg",
                                             elem_classes="custom-button")

                load_status = gr.Textbox(
                    label="📋 Status",
                    lines=6,
                    max_lines=10,
                    placeholder="Status information will appear here..."
                )

                load_btn.click(load_documents_interface, outputs=[load_status])
                clear_db_btn.click(clear_vector_store_interface, outputs=[load_status])

        with gr.Tab("⚙️ System Status"):
            # Центрированный контейнер для статуса
            with gr.Column(elem_classes="tabitem"):
                status_display = gr.Markdown(get_system_status_text())
                refresh_btn = gr.Button("🔄 Refresh Status", variant="primary", size="lg", elem_classes="custom-button")

                refresh_btn.click(get_system_status_text, outputs=[status_display])

        with gr.Tab("🔍 Search & Debug"):
            # Центрированный контейнер для поиска
            with gr.Column(elem_classes="tabitem"):
                gr.Markdown("""
                ### 🔍 Search in Vector Database
                *This tab is for debugging and testing document search functionality*
                """)

                with gr.Row():
                    search_input = gr.Textbox(
                        label="🔎 Search Query",
                        placeholder="Enter search terms to find relevant documents...",
                        scale=3
                    )
                    search_k = gr.Number(
                        label="📊 Results Count",
                        value=5,
                        minimum=1,
                        maximum=20,
                        scale=1
                    )

                search_btn = gr.Button("🔍 Search Documents", variant="primary", size="lg", elem_classes="custom-button")

                search_output = gr.Markdown(
                    label="📋 Search Results",
                    value="*Search results will appear here...*"
                )

                search_btn.click(
                    search_documents_interface,
                    inputs=[search_input, search_k],
                    outputs=[search_output]
                )

                search_input.submit(
                    search_documents_interface,
                    inputs=[search_input, search_k],
                    outputs=[search_output]
                )

    return interface


def main():
    """Main function to start the application"""
    logger.info("🚀 Starting RAG system...")

    # Check for necessary folders
    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs("chroma_db", exist_ok=True)

    # Create and launch interface
    interface = create_interface()

    logger.info(f"🌐 Starting Gradio interface on port {GRADIO_PORT}")
    logger.info(f"📂 Data folder: {DATA_PATH}")
    logger.info(f"🗄️ Database folder: chroma_db/")

    # Launch Gradio interface
    interface.launch(
        share=GRADIO_SHARE,
        server_port=GRADIO_PORT,
        show_error=True,
        server_name="0.0.0.0",
        inbrowser=True  # автоматически открыть браузер
    )


if __name__ == "__main__":
    main()