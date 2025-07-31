import gradio as gr
import os
from pathlib import Path

from src.rag_pipeline import RAGPipeline
from config.settings import DATA_PATH, GRADIO_SHARE, GRADIO_PORT

# Initialize RAG pipeline
rag_pipeline = None
try:
    rag_pipeline = RAGPipeline()
except Exception as e:
    print(f"Error initializing RAG Pipeline: {e}")


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
        return f"❌ Error: {str(e)}"


def get_system_status_text():
    """Get system status"""
    if not rag_pipeline:
        return "❌ RAG Pipeline not initialized"

    try:
        status = rag_pipeline.get_system_status()

        # Check if there's a general error
        if "error" in status:
            return f"❌ System Error: {status['error']}"

        return f"""
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

### 🗣️ Conversation
- History length: {status['conversation']['history_length']}
- Max history: {status['conversation']['max_history']}
"""
    except Exception as e:
        return f"❌ Error getting status: {str(e)}"


def clear_vector_store_interface():
    """Interface for clearing vector store"""
    if not rag_pipeline:
        return "❌ RAG Pipeline not initialized"

    try:
        result = rag_pipeline.clear_vector_store()
        return f"✅ {result['message']}" if result["success"] else f"❌ {result['message']}"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def search_documents_interface(query, k):
    """Interface for document search"""
    if not rag_pipeline or not query.strip():
        return "❌ RAG Pipeline not initialized" if not rag_pipeline else "Enter a search query"

    try:
        results = rag_pipeline.search_documents(query, int(k))
        if not results:
            return "🔍 No documents found"

        search_results = f"🔍 Found {len(results)} documents:\n\n"
        for i, result in enumerate(results, 1):
            search_results += f"""**{i}. {result['filename']}** (Score: {result['score']})
- Folder: {result['directory']}
- Chunk: {result['chunk_index']}

{result['content']}

---
"""
        return search_results
    except Exception as e:
        return f"❌ Error: {str(e)}"


def respond_to_user_with_history(message, history):
    """Chat function with conversation history"""
    if not rag_pipeline or not message.strip():
        return "❌ RAG Pipeline not initialized" if not rag_pipeline else "Please ask a question."

    try:
        return rag_pipeline.query_with_history(message)
    except Exception as e:
        return f"❌ An error occurred: {str(e)}"


def clear_conversation_history():
    """Clear conversation history"""
    if not rag_pipeline:
        return "❌ RAG Pipeline not initialized"

    result = rag_pipeline.clear_conversation_history()
    return f"✅ {result['message']}" if result["success"] else f"❌ {result['message']}"


def get_conversation_history():
    """Get conversation history for display"""
    if not rag_pipeline:
        return "❌ RAG Pipeline not initialized"

    history = rag_pipeline.get_conversation_history()
    if not history:
        return "📭 No conversation history"

    history_text = "📜 **Conversation History:**\n\n"
    for i, turn in enumerate(history, 1):
        history_text += f"**Turn {i}:** _{turn.get('timestamp', 'Unknown time')}_\n"
        history_text += f"👤 **User:** {turn['question']}\n"
        history_text += f"🤖 **Assistant:** {turn['answer']}\n\n"
        history_text += "---\n\n"

    return history_text


def create_interface():
    """Create Gradio interface"""
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: 0 auto !important;
        padding: 20px !important;
    }

    .header-container {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 40px 30px;
        border-radius: 15px;
        margin: 0 auto 30px auto;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        width: 900px !important;
        max-width: 900px !important;
        min-width: 900px !important;
    }

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
    }

    .header-subtitle {
        color: rgba(255, 255, 255, 0.9) !important;
        font-size: 1.2em !important;
        font-weight: 300 !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
    }

    .custom-button {
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }

    .custom-button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2) !important;
    }

    .tabitem {
        max-width: 900px !important;
        width: 900px !important;
        margin: 0 auto !important;
        padding: 20px !important;
    }

    @media (max-width: 950px) {
        .tabitem {
            width: calc(100vw - 40px) !important;
            max-width: calc(100vw - 40px) !important;
        }
    }
    """

    with gr.Blocks(title="🤖 RAG System", css=custom_css) as interface:
        gr.HTML("""
        <div class="header-container">
            <h1 class="header-title">🤖 RAG System with Memory</h1>
            <p class="header-subtitle">Intelligent conversational AI that remembers your dialogue</p>
        </div>
        """)

        with gr.Tab("💬 Chat"):
            with gr.Column(elem_classes="tabitem"):
                gr.Markdown("### 💬 Conversational Chat with Memory")
                gr.Markdown("*I remember our conversation and can reference previous messages naturally*")

                chatbot = gr.Chatbot(label="Chat", height=500)

                with gr.Row():
                    msg = gr.Textbox(
                        label="Your message",
                        placeholder="Ask me anything about your documents",
                        scale=4,
                        container=False
                    )
                    submit_btn = gr.Button("Send 💬", variant="primary", scale=1, elem_classes="custom-button")

                with gr.Row():
                    clear_chat_btn = gr.Button("Clear Chat 🗑️", variant="secondary", elem_classes="custom-button")
                    clear_memory_btn = gr.Button("Clear Memory 🧠", variant="secondary", elem_classes="custom-button")
                    show_history_btn = gr.Button("Show History 📜", variant="secondary", elem_classes="custom-button")

                history_display = gr.Markdown(label="Conversation History",
                                              value="*Click 'Show History' to see our conversation...*")

                # Chat handlers
                def user_message(message, history):
                    return "", history + [[message, None]] if message.strip() else ("", history)

                def bot_response(history):
                    if not history or not history[-1][0]:
                        return history
                    user_msg = history[-1][0]
                    response = respond_to_user_with_history(user_msg, history)
                    history[-1][1] = response
                    return history

                def clear_chat():
                    return [], ""

                def clear_memory():
                    clear_conversation_history()
                    return "✅ Conversation memory cleared! I won't remember our previous discussion."

                # Connect events
                msg.submit(user_message, [msg, chatbot], [msg, chatbot], queue=False).then(bot_response, chatbot,
                                                                                           chatbot)
                submit_btn.click(user_message, [msg, chatbot], [msg, chatbot], queue=False).then(bot_response, chatbot,
                                                                                                 chatbot)
                clear_chat_btn.click(clear_chat, outputs=[chatbot, msg], queue=False)
                clear_memory_btn.click(clear_memory, outputs=[history_display])
                show_history_btn.click(get_conversation_history, outputs=[history_display])

        with gr.Tab("📚 Documents"):
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
            with gr.Column(elem_classes="tabitem"):
                status_display = gr.Markdown(get_system_status_text())
                refresh_btn = gr.Button("🔄 Refresh Status", variant="primary", size="lg", elem_classes="custom-button")
                refresh_btn.click(get_system_status_text, outputs=[status_display])

        with gr.Tab("🔍 Search & Debug"):
            with gr.Column(elem_classes="tabitem"):
                gr.Markdown(
                    "### 🔍 Search in Vector Database\n*This tab is for debugging and testing document search functionality*")

                with gr.Row():
                    search_input = gr.Textbox(
                        label="🔎 Search Query",
                        placeholder="Enter search terms to find relevant documents...",
                        scale=3
                    )
                    search_k = gr.Number(label="📊 Results Count", value=5, minimum=1, maximum=20, scale=1)

                search_btn = gr.Button("🔍 Search Documents", variant="primary", size="lg", elem_classes="custom-button")
                search_output = gr.Markdown(label="📋 Search Results", value="*Search results will appear here...*")

                search_btn.click(search_documents_interface, inputs=[search_input, search_k], outputs=[search_output])
                search_input.submit(search_documents_interface, inputs=[search_input, search_k],
                                    outputs=[search_output])

    return interface


def main():
    """Main function to start the application"""
    print("🚀 Starting RAG system with conversation memory...")

    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs("chroma_db", exist_ok=True)

    interface = create_interface()

    print(f"🌐 Starting Gradio interface on port {GRADIO_PORT}")
    print(f"📂 Data folder: {DATA_PATH}")
    print(f"🗄️ Database folder: chroma_db/")
    print(f"🗣️ Conversation memory enabled!")
    print(f"💡 The AI will remember your conversation context!")

    interface.launch(
        share=GRADIO_SHARE,
        server_port=GRADIO_PORT,
        show_error=True,
        server_name="0.0.0.0",
        inbrowser=True
    )


if __name__ == "__main__":
    main()