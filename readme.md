# 🤖 RAG System with ChromaDB

An intelligent question-answering system based on your documents using ChromaDB, LangChain, and Gradio.

## 🌟 Features

- 📚 **Multiple Format Support**: TXT, PDF, DOCX, MD, CSV
- 🗄️ **Vector Database**: ChromaDB for fast document search
- 🤖 **OpenAI GPT Integration**: ChatGPT for intelligent response generation
- 🎨 **Modern Interface**: Intuitive Gradio web interface
- 🔍 **Smart Search**: Semantic search using vector embeddings
- 📁 **Recursive Processing**: Automatic processing of folders and subfolders

## 🚀 Quick Start

### 1. Installation with Conda (Recommended)

```bash
# Clone the repository
git clone <your-repo> rag_project
cd rag_project

# Create environment from file
conda env create -f environment.yml
conda activate rag_project
```

### 2. Alternative: Installation with Pip

```bash
# Create virtual environment
python -m venv rag_env

# Activate environment
# Windows:
rag_env\Scripts\activate
# macOS/Linux:
source rag_env/bin/activate

# Install packages
pip install -r requirements.txt
```

### 3. OpenAI API Setup

Create a `.env` file and add your OpenAI API key:

```env
OPENAI_API_KEY=your_actual_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
```

### 4. Document Preparation

Place your documents in the `data/` folder. The system will automatically process all supported files in folders and subfolders.

```
data/
├── folder1/
│   ├── document1.pdf
│   └── notes.txt
├── folder2/
│   ├── spreadsheet.csv
│   └── report.docx
└── readme.md
```

### 5. Run the System

```bash
conda activate rag_project
python main.py
```

Open your browser at: `http://localhost:7860`

## 📁 Project Structure

```
rag_project/
├── main.py                 # Gradio interface
├── environment.yml         # Conda dependencies  
├── requirements.txt        # pip dependencies
├── .env                   # Environment variables
├── README.md              # Documentation
├── config/
│   └── settings.py        # Configuration
├── src/
│   ├── __init__.py
│   ├── document_loader.py  # Document loading
│   ├── vector_store.py    # ChromaDB interface
│   ├── llm_handler.py     # OpenAI interface
│   └── rag_pipeline.py    # Main logic
├── utils/
│   ├── __init__.py
│   └── text_processing.py # Text processing
├── data/                  # Your documents
└── chroma_db/            # Vector database (auto-created)
```

## 🎯 Usage

### Web Interface

1. **💬 Chat**: Ask questions about your documents
2. **📚 Document Management**: Load and manage documents
3. **⚙️ System Status**: Monitor system status
4. **🔍 Document Search**: Debug and test search functionality

### Programmatic Interface

```python
from src.rag_pipeline import RAGPipeline

# Initialize
rag = RAGPipeline()

# Load documents
result = rag.load_documents("path/to/your/documents")
print(result)

# Ask a question
response = rag.query("How to configure the system?")
print(response)

# Get system status
status = rag.get_system_status()
print(status)
```

## ⚙️ Configuration

Main settings in `config/settings.py`:

```python
# Document chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Search parameters
SEARCH_K = 5  # Number of relevant documents
SIMILARITY_THRESHOLD = 0.7  # Relevance threshold

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# OpenAI model
OPENAI_MODEL = "gpt-3.5-turbo"
```

## 🔧 Supported Formats

| Format | Description | Extension |
|--------|-------------|-----------|
| **TXT** | Text files | `.txt` |
| **PDF** | PDF documents | `.pdf` |
| **DOCX** | Word documents | `.docx` |
| **Markdown** | Markdown files | `.md` |
| **CSV** | Excel/CSV tables | `.csv` |

## 🚨 Troubleshooting

### "OpenAI API key not found" Error
```bash
# Check .env file
cat .env
# Make sure OPENAI_API_KEY is set correctly
```

### Documents Not Loading
```bash
# Check data folder permissions
ls -la data/
# Make sure files have supported extensions
```

### Dependency Errors
```bash
# Reinstall environment
conda env remove -n rag_project
conda env create -f environment.yml
conda activate rag_project
```

### PyTorch Issues
```bash
# For GPU (NVIDIA):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## 📊 Performance

- **Indexing time**: ~1-2 seconds per document
- **Response time**: ~3-5 seconds per query
- **Supported volume**: up to 10,000 documents
- **Memory requirements**: ~2-4 GB RAM

## 🔒 Security

- Data is stored locally in ChromaDB
- OpenAI API is only used for response generation
- Original documents are not sent to OpenAI
- Logs do not contain sensitive information

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and add tests
4. Submit a Pull Request

---

**Built with ❤️ for efficient document processing**