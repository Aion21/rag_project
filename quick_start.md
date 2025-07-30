# 🚀 Quick Start Guide

## 1. Installation

### Option 1: Conda (Recommended)
```bash
# 1. Create conda environment
conda env create -f environment.yml

# 2. Activate environment
conda activate rag_project
```

### Option 2: Pip
```bash
# 1. Create virtual environment
python -m venv rag_env

# 2. Activate environment
# Windows:
rag_env\Scripts\activate
# macOS/Linux:
source rag_env/bin/activate

# 3. Install packages
pip install -r requirements.txt
```

## 2. Setup

### Environment Configuration
```bash
# 1. Create .env file
cp .env.example .env

# 2. Edit .env file and add your OpenAI API key
# OPENAI_API_KEY=your_actual_api_key_here
```

### Document Preparation
```bash
# Create data folder (if not exists)
mkdir -p data

# Add your documents to the data/ folder
# Example structure:
# data/
# ├── project1/
# │   ├── technical_documentation.pdf
# │   └── notes.txt
# └── project2/
#     └── report.docx
```

## 3. Run

### Method 1: Direct Run
```bash
python main.py
```

### Method 2: Using run.py (with checks)
```bash
python run.py
```

### Method 3: Docker (optional)
```bash
docker-compose up --build
```

## 4. Usage

1. **Open browser**: http://localhost:7860
2. **Go to "Documents" tab**
3. **Click "Load Documents"**
4. **Go to "Chat" tab** and start asking questions!

## 🔧 Common Issues

### OpenAI API Error
```bash
# Make sure API key is correct in .env file
grep OPENAI_API_KEY .env
```

### Documents Not Loading
```bash
# Check data/ folder
ls -la data/
# Ensure files have supported extensions: .txt, .pdf, .docx, .md, .csv
```

### Dependencies Issues
```bash
# Reinstall environment
conda env remove -n rag_project
conda env create -f environment.yml
conda activate rag_project
```

### PyTorch Installation Issues
```bash
# For NVIDIA GPU:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For Apple Silicon (M1/M2):
pip install torch torchvision torchaudio
```

### ChromaDB Issues
```bash
# Clear database if corrupted
rm -rf chroma_db/
# System will recreate it automatically
```

## 🎯 Quick Test

### Verify Installation
```bash
python -c "
import gradio as gr
import chromadb
import langchain
import openai
print('✅ All packages installed successfully!')
print('Gradio version:', gr.__version__)
"
```

### Test API Connection
```bash
python -c "
from src.rag_pipeline import RAGPipeline
rag = RAGPipeline()
status = rag.get_system_status()
print('System Status:', status)
"
```

## ✨ You're Ready!

Once everything is set up, you'll have a fully functional RAG system with a beautiful web interface!

### Next Steps:
1. 📁 **Add documents** to `data/` folder
2. 🔄 **Load documents** via web interface
3. 💬 **Start chatting** with your documents
4. 🔍 **Explore features** in different tabs

### Example Questions to Try:
- "What documents do you have?"
- "Summarize the main topics"
- "Tell me about [specific topic from your docs]"
- "What are the key findings?"

### Advanced Usage:
- **System Status tab**: Monitor system health
- **Search & Debug tab**: Test document search functionality
- **Configuration**: Modify settings in `config/settings.py`

## 📋 Checklist

- [ ] Environment created and activated
- [ ] Dependencies installed
- [ ] `.env` file configured with OpenAI API key
- [ ] Documents placed in `data/` folder
- [ ] System launched successfully
- [ ] Documents loaded via web interface
- [ ] First question asked in chat

**Happy document chatting! 🎉**