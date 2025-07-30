# 🚀 Quick Start

## 1. Installation

```bash
# 1. Create conda environment
conda env create -f environment.yml

# 2. Activate environment
conda activate rag_project

# 3. Create .env file
cp .env.example .env
# Edit the .env file and add your OpenAI API key
```

## 2. Setup

```bash
# Create necessary folders
mkdir -p data chroma_db

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

```bash
# Method 1: Direct run
python main.py

# Method 2: Via run.py (with checks)
python run.py

# Method 3: Docker (if needed)
docker-compose up --build
```

## 4. Usage

1. Open your browser: http://localhost:7860
2. Go to the “Document Management” tab
3. Click “Upload Documents”
4. Go to the “Chat” tab and start asking questions!

## 🔧 Common issues

**OpenAI API Error:**
```bash
# Make sure the API key is correct in the .env file
grep OPENAI_API_KEY .env
```

**Documents not uploading:**
```bash
# Check the data/ folder
ls -la data/
# Make sure files have supported extensions: .txt, .pdf, .docx, .md, .csv
```

**Dependencies:**
```bash
# Reinstall the environment
conda env remove -n rag_project
conda env create -f environment.yml
conda activate rag_project
```

## ✨ Done!

Now you have a working RAG system with a beautiful interface!