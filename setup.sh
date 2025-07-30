#!/bin/bash

# Smart Setup Script for RAG System with ChromaDB
# Automatically detects platform and installs appropriate packages

echo "🚀 Smart Setup for RAG System with ChromaDB"
echo "==========================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }

# Detect platform and system information
detect_platform() {
    print_info "Detecting platform and system information..."

    OS=$(uname -s | tr '[:upper:]' '[:lower:]')
    ARCH=$(uname -m | tr '[:upper:]' '[:lower:]')
    PYTHON_VERSION=$(python3 --version 2>/dev/null | cut -d' ' -f2 | cut -d'.' -f1,2)

    # Check if conda is available
    CONDA_AVAILABLE=false
    if command -v conda &> /dev/null; then
        CONDA_AVAILABLE=true
        CONDA_VERSION=$(conda --version | cut -d' ' -f2)
    fi

    # Check for GPU
    GPU_AVAILABLE=false
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null 2>&1; then
            GPU_AVAILABLE=true
        fi
    fi

    # Determine platform type
    PLATFORM_TYPE="unknown"
    if [[ "$OS" == "darwin" && "$ARCH" == "arm64" ]]; then
        PLATFORM_TYPE="apple_silicon"
    elif [[ "$OS" == "darwin" ]]; then
        PLATFORM_TYPE="macos_intel"
    elif [[ "$OS" == "linux" && "$GPU_AVAILABLE" == true ]]; then
        PLATFORM_TYPE="linux_gpu"
    elif [[ "$OS" == "linux" ]]; then
        PLATFORM_TYPE="linux_cpu"
    elif [[ "$OS" == *"mingw"* ]] || [[ "$OS" == *"cygwin"* ]] || [[ "$OS" == *"msys"* ]]; then
        PLATFORM_TYPE="windows"
    fi

    print_success "Platform detected: $PLATFORM_TYPE"
    print_info "OS: $OS, Architecture: $ARCH"
    print_info "Python: $PYTHON_VERSION, Conda: $CONDA_AVAILABLE, GPU: $GPU_AVAILABLE"
}

# Remove existing environment if it exists
cleanup_environment() {
    print_info "Cleaning up existing environment..."

    if [ "$CONDA_AVAILABLE" = true ]; then
        conda env remove -n rag_project --yes 2>/dev/null || true
        print_success "Removed existing conda environment"
    fi
}

# Get PyTorch installation command based on platform
get_pytorch_packages() {
    case $PLATFORM_TYPE in
        apple_silicon)
            echo "torch torchvision torchaudio"
            ;;
        linux_gpu)
            echo "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
            ;;
        linux_cpu|macos_intel)
            echo "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
            ;;
        windows)
            echo "torch torchvision torchaudio"
            ;;
        *)
            echo "torch torchvision torchaudio"
            ;;
    esac
}

# Create optimized requirements files
create_requirements() {
    print_info "Creating platform-optimized requirements..."

    PYTORCH_PACKAGES=$(get_pytorch_packages)

    # Create requirements.txt
    cat > requirements-optimized.txt << EOF
# Platform-optimized requirements for $PLATFORM_TYPE
# Generated on $(date)

# Core scientific packages
pandas
numpy
scikit-learn

# Environment and configuration
python-dotenv
pydantic

# OpenAI and LangChain ecosystem
openai
tiktoken
langchain
langchain-community
langchain-openai

# Vector database and embeddings
chromadb
sentence-transformers

# Document processing
pypdf2
python-docx

# Web interface
gradio

# PyTorch (platform-optimized)
$PYTORCH_PACKAGES
EOF

    print_success "Created requirements-optimized.txt"

    # Create conda environment file if conda is available
    if [ "$CONDA_AVAILABLE" = true ]; then
        cat > environment-optimized.yml << EOF
name: rag_project
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pandas
  - numpy
  - scikit-learn
  - pip
  - pip:
    - python-dotenv
    - pydantic
    - openai
    - tiktoken
    - langchain
    - langchain-community
    - langchain-openai
    - chromadb
    - sentence-transformers
    - pypdf2
    - python-docx
    - gradio
$(for pkg in $PYTORCH_PACKAGES; do
    if [[ "$pkg" != --* ]]; then
        echo "    - $pkg"
    fi
done)
EOF
        print_success "Created environment-optimized.yml"
    fi
}

# Install packages using the best method
install_packages() {
    print_info "Installing packages using optimal method..."

    if [ "$CONDA_AVAILABLE" = true ]; then
        print_info "Using conda for installation..."

        # Create conda environment
        print_info "Creating conda environment..."
        if conda env create -f environment-optimized.yml; then
            print_success "Conda environment created successfully!"
            print_warning "Activate with: conda activate rag_project"
        else
            print_error "Conda installation failed!"
            print_info "Falling back to pip installation..."
            CONDA_AVAILABLE=false
            install_packages
        fi

    else
        print_info "Using pip for installation..."

        # Check if we're in a virtual environment
        if [[ "$VIRTUAL_ENV" == "" ]]; then
            print_warning "Not in a virtual environment. Creating one..."
            if python3 -m venv rag_env; then
                source rag_env/bin/activate
                print_success "Virtual environment created and activated"
            else
                print_error "Failed to create virtual environment!"
                print_info "Installing in system Python (not recommended)"
            fi
        fi

        # Install packages
        print_info "Upgrading pip..."
        pip install --upgrade pip

        print_info "Installing packages..."
        if pip install -r requirements-optimized.txt; then
            print_success "Pip installation completed!"
        else
            print_error "Pip installation failed!"
            print_info "Try manual installation with verbose output:"
            print_info "pip install -v -r requirements-optimized.txt"
            return 1
        fi

        if [[ "$VIRTUAL_ENV" == *"rag_env"* ]]; then
            print_warning "Activate with: source rag_env/bin/activate"
        fi
    fi
}

# Setup project structure and configuration
setup_project() {
    print_info "Setting up project structure..."

    # Create necessary directories
    mkdir -p data chroma_db src config utils

    # Create __init__.py files
    touch src/__init__.py config/__init__.py utils/__init__.py

    print_success "Project directories created"

    # Setup .env file
    if [ ! -f .env ]; then
        if [ -f .env.example ]; then
            cp .env.example .env
            print_success "Created .env from .env.example"
        else
            cat > .env << 'EOF'
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo

# Logging Configuration
LOG_LEVEL=INFO

# Gradio Configuration
GRADIO_SHARE=False
GRADIO_PORT=7860

# Advanced Settings
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
SEARCH_K=5
SIMILARITY_THRESHOLD=0.20
EOF
            print_success "Created basic .env file"
        fi
        print_warning "⚠️  Add your OpenAI API key to .env file!"
    else
        print_info ".env file already exists"
    fi
}

# Verify installation
verify_installation() {
    print_info "Verifying installation..."

    # Test imports
    if [ "$CONDA_AVAILABLE" = true ]; then
        PYTHON_CMD="conda run -n rag_project python"
    else
        PYTHON_CMD="python"
    fi

    $PYTHON_CMD -c "
import sys
try:
    import gradio as gr
    import chromadb
    import langchain
    import openai
    import torch
    import pandas as pd
    import numpy as np
    print('✅ All core packages imported successfully!')
    print(f'📦 Gradio version: {gr.__version__}')
    print(f'🔥 PyTorch version: {torch.__version__}')
    if torch.backends.mps.is_available():
        print('🍎 MPS (Apple Silicon) support available')
    elif torch.cuda.is_available():
        print('🎮 CUDA support available')
    else:
        print('💻 CPU version installed')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
" 2>/dev/null

    if [ $? -eq 0 ]; then
        print_success "Installation verification passed!"
    else
        print_error "Installation verification failed!"
        print_info "Try manual installation: pip install -r requirements-optimized.txt"
    fi
}

# Save platform information
save_platform_info() {
    cat > platform-info.txt << EOF
RAG System Platform Information
==============================
Generated: $(date)

Platform Type: $PLATFORM_TYPE
OS: $OS
Architecture: $ARCH
Python Version: $PYTHON_VERSION
Conda Available: $CONDA_AVAILABLE
GPU Available: $GPU_AVAILABLE

PyTorch Packages: $(get_pytorch_packages)

Installation Method: $(if [ "$CONDA_AVAILABLE" = true ]; then echo "conda"; else echo "pip"; fi)
Environment Name: $(if [ "$CONDA_AVAILABLE" = true ]; then echo "rag_project"; else echo "rag_env"; fi)
EOF

    print_success "Platform info saved to platform-info.txt"
}

# Main execution
main() {
    # Parse command line arguments
    SKIP_INSTALL=false
    FORCE_PIP=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-install)
                SKIP_INSTALL=true
                shift
                ;;
            --pip-only)
                FORCE_PIP=true
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [options]"
                echo ""
                echo "Options:"
                echo "  --skip-install  Create files but skip package installation"
                echo "  --pip-only      Force pip installation (skip conda)"
                echo "  --help          Show this help"
                echo ""
                exit 0
                ;;
            *)
                print_warning "Unknown option: $1"
                shift
                ;;
        esac
    done

    # Override conda detection if forced to use pip
    if [ "$FORCE_PIP" = true ]; then
        CONDA_AVAILABLE=false
        print_info "Forced to use pip-only installation"
    fi

    # Execute setup steps
    detect_platform
    cleanup_environment
    create_requirements
    setup_project
    save_platform_info

    if [ "$SKIP_INSTALL" = false ]; then
        echo ""
        read -p "🤔 Install packages now? (y/n): " -n 1 -r
        echo ""

        if [[ $REPLY =~ ^[Yy]$ ]]; then
            install_packages
            verify_installation

            echo ""
            print_success "🎉 Setup completed successfully!"
            echo ""
            print_info "📋 Next steps:"
            echo "1. Edit .env file and add your OpenAI API key"
            echo "2. Add documents to data/ folder"
            if [ "$CONDA_AVAILABLE" = true ]; then
                echo "3. Run: conda activate rag_project && python main.py"
            else
                echo "3. Run: source rag_env/bin/activate && python main.py"
            fi
            echo "4. Open browser: http://localhost:7860"
        else
            echo ""
            print_info "📋 Files created. To install manually:"
            echo "• pip install -r requirements-optimized.txt"
            if [ "$CONDA_AVAILABLE" = true ]; then
                echo "• conda env create -f environment-optimized.yml"
            fi
            echo "• Edit .env file and add your OpenAI API key"
            echo "• Run: python main.py"
        fi
    else
        print_info "📋 Files created. Package installation skipped."
        echo "• Install with: pip install -r requirements-optimized.txt"
        if [ "$CONDA_AVAILABLE" = true ]; then
            echo "• Or with conda: conda env create -f environment-optimized.yml"
        fi
    fi
}

# Run main function with all arguments
main "$@"