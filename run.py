#!/usr/bin/env python3
"""
Quick launch script for RAG system
"""

import sys
import os
from pathlib import Path

# Add root folder to PYTHONPATH
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def check_environment():
    """Checks environment readiness"""
    print("🔍 Checking environment...")

    # Check .env file
    env_file = project_root / ".env"
    if not env_file.exists():
        print("❌ .env file not found!")
        print("📝 Copy .env.example to .env and add your OpenAI API key")
        return False

    # Check data folder
    data_dir = project_root / "data"
    if not data_dir.exists():
        print("📁 Creating data/ folder...")
        data_dir.mkdir(exist_ok=True)

    # Check chroma_db folder
    chroma_dir = project_root / "chroma_db"
    if not chroma_dir.exists():
        print("📁 Creating chroma_db/ folder...")
        chroma_dir.mkdir(exist_ok=True)

    print("✅ Environment ready!")
    return True


def main():
    """Main launch function"""
    print("""
    🤖 RAG System with ChromaDB
    ========================
    """)

    if not check_environment():
        sys.exit(1)

    try:
        print("🚀 Starting system...")
        from main import main as run_main
        run_main()
    except KeyboardInterrupt:
        print("\n👋 System stopped by user")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   conda activate rag_project")
        print("   pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Launch error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()