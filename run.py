#!/usr/bin/env python3
"""
Скрипт для быстрого запуска RAG системы
"""

import sys
import os
from pathlib import Path

# Добавляем корневую папку в PYTHONPATH
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def check_environment():
    """Проверяет готовность окружения"""
    print("🔍 Проверка окружения...")

    # Проверка .env файла
    env_file = project_root / ".env"
    if not env_file.exists():
        print("❌ Файл .env не найден!")
        print("📝 Скопируйте .env.example в .env и добавьте ваш OpenAI API ключ")
        return False

    # Проверка папки data
    data_dir = project_root / "data"
    if not data_dir.exists():
        print("📁 Создаем папку data/...")
        data_dir.mkdir(exist_ok=True)

    # Проверка папки chroma_db
    chroma_dir = project_root / "chroma_db"
    if not chroma_dir.exists():
        print("📁 Создаем папку chroma_db/...")
        chroma_dir.mkdir(exist_ok=True)

    print("✅ Окружение готово!")
    return True


def main():
    """Основная функция запуска"""
    print("""
    🤖 RAG Система с ChromaDB
    ========================
    """)

    if not check_environment():
        sys.exit(1)

    try:
        print("🚀 Запуск системы...")
        from main import main as run_main
        run_main()
    except KeyboardInterrupt:
        print("\n👋 Система остановлена пользователем")
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        print("💡 Убедитесь, что все зависимости установлены:")
        print("   conda activate rag_project")
        print("   pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Ошибка запуска: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()