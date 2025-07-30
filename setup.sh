#!/bin/bash

echo "🍎 Настройка RAG проекта для macOS ARM64..."

# Удаление существующего окружения
echo "🧹 Очистка старого окружения..."
conda env remove -n rag_project --yes 2>/dev/null || true

# Создание базового окружения
echo "📦 Создание базового conda окружения..."
conda create -n rag_project python=3.10 -y

if [ $? -ne 0 ]; then
    echo "❌ Ошибка при создании окружения!"
    exit 1
fi

# Активация окружения
echo "✅ Активация окружения..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate rag_project

# Установка базовых пакетов через conda
echo "📚 Установка базовых пакетов..."
conda install -c conda-forge pandas numpy matplotlib jupyter -y

# Установка специализированных пакетов через pip
echo "🤖 Установка RAG компонентов..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install chromadb gradio langchain langchain-community langchain-openai
pip install openai sentence-transformers python-dotenv pypdf2 python-docx tiktoken scikit-learn

# Создание структуры папок
echo "📁 Создание папок..."
mkdir -p data chroma_db src config utils

# Создание .env файла
echo "🔧 Создание .env..."
if [ ! -f .env ]; then
    cat > .env << EOL
# OpenAI API ключ (обязательно заполните!)
OPENAI_API_KEY=your_openai_api_key_here

# Модель OpenAI
OPENAI_MODEL=gpt-3.5-turbo

# Логирование
LOG_LEVEL=INFO
EOL
    echo "⚠️  Добавьте ваш OPENAI_API_KEY в .env"
fi

# Создание __init__.py файлов
touch src/__init__.py utils/__init__.py

# Проверка установки
echo "🔍 Проверка установки..."
python -c "
try:
    import chromadb, gradio, langchain, torch
    print('✅ Все пакеты установлены успешно!')
    print('🚀 Готово к запуску!')
except ImportError as e:
    print(f'❌ Ошибка импорта: {e}')
"

echo ""
echo "Следующие шаги:"
echo "1. conda activate rag_project"
echo "2. Добавьте OpenAI API ключ в .env"
echo "3. Поместите документы в data/"
echo "4. python main.py"