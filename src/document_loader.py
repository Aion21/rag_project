import os
import logging
from typing import List, Dict
from pathlib import Path

import PyPDF2
import pandas as pd
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LangchainDocument

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP, SUPPORTED_EXTENSIONS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentLoader:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )

    def load_documents_from_directory(self, directory_path: str) -> List[LangchainDocument]:
        """
        Загружает все документы из указанной папки и подпапок
        """
        documents = []
        directory_path = Path(directory_path)

        if not directory_path.exists():
            logger.error(f"Директория {directory_path} не существует")
            return documents

        # Рекурсивно проходим по всем файлам
        for file_path in directory_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                try:
                    logger.info(f"Обрабатываем файл: {file_path}")
                    content = self._load_file_content(file_path)

                    if content:
                        # Создаем метаданные
                        metadata = {
                            "source": str(file_path),
                            "filename": file_path.name,
                            "directory": str(file_path.parent),
                            "extension": file_path.suffix.lower()
                        }

                        # Разбиваем на чанки
                        chunks = self.text_splitter.split_text(content)

                        # Создаем документы для каждого чанка
                        for i, chunk in enumerate(chunks):
                            chunk_metadata = metadata.copy()
                            chunk_metadata["chunk_index"] = i
                            chunk_metadata["total_chunks"] = len(chunks)

                            doc = LangchainDocument(
                                page_content=chunk,
                                metadata=chunk_metadata
                            )
                            documents.append(doc)

                except Exception as e:
                    logger.error(f"Ошибка при обработке файла {file_path}: {e}")
                    continue

        logger.info(f"Загружено {len(documents)} документов/чанков")
        return documents

    def _load_file_content(self, file_path: Path) -> str:
        """
        Загружает содержимое файла в зависимости от его типа
        """
        try:
            if file_path.suffix.lower() == '.txt' or file_path.suffix.lower() == '.md':
                return self._load_text_file(file_path)
            elif file_path.suffix.lower() == '.pdf':
                return self._load_pdf_file(file_path)
            elif file_path.suffix.lower() == '.docx':
                return self._load_docx_file(file_path)
            elif file_path.suffix.lower() == '.csv':
                return self._load_csv_file(file_path)
            else:
                logger.warning(f"Неподдерживаемый формат файла: {file_path}")
                return ""
        except Exception as e:
            logger.error(f"Ошибка при загрузке файла {file_path}: {e}")
            return ""

    def _load_text_file(self, file_path: Path) -> str:
        """Загружает текстовые файлы"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            return file.read()

    def _load_pdf_file(self, file_path: Path) -> str:
        """Загружает PDF файлы"""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text

    def _load_docx_file(self, file_path: Path) -> str:
        """Загружает DOCX файлы"""
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text

    def _load_csv_file(self, file_path: Path) -> str:
        """Загружает CSV файлы"""
        try:
            df = pd.read_csv(file_path)
            # Преобразуем DataFrame в текстовое представление
            text = f"Файл: {file_path.name}\n"
            text += f"Колонки: {', '.join(df.columns.tolist())}\n"
            text += f"Количество строк: {len(df)}\n\n"

            # Добавляем содержимое таблицы
            text += df.to_string(index=False)
            return text
        except Exception as e:
            logger.error(f"Ошибка при чтении CSV файла {file_path}: {e}")
            return ""