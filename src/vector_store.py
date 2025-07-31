from typing import List, Tuple
import chromadb
from chromadb.config import Settings
from langchain.schema import Document
from sentence_transformers import SentenceTransformer

from config.settings import CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL, SEARCH_K


class VectorStore:
    def __init__(self):
        """Initialize ChromaDB vector store"""
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        self.client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        try:
            self.collection = self.client.get_collection(COLLECTION_NAME)
        except:
            self.collection = self.client.create_collection(
                name=COLLECTION_NAME,
                metadata={"description": "RAG documents"}
            )

    def add_documents(self, documents: List[Document]) -> int:
        """Add documents to vector store"""
        if not documents:
            return 0
            0

        # Get existing document IDs to avoid duplicates
        try:
            existing_result = self.collection.get(include=['metadatas'])

            # Create set of existing document identifiers
            existing_docs = set()
            if existing_result.get('metadatas'):
                for metadata in existing_result['metadatas']:
                    source = metadata.get('source', '')
                    chunk_idx = metadata.get('chunk_index', 0)
                    doc_identifier = f"{source}__chunk_{chunk_idx}"
                    existing_docs.add(doc_identifier)
        except:
            existing_docs = set()

        batch_size = 100
        new_documents_count = 0

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            texts = []
            metadatas = []
            ids = []

            for j, doc in enumerate(batch):
                # Create STABLE ID based on file path and chunk index
                source = doc.metadata.get('source', '')
                chunk_idx = doc.metadata.get('chunk_index', 0)

                # Create identifier to check for duplicates
                doc_identifier = f"{source}__chunk_{chunk_idx}"

                # Skip if document already exists
                if doc_identifier in existing_docs:
                    continue

                # Create unique ID for ChromaDB (must be unique)
                doc_id = f"doc_{len(existing_docs) + len(ids)}_{hash(doc_identifier) % 1000000}"

                texts.append(doc.page_content)
                metadatas.append(doc.metadata)
                ids.append(doc_id)

                # Add to existing set to avoid duplicates within this batch
                existing_docs.add(doc_identifier)

            # Only add if we have new documents
            if texts:
                try:
                    embeddings = self.embedding_model.encode(texts).tolist()
                    self.collection.add(
                        documents=texts,
                        metadatas=metadatas,
                        ids=ids,
                        embeddings=embeddings
                    )
                    new_documents_count += len(texts)
                except Exception as e:
                    # If batch fails, try adding documents one by one
                    for k in range(len(texts)):
                        try:
                            self.collection.add(
                                documents=[texts[k]],
                                metadatas=[metadatas[k]],
                                ids=[ids[k]],
                                embeddings=[embeddings[k]]
                            )
                            new_documents_count += 1
                        except:
                            continue

        return new_documents_count

    def search_similar_documents(self, query: str, k: int = SEARCH_K) -> List[Tuple[Document, float]]:
        """Search for similar documents"""
        try:
            query_embedding = self.embedding_model.encode([query]).tolist()

            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=k,
                include=['documents', 'metadatas', 'distances']
            )

            similar_docs = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    doc = Document(
                        page_content=results['documents'][0][i],
                        metadata=results['metadatas'][0][i]
                    )

                    distance = results['distances'][0][i]

                    # Convert distance to similarity score
                    similarity_base = max(0, 1 - (distance / 2))

                    if distance <= 0.8:
                        similarity = min(0.95, similarity_base * 1.3)
                    elif distance <= 1.2:
                        similarity = min(0.85, similarity_base * 1.2)
                    elif distance <= 1.6:
                        similarity = min(0.75, similarity_base * 1.1)
                    else:
                        similarity = similarity_base

                    # Bonus for top document
                    if i == 0:
                        similarity = min(0.95, similarity * 1.15)

                    # Minimum threshold for relevant documents
                    if distance < 2.0:
                        similarity = max(similarity, 0.15)

                    similar_docs.append((doc, similarity))

            # Sort by similarity (highest first)
            similar_docs.sort(key=lambda x: x[1], reverse=True)
            return similar_docs

        except:
            return []

    def get_collection_info(self) -> dict:
        """Get collection information"""
        try:
            count = self.collection.count()
            return {
                "name": COLLECTION_NAME,
                "document_count": count,
                "embedding_model": EMBEDDING_MODEL
            }
        except Exception as e:
            return {
                "name": COLLECTION_NAME,
                "document_count": 0,
                "embedding_model": EMBEDDING_MODEL,
                "error": str(e)
            }

    def clear_collection(self) -> None:
        """Clear collection"""
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "RAG documents"}
        )