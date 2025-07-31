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

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to vector store"""
        if not documents:
            return

        batch_size = 100

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            texts = []
            metadatas = []
            ids = []

            for j, doc in enumerate(batch):
                texts.append(doc.page_content)
                metadatas.append(doc.metadata)
                doc_id = f"doc_{i + j}_{hash(doc.page_content) % 1000000}"
                ids.append(doc_id)

            try:
                embeddings = self.embedding_model.encode(texts).tolist()
                self.collection.add(
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids,
                    embeddings=embeddings
                )
            except:
                continue

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
        except:
            return {
                "name": COLLECTION_NAME,
                "document_count": 0,
                "embedding_model": EMBEDDING_MODEL
            }

    def clear_collection(self) -> None:
        """Clear collection"""
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "RAG documents"}
        )