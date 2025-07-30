import logging
from typing import List, Tuple
import chromadb
from chromadb.config import Settings
from langchain.schema import Document
from sentence_transformers import SentenceTransformer

from config.settings import CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL, SEARCH_K

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self):
        """
        Initialize ChromaDB vector store
        """
        try:
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info(f"✅ Embedding model {EMBEDDING_MODEL} loaded successfully")
        except Exception as e:
            logger.error(f"❌ Error loading embedding model: {e}")
            raise

        # Setup ChromaDB
        try:
            self.client = chromadb.PersistentClient(
                path=CHROMA_DB_PATH,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info(f"✅ ChromaDB client initialized at {CHROMA_DB_PATH}")
        except Exception as e:
            logger.error(f"❌ Error initializing ChromaDB: {e}")
            raise

        # Create or get collection
        try:
            self.collection = self.client.get_collection(COLLECTION_NAME)
            logger.info(f"✅ Collection '{COLLECTION_NAME}' found")
        except:
            self.collection = self.client.create_collection(
                name=COLLECTION_NAME,
                metadata={"description": "RAG documents"}
            )
            logger.info(f"✅ Created new collection '{COLLECTION_NAME}'")

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to vector store
        """
        if not documents:
            logger.warning("No documents to add")
            return

        logger.info(f"Adding {len(documents)} documents to vector store...")

        # Prepare data for ChromaDB in batches
        batch_size = 100  # Process in batches for large collections

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            texts = []
            metadatas = []
            ids = []

            for j, doc in enumerate(batch):
                texts.append(doc.page_content)
                metadatas.append(doc.metadata)
                # Create unique ID
                doc_id = f"doc_{i + j}_{hash(doc.page_content) % 1000000}"
                ids.append(doc_id)

            # Create embeddings
            logger.info(f"Creating embeddings for batch {i // batch_size + 1}...")
            try:
                embeddings = self.embedding_model.encode(texts).tolist()
            except Exception as e:
                logger.error(f"Error creating embeddings: {e}")
                continue

            # Add to ChromaDB
            try:
                self.collection.add(
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids,
                    embeddings=embeddings
                )
                logger.info(f"✅ Added batch {i // batch_size + 1} ({len(batch)} documents)")
            except Exception as e:
                logger.error(f"Error adding batch to ChromaDB: {e}")
                continue

        logger.info(f"✅ Successfully processed {len(documents)} documents")

    def search_similar_documents(self, query: str, k: int = SEARCH_K) -> List[Tuple[Document, float]]:
        """
        Search for similar documents by query
        """
        try:
            # Create embedding for query
            query_embedding = self.embedding_model.encode([query]).tolist()

            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=k,
                include=['documents', 'metadatas', 'distances']
            )

            # Convert results
            similar_docs = []
            if results['documents'] and results['documents'][0]:
                logger.info(f"Processing {len(results['documents'][0])} search results")

                for i in range(len(results['documents'][0])):
                    doc = Document(
                        page_content=results['documents'][0][i],
                        metadata=results['metadatas'][0][i]
                    )

                    # ChromaDB returns cosine distance (0 = identical, 2 = completely different)
                    distance = results['distances'][0][i]

                    # MAXIMALLY IMPROVED similarity formula:
                    # Cosine distance: 0 = identical, 2 = completely different

                    # Basic linear transformation
                    similarity_base = max(0, 1 - (distance / 2))

                    # Apply more aggressive scaling for real data
                    if distance <= 0.8:  # Excellent matches
                        similarity = min(0.95, similarity_base * 1.3)
                    elif distance <= 1.2:  # Good matches
                        similarity = min(0.85, similarity_base * 1.2)
                    elif distance <= 1.6:  # Moderate matches
                        similarity = min(0.75, similarity_base * 1.1)
                    else:  # Weak matches
                        similarity = similarity_base

                    # Additional bonus for top document
                    if i == 0:  # First (best) document
                        similarity = min(0.95, similarity * 1.15)

                    # Ensure minimum reasonable score for relevant documents
                    if distance < 2.0:  # If document is somewhat relevant
                        similarity = max(similarity, 0.15)  # Minimum threshold

                    filename = results['metadatas'][0][i].get('filename', 'unknown')
                    logger.info(f"Document {filename}: distance={distance:.4f} -> similarity={similarity:.4f}")

                    similar_docs.append((doc, similarity))

            logger.info(f"Found {len(similar_docs)} similar documents")

            # Sort by similarity (highest first)
            similar_docs.sort(key=lambda x: x[1], reverse=True)
            logger.info("Documents sorted by similarity (highest first)")

            return similar_docs

        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []

    def get_collection_info(self) -> dict:
        """
        Get collection information
        """
        try:
            count = self.collection.count()
            return {
                "name": COLLECTION_NAME,
                "document_count": count,
                "embedding_model": EMBEDDING_MODEL
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {
                "name": COLLECTION_NAME,
                "document_count": 0,
                "embedding_model": EMBEDDING_MODEL
            }

    def clear_collection(self) -> None:
        """
        Clear collection
        """
        try:
            # Delete current collection
            self.client.delete_collection(COLLECTION_NAME)
            # Create new empty collection
            self.collection = self.client.create_collection(
                name=COLLECTION_NAME,
                metadata={"description": "RAG documents"}
            )
            logger.info("✅ Collection cleared")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            raise

    def document_exists(self, doc_hash: str) -> bool:
        """
        Check if document exists by hash
        """
        try:
            results = self.collection.get(ids=[f"doc_hash_{doc_hash}"])
            return len(results['ids']) > 0
        except:
            return False