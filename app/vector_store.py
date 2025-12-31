import logging
from pathlib import Path
from typing import List, Dict

import faiss
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.settings import Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import load_index_from_storage



from ingest import load_pdfs, chunk_documents

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = BASE_DIR / "data" / "raw_pdfs"
VECTOR_STORE_DIR = BASE_DIR / "data" / "vector_store" / "faiss_index"

VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# FREE Embedding Model
# -----------------------------
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

EMBEDDING_DIM = 384  # dimension for MiniLM

# -----------------------------
# Build Vector Store
# -----------------------------
def build_vector_store(chunks: List[Dict]) -> VectorStoreIndex:
    documents = []

    for chunk in chunks:
        documents.append(
            Document(
                text=chunk["text"],
                metadata={
                    "source_file": chunk["source_file"],
                    "page_number": chunk["page_number"],
                    "chunk_id": chunk["chunk_id"],
                }
            )
        )

    faiss_index = faiss.IndexFlatL2(EMBEDDING_DIM)

    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )

    return index

# -----------------------------
# Persist Index
# -----------------------------
def persist_index(index: VectorStoreIndex):
    index.storage_context.persist(persist_dir=VECTOR_STORE_DIR)
    logger.info("Vector store persisted to disk.")

# -----------------------------
# Load Index
# -----------------------------

def load_index() -> VectorStoreIndex:
    vector_store = FaissVectorStore.from_persist_dir(
        VECTOR_STORE_DIR
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir=VECTOR_STORE_DIR,
    )

    return load_index_from_storage(storage_context)

# -----------------------------
# Similarity Search Test
# -----------------------------
def similarity_search(query: str, top_k: int = 3):
    index = load_index()
    retriever = index.as_retriever(similarity_top_k=top_k)
    return retriever.retrieve(query)

# -----------------------------
# Script Entry Point
# -----------------------------
if __name__ == "__main__":
    logger.info("Loading and chunking PDFs...")
    documents = load_pdfs(PDF_DIR)
    chunks = chunk_documents(documents)

    logger.info("Building vector store (FREE embeddings)...")
    index = build_vector_store(chunks)

    persist_index(index)

    test_query = "What does the policy say about employee leave?"
    logger.info(f"Running test query: {test_query}")

    results = similarity_search(test_query)

    print("\n" + "=" * 80)
    print("SIMILARITY SEARCH RESULTS")
    print("=" * 80)

    for i, node in enumerate(results, start=1):
        print(f"""
Result {i}
Score: {node.score}
Source: {node.metadata['source_file']}
Page: {node.metadata['page_number']}
Text Preview:
{node.text[:400]}
""")
