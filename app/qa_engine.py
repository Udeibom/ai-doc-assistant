import logging
import time
from pathlib import Path

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.settings import Settings

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_STORE_DIR = BASE_DIR / "storage" / "vector_store"

# -----------------------------
# Embeddings (local)
# -----------------------------
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# LLM (remote via Groq)
# -----------------------------
Settings.llm = Groq(
    model="llama-3.1-8b-instant",
    temperature=0.1,
    max_tokens=256,
)


# -----------------------------
# Query Engine Loader
# -----------------------------
def load_query_engine(top_k: int = 2):
    logger.info("Loading vector index...")

    storage_context = StorageContext.from_defaults(
        persist_dir=VECTOR_STORE_DIR
    )

    index = load_index_from_storage(storage_context)

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k,
    )

    return RetrieverQueryEngine(retriever=retriever)

# -----------------------------
# Load ONCE
# -----------------------------
QUERY_ENGINE = load_query_engine()

# -----------------------------
# QA Function
# -----------------------------
def answer_question(query: str) -> str:
    print("⏳ Running query...")
    start = time.time()

    response = QUERY_ENGINE.query(query)

    print(f"✅ Done in {time.time() - start:.2f}s")
    return str(response)

# -----------------------------
# CLI Entry
# -----------------------------
if __name__ == "__main__":
    question = "What is the termination clause in the contract?"
    print("\n📄 Question:", question)
    print("\n🧠 Answer:\n")
    print(answer_question(question))
