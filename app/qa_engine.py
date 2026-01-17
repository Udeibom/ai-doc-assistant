import logging
import time
from pathlib import Path

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.settings import Settings

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

from prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


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
def load_retriever(top_k: int = 2):
    logger.info("Loading vector index...")

    storage_context = StorageContext.from_defaults(
        persist_dir=VECTOR_STORE_DIR
    )

    index = load_index_from_storage(storage_context)

    return VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k,
    )



# -----------------------------
# Load ONCE
# -----------------------------
RETRIEVER = load_retriever()


# -----------------------------
# QA Function
# -----------------------------
def answer_question(query: str) -> str:
    print("⏳ Running query...")
    start = time.time()

    nodes = RETRIEVER.retrieve(query)

    if not nodes:
        return "I don’t know based on the provided documents."

    # 🔹 Build citation-aware context
    context_blocks = []
    for node in nodes:
        meta = node.node.metadata
        source = meta.get("source_file", "unknown")
        page = meta.get("page_number", "unknown")

        block = (
            f"[source: {source}, page: {page}]\n"
            f"{node.node.text.strip()}"
        )
        context_blocks.append(block)

    context = "\n\n".join(context_blocks)

    # 🔹 Construct strict prompt
    prompt = f"""{SYSTEM_PROMPT}

Context (with sources):
{context}

Question:
{query}

Answer (with citations):
"""

    # 🔹 Call LLM directly
    answer = Settings.llm.complete(prompt).text.strip()

    print(f"✅ Done in {time.time() - start:.2f}s")

    # Hard citation guardrail
    if "[source:" not in answer:
        return "I don’t know based on the provided documents."

    return answer


# -----------------------------
# CLI Entry
# -----------------------------
if __name__ == "__main__":
    question = "What notice period is required before changes to the General Terms take effect?"
    print("\n📄 Question:", question)
    print("\n🧠 Answer:\n")
    print(answer_question(question))
