import logging
import time
from pathlib import Path
from statistics import mean

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.settings import Settings

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

from app.prompts import SYSTEM_PROMPT, QUERY_REWRITE_PROMPT



# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
VECTOR_STORE_DIR = BASE_DIR / "storage" / "vector_store"


# -----------------------------
# Embeddings
# -----------------------------
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# LLM
# -----------------------------
Settings.llm = Groq(
    model="llama-3.1-8b-instant",
    temperature=0.1,
    max_tokens=256,
)

# -----------------------------
# Retriever Loader
# -----------------------------
def load_retriever(top_k: int = 2):
    logger.info("Loading vector index...")

    storage_context = StorageContext.from_defaults(
        persist_dir=VECTOR_STORE_DIR
    )

    index = load_index_from_storage(storage_context)

    if not VECTOR_STORE_DIR.exists():
        raise RuntimeError(
            f"Vector store not found at {VECTOR_STORE_DIR}. "
            "Ensure the index is built before starting the API."
        )


    return VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k,
    )

# -----------------------------
# Load ONCE
# -----------------------------
RETRIEVER = load_retriever()

# -----------------------------
# Query Rewriting
# -----------------------------
def rewrite_query(question: str) -> str:
    """
    Rewrite query for better retrieval (NOT answering)
    """
    prompt = QUERY_REWRITE_PROMPT.format(question=question)
    suggests = Settings.llm.complete(prompt).text.strip()

    # Defensive fallback
    if not suggests:
        return question

    return suggests

# -----------------------------
# Confidence computation
# -----------------------------
def compute_confidence(nodes, answer: str) -> float:
    if not nodes:
        return 0.0

    if "[source:" not in answer:
        return 0.0

    scores = [n.score for n in nodes if n.score is not None]

    if not scores:
        return 0.0

    avg_similarity = mean(scores)
    avg_similarity = max(0.0, min(avg_similarity, 1.0))

    return round(avg_similarity, 3)

# -----------------------------
# QA Function
# -----------------------------
def answer_question(query: str) -> str:
    print("⏳ Running query...")
    start = time.time()

    # 🔹 Rewrite query for retrieval
    rewritten_query = rewrite_query(query)
    logger.info(f"Rewritten query: {rewritten_query}")

    # 🔹 Retrieve with rewritten query
    nodes = RETRIEVER.retrieve(rewritten_query)

    if not nodes:
        return "I don’t know based on the provided documents.\n\nConfidence: 0.00"

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

    # 🔹 Answer using ORIGINAL query
    prompt = f"""{SYSTEM_PROMPT}

Context (with sources):
{context}

Question:
{query}

Answer (with citations):
"""

    answer = Settings.llm.complete(prompt).text.strip()

    confidence = compute_confidence(nodes, answer)

    print(f"✅ Done in {time.time() - start:.2f}s")

    if confidence == 0.0:
        return "I don’t know based on the provided documents.\n\nConfidence: 0.00"

    return f"{answer}\n\nConfidence: {confidence:.2f}"


def answer_question_stream(question: str):
    """
    Streams the answer token-by-token using SSE.
    """
    # 🔹 Rewrite query for retrieval
    rewritten_query = rewrite_query(question)
    logger.info(f"Rewritten query (stream): {rewritten_query}")

    # 🔹 Retrieve
    nodes = RETRIEVER.retrieve(rewritten_query)

    if not nodes:
        yield "data: I don’t know based on the provided documents.\n\n"
        yield "event: end\ndata: [DONE]\n\n"
        return

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

    prompt = f"""{SYSTEM_PROMPT}

Context (with sources):
{context}

Question:
{question}

Answer (with citations):
"""

    # 🔹 Stream from LLM
    stream = Settings.llm.stream_complete(prompt)

    for chunk in stream:
        if chunk.delta:
            yield f"data: {chunk.delta}\n\n"

    # 🔹 Final metadata
    confidence = compute_confidence(nodes, "")
    yield f"event: metadata\ndata: Confidence: {confidence:.2f}\n\n"
    yield "event: end\ndata: [DONE]\n\n"


# -----------------------------
# CLI Entry
# -----------------------------
if __name__ == "__main__":
    question = "What notice period is required before changes to the General Terms take effect?"
    print("\n📄 Question:", question)
    print("\n🧠 Answer:\n")
    print(answer_question(question))
