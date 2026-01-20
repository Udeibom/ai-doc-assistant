import logging
import time
from pathlib import Path
from statistics import mean
from threading import Lock
from typing import Generator, Optional

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.retrievers import VectorIndexRetriever

from app.prompts import SYSTEM_PROMPT, QUERY_REWRITE_PROMPT
from app.core.model_manager import get_models

# ===============================
# Logging
# ===============================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===============================
# Paths
# ===============================

BASE_DIR = Path(__file__).resolve().parents[2]
VECTOR_STORE_DIR = BASE_DIR / "storage" / "vector_store"

# ===============================
# Guardrails
# ===============================

MIN_SIMILARITY = 0.35
MIN_CONFIDENCE = 0.30

# ===============================
# Retriever (Lazy + Thread-safe)
# ===============================

_RETRIEVER: Optional[VectorIndexRetriever] = None
_RETRIEVER_LOCK = Lock()


def _load_retriever(top_k: int = 2) -> Optional[VectorIndexRetriever]:
    logger.info("📦 Loading vector index (lazy)...")

    if not VECTOR_STORE_DIR.exists():
        logger.warning(
            f"⚠️ Vector store not found at {VECTOR_STORE_DIR}. "
            "Retrieval unavailable."
        )
        return None

    storage_context = StorageContext.from_defaults(
        persist_dir=VECTOR_STORE_DIR
    )

    index = load_index_from_storage(storage_context)

    return VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k,
    )


def get_retriever() -> Optional[VectorIndexRetriever]:
    global _RETRIEVER

    if _RETRIEVER is not None:
        return _RETRIEVER

    with _RETRIEVER_LOCK:
        if _RETRIEVER is None:
            _RETRIEVER = _load_retriever()

    return _RETRIEVER


def reload_retriever():
    global _RETRIEVER
    with _RETRIEVER_LOCK:
        logger.info("🔄 Reloading vector retriever")
        _RETRIEVER = _load_retriever()


# ===============================
# Query Rewriting
# ===============================

def rewrite_query(question: str) -> str:
    _, llm = get_models()

    prompt = QUERY_REWRITE_PROMPT.format(question=question)
    rewritten = llm.complete(prompt).text.strip()

    if not rewritten or len(rewritten) > 200:
        return question

    return rewritten


# ===============================
# Context Builder
# ===============================

def build_context(nodes) -> str:
    blocks = []

    for node in nodes:
        meta = node.node.metadata or {}
        source = meta.get("source_file", "unknown")
        page = meta.get("page_number", "unknown")

        blocks.append(
            f"[source: {source}, page: {page}]\n"
            f"{node.node.text.strip()}"
        )

    return "\n\n".join(blocks)


# ===============================
# Confidence Computation
# ===============================

def compute_confidence(nodes) -> float:
    if not nodes:
        return 0.0

    scores = [n.score for n in nodes if n.score is not None]
    if not scores:
        return 0.0

    avg = mean(scores)
    avg = max(0.0, min(avg, 1.0))

    return round(avg, 3)


# ===============================
# QA — Non-Streaming
# ===============================

def answer_question(question: str) -> str:
    start = time.time()

    # Lazy model load (safe)
    _, llm = get_models()

    rewritten = rewrite_query(question)
    logger.info(f"🔍 Rewritten query: {rewritten}")

    retriever = get_retriever()
    if retriever is None:
        return "Knowledge base not initialized yet.\n\nConfidence: 0.00"

    nodes = retriever.retrieve(rewritten)

    strong_nodes = [
        n for n in nodes
        if n.score is not None and n.score >= MIN_SIMILARITY
    ]

    if not strong_nodes:
        return "I don’t know based on the provided documents.\n\nConfidence: 0.00"

    context = build_context(strong_nodes)

    prompt = f"""{SYSTEM_PROMPT}

Context (with sources):
{context}

Question:
{question}

Answer (with citations):
"""

    answer = llm.complete(prompt).text.strip()

    if "[source:" not in answer:
        return "I don’t know based on the provided documents.\n\nConfidence: 0.00"

    confidence = compute_confidence(strong_nodes)

    if confidence < MIN_CONFIDENCE:
        return "I don’t know based on the provided documents.\n\nConfidence: 0.00"

    logger.info(f"✅ Completed in {time.time() - start:.2f}s")

    return f"{answer}\n\nConfidence: {confidence:.2f}"


# ===============================
# QA — Streaming (SSE)
# ===============================

def answer_question_stream(question: str) -> Generator[str, None, None]:
    _, llm = get_models()

    rewritten = rewrite_query(question)
    logger.info(f"🔍 Rewritten query (stream): {rewritten}")

    retriever = get_retriever()
    if retriever is None:
        yield "data: Knowledge base not initialized yet.\n\n"
        yield "event: end\ndata: [DONE]\n\n"
        return

    nodes = retriever.retrieve(rewritten)

    strong_nodes = [
        n for n in nodes
        if n.score is not None and n.score >= MIN_SIMILARITY
    ]

    if not strong_nodes:
        yield "data: I don’t know based on the provided documents.\n\n"
        yield "event: end\ndata: [DONE]\n\n"
        return

    context = build_context(strong_nodes)

    prompt = f"""{SYSTEM_PROMPT}

Context (with sources):
{context}

Question:
{question}

Answer (with citations):
"""

    stream = llm.stream_complete(prompt)

    for chunk in stream:
        if chunk.delta:
            yield f"data: {chunk.delta}\n\n"

    confidence = compute_confidence(strong_nodes)
    yield f"event: metadata\ndata: Confidence: {confidence:.2f}\n\n"
    yield "event: end\ndata: [DONE]\n\n"
