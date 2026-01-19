import logging
import os
import time
from pathlib import Path
from statistics import mean
from threading import Lock

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

from app.prompts import SYSTEM_PROMPT, QUERY_REWRITE_PROMPT


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
# Guardrail Constants
# ===============================

MIN_SIMILARITY = 0.35
MIN_CONFIDENCE = 0.30


# ===============================
# Lazy Model Initialization
# ===============================

_EMBED_MODEL = None
_LLM = None
_MODEL_LOCK = Lock()


def init_models():
    """
    Lazily initialize embedding + LLM models.
    Safe for CPU-only environments (Render).
    """
    global _EMBED_MODEL, _LLM

    if _EMBED_MODEL and _LLM:
        return

    with _MODEL_LOCK:
        if _EMBED_MODEL is None:
            logger.info("🔧 Initializing embedding model (CPU)...")
            _EMBED_MODEL = HuggingFaceEmbedding(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                device="cpu",
            )
            Settings.embed_model = _EMBED_MODEL

        if _LLM is None:
            logger.info("🔧 Initializing Groq LLM...")
            _LLM = Groq(
                model="llama-3.1-8b-instant",
                temperature=0.1,
                max_tokens=256,
                api_key=os.environ.get("GROQ_API_KEY"),
            )
            Settings.llm = _LLM


# ===============================
# Retriever Loader
# ===============================

_RETRIEVER = None
_RETRIEVER_LOCK = Lock()


def load_retriever(top_k: int = 2):
    logger.info("📦 Loading vector index...")

    if not VECTOR_STORE_DIR.exists():
        logger.warning(
            f"⚠️ Vector store not found at {VECTOR_STORE_DIR}. "
            "Retrieval will be unavailable."
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


def get_retriever():
    global _RETRIEVER

    if _RETRIEVER is None:
        with _RETRIEVER_LOCK:
            if _RETRIEVER is None:
                _RETRIEVER = load_retriever()

    return _RETRIEVER


def reload_retriever():
    global _RETRIEVER

    with _RETRIEVER_LOCK:
        logger.info("🔄 Reloading vector retriever...")
        _RETRIEVER = load_retriever()


# ===============================
# Query Rewriting
# ===============================

def rewrite_query(question: str) -> str:
    init_models()

    prompt = QUERY_REWRITE_PROMPT.format(question=question)
    rewritten = Settings.llm.complete(prompt).text.strip()

    if not rewritten or len(rewritten) > 200:
        return question

    return rewritten


# ===============================
# Context Builder
# ===============================

def build_context(nodes) -> str:
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

    return "\n\n".join(context_blocks)


# ===============================
# Confidence Computation
# ===============================

def compute_confidence(nodes, answer: str) -> float:
    if not nodes:
        return 0.0

    scores = [n.score for n in nodes if n.score is not None]
    if not scores:
        return 0.0

    avg_similarity = mean(scores)
    avg_similarity = max(0.0, min(avg_similarity, 1.0))

    return round(avg_similarity, 3)


# ===============================
# QA Function (Non-Streaming)
# ===============================

def answer_question(query: str) -> str:
    init_models()
    start = time.time()

    rewritten_query = rewrite_query(query)
    logger.info(f"🔍 Rewritten query: {rewritten_query}")

    retriever = get_retriever()
    if retriever is None:
        return "Knowledge base not initialized yet.\n\nConfidence: 0.00"

    nodes = retriever.retrieve(rewritten_query)

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
{query}

Answer (with citations):
"""

    answer = Settings.llm.complete(prompt).text.strip()

    if "[source:" not in answer:
        return "I don’t know based on the provided documents.\n\nConfidence: 0.00"

    confidence = compute_confidence(strong_nodes, answer)

    if confidence < MIN_CONFIDENCE:
        return "I don’t know based on the provided documents.\n\nConfidence: 0.00"

    logger.info(f"✅ Completed in {time.time() - start:.2f}s")

    return f"{answer}\n\nConfidence: {confidence:.2f}"


# ===============================
# QA Function (Streaming)
# ===============================

def answer_question_stream(question: str):
    init_models()

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

    stream = Settings.llm.stream_complete(prompt)

    for chunk in stream:
        if chunk.delta:
            yield f"data: {chunk.delta}\n\n"

    confidence = compute_confidence(strong_nodes, "")
    yield f"event: metadata\ndata: Confidence: {confidence:.2f}\n\n"
    yield "event: end\ndata: [DONE]\n\n"
