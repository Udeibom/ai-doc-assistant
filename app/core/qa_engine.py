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


# Logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths

BASE_DIR = Path(__file__).resolve().parents[2]
VECTOR_STORE_DIR = BASE_DIR / "storage" / "vector_store"


# Guardrail Constants

MIN_SIMILARITY = 0.35
MIN_CONFIDENCE = 0.30


# Embeddings

Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# LLM

Settings.llm = Groq(
    model="llama-3.1-8b-instant",
    temperature=0.1,
    max_tokens=256,
)


# Retriever Loader

def load_retriever(top_k: int = 2):
    logger.info("Loading vector index...")

    if not VECTOR_STORE_DIR.exists():
        raise RuntimeError(
            f"Vector store not found at {VECTOR_STORE_DIR}. "
            "Ensure the index is built before starting the API."
        )

    storage_context = StorageContext.from_defaults(
        persist_dir=VECTOR_STORE_DIR
    )

    index = load_index_from_storage(storage_context)

    return VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k,
    )


# Load ONCE

RETRIEVER = load_retriever()


# Query Rewriting

def rewrite_query(question: str) -> str:
    """
    Rewrite query for better retrieval (NOT answering)
    """
    prompt = QUERY_REWRITE_PROMPT.format(question=question)
    rewritten = Settings.llm.complete(prompt).text.strip()

    # Guardrail: prevent verbose or hijacked rewrites
    if not rewritten or len(rewritten) > 200:
        return question

    return rewritten


# Context Builder

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


# Confidence computation

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


# QA Function (Non-streaming)

def answer_question(query: str) -> str:
    print("⏳ Running query...")
    start = time.time()

    # 🔹 Rewrite query
    rewritten_query = rewrite_query(query)
    logger.info(f"Rewritten query: {rewritten_query}")

    # 🔹 Retrieve
    nodes = RETRIEVER.retrieve(rewritten_query)

    #similarity threshold
    strong_nodes = [
        n for n in nodes
        if n.score is not None and n.score >= MIN_SIMILARITY
    ]

    if not strong_nodes:
        return "I don’t know based on the provided documents.\n\nConfidence: 0.00"

    # 🔹 Build locked context
    context = build_context(strong_nodes)

    # 🔹 Answer using ORIGINAL query
    prompt = f"""{SYSTEM_PROMPT}

Context (with sources):
{context}

Question:
{query}

Answer (with citations):
"""

    answer = Settings.llm.complete(prompt).text.strip()

    #citation enforcement
    if "[source:" not in answer:
        return "I don’t know based on the provided documents.\n\nConfidence: 0.00"

    confidence = compute_confidence(strong_nodes, answer)

    #confidence gate
    if confidence < MIN_CONFIDENCE:
        return "I don’t know based on the provided documents.\n\nConfidence: 0.00"

    print(f"✅ Done in {time.time() - start:.2f}s")

    return f"{answer}\n\nConfidence: {confidence:.2f}"


# QA Function (Streaming)

def answer_question_stream(question: str):
    """
    Streams the answer token-by-token using SSE.
    """

    #pre-check before streaming
    rewritten = rewrite_query(question)
    logger.info(f"Rewritten query (stream): {rewritten}")

    nodes = RETRIEVER.retrieve(rewritten)

    strong_nodes = [
        n for n in nodes
        if n.score is not None and n.score >= MIN_SIMILARITY
    ]

    if not strong_nodes:
        yield "data: I don’t know based on the provided documents.\n\n"
        yield "event: end\ndata: [DONE]\n\n"
        return

    #lock context BEFORE streaming
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


# CLI Entry

if __name__ == "__main__":
    question = "What notice period is required before changes to the General Terms take effect?"
    print("\n📄 Question:", question)
    print("\n🧠 Answer:\n")
    print(answer_question(question))
