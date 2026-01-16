import logging
from typing import List

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.openai import OpenAI


logger = logging.getLogger(__name__)


VECTOR_STORE_DIR = "storage/vector_store"


def load_query_engine(top_k: int = 5):
    """
    Load persisted vector index and create a query engine.
    """
    logger.info("Loading vector store from disk...")

    storage_context = StorageContext.from_defaults(
        persist_dir=VECTOR_STORE_DIR
    )

    index = load_index_from_storage(storage_context)

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k,
    )

    llm = OpenAI(
        model="gpt-4o-mini",
        temperature=0.0
    )

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        llm=llm,
    )

    return query_engine


def answer_question(query: str, top_k: int = 5) -> str:
    """
    Answer a question using retrieved document context.
    """
    logger.info(f"Received query: {query}")

    query_engine = load_query_engine(top_k=top_k)

    response = query_engine.query(query)

    return str(response)
