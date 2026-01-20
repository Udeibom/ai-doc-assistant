import os
import logging
from threading import Lock

from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

logger = logging.getLogger(__name__)

_embed_model = None
_llm = None
_lock = Lock()


def get_models():
    global _embed_model, _llm

    if _embed_model and _llm:
        return _embed_model, _llm

    with _lock:
        if _embed_model is None:
            logger.info("🔧 Loading embedding model (CPU)")
            _embed_model = HuggingFaceEmbedding(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                device="cpu",
            )

        if _llm is None:
            logger.info("🔧 Initializing Groq LLM")
            _llm = Groq(
                model="llama-3.1-8b-instant",
                temperature=0.1,
                max_tokens=256,
                api_key=os.environ.get("GROQ_API_KEY"),
            )

        Settings.embed_model = _embed_model
        Settings.llm = _llm

    return _embed_model, _llm
