from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import Document
from llama_index.core.settings import Settings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
VECTOR_STORE_DIR = BASE_DIR / "storage" / "vector_store"

def ingest_text(text: str, source_file: str):
    document = Document(
        text=text,
        metadata={"source_file": source_file}
    )

    storage_context = StorageContext.from_defaults(
        persist_dir=VECTOR_STORE_DIR
    )

    index = VectorStoreIndex.from_documents(
        [document],
        storage_context=storage_context,
        embed_model=Settings.embed_model
    )

    index.storage_context.persist()
