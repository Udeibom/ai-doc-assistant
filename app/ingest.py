import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict

from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from loaders import load_pdfs
from chunking import chunk_documents

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# -----------------------------
# Paths (ONE source of truth)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = BASE_DIR / "data" / "raw_pdfs"
VECTOR_STORE_DIR = BASE_DIR / "storage" / "vector_store"
MANIFEST_PATH = VECTOR_STORE_DIR / "ingestion_manifest.json"

VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Embedding Model
# -----------------------------
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# Utilities
# -----------------------------
def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def load_manifest() -> Dict[str, str]:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}

def save_manifest(manifest: Dict[str, str]):
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

# -----------------------------
# Ingestion
# -----------------------------
def ingest():
    manifest = load_manifest()
    new_documents: List[Document] = []

    logger.info("Scanning PDF directory...")

    for pdf_path in PDF_DIR.glob("*.pdf"):
        pdf_hash = file_hash(pdf_path)

        if pdf_hash in manifest:
            logger.info(f"Skipping already ingested: {pdf_path.name}")
            continue

        logger.info(f"Ingesting: {pdf_path.name}")

        docs = load_pdfs([pdf_path])
        chunks = chunk_documents(docs)

        for c in chunks:
            new_documents.append(
                Document(
                    text=c["text"],
                    metadata={
                        "source_file": c["source_file"],
                        "page_number": c["page_number"],
                        "chunk_id": c["chunk_id"],
                    }
                )
            )

        manifest[pdf_hash] = pdf_path.name

    if not new_documents:
        logger.info("No new documents found.")
        return

    logger.info(f"Embedding {len(new_documents)} chunks...")

    # 🔑 THIS is the correct, stable way
    index = VectorStoreIndex.from_documents(new_documents)

    logger.info("Persisting index...")
    index.storage_context.persist(persist_dir=VECTOR_STORE_DIR)

    save_manifest(manifest)
    logger.info("Ingestion complete.")

if __name__ == "__main__":
    ingest()
