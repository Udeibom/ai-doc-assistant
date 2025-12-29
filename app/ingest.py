import os
import logging
from pathlib import Path
from typing import List, Dict

from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import TokenTextSplitter

# -----------------------------
# Logging Configuration
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = BASE_DIR / "data" / "raw_pdfs"

# -----------------------------
# Chunking Configuration
# -----------------------------
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

# -----------------------------
# PDF Loading
# -----------------------------
def load_pdfs(pdf_dir: Path):
    reader = PDFReader()
    documents = []

    for pdf_file in pdf_dir.glob("*.pdf"):
        try:
            logger.info(f"Loading PDF: {pdf_file.name}")

            pages = reader.load_data(file=pdf_file)
            logger.info(f"Extracted {len(pages)} pages")

            for page_num, page in enumerate(pages, start=1):
                text = page.text.strip()

                if not text:
                    continue

                documents.append({
                    "text": text,
                    "source_file": pdf_file.name,
                    "page_number": page_num,
                })

        except Exception as e:
            logger.error(f"Failed to load {pdf_file.name}: {e}")

    return documents

# -----------------------------
# Chunking Logic
# -----------------------------
def chunk_documents(
    documents: List[Dict]
) -> List[Dict]:
    splitter = TokenTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = []
    chunk_id = 0

    for doc in documents:
        text_chunks = splitter.split_text(doc["text"])

        for chunk_text in text_chunks:
            chunk_id += 1
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "source_file": doc["source_file"],
                "page_number": doc["page_number"],
            })

    return chunks

# -----------------------------
# Script Entry Point
# -----------------------------
if __name__ == "__main__":
    documents = load_pdfs(PDF_DIR)
    logger.info(f"Loaded {len(documents)} pages")

    chunks = chunk_documents(documents)
    logger.info(f"Created {len(chunks)} chunks")

    print("\n" + "=" * 80)
    print("SAMPLE CHUNKS WITH METADATA")
    print("=" * 80)

    for chunk in chunks[:3]:
        print(f"""
Chunk ID: {chunk['chunk_id']}
Source File: {chunk['source_file']}
Page Number: {chunk['page_number']}
Text Preview:
{chunk['text'][:500]}
""")
