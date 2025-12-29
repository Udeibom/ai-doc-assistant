import os
import logging
from pathlib import Path
from llama_index.readers.file import PDFReader

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
# PDF Loading Logic
# -----------------------------
def load_pdfs(pdf_dir: Path):
    """
    Load and extract text from all PDFs in a directory.

    Args:
        pdf_dir (Path): Path to directory containing PDF files

    Returns:
        list[str]: Extracted text from all PDFs
    """
    reader = PDFReader()
    extracted_texts = []

    if not pdf_dir.exists():
        logger.error(f"PDF directory not found: {pdf_dir}")
        return extracted_texts

    pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        logger.warning("No PDF files found.")
        return extracted_texts

    for pdf_file in pdf_files:
        try:
            logger.info(f"Loading PDF: {pdf_file.name}")

            documents = reader.load_data(file=pdf_file)

            logger.info(
                f"Extracted {len(documents)} pages from {pdf_file.name}"
            )

            for page_num, doc in enumerate(documents, start=1):
                text = doc.text.strip()

                if text:
                    extracted_texts.append(text)
                else:
                    logger.warning(
                        f"Empty text on page {page_num} of {pdf_file.name}"
                    )

        except Exception as e:
            logger.error(
                f"Failed to process {pdf_file.name}: {str(e)}"
            )
            continue

    return extracted_texts

# -----------------------------
# Script Entry Point
# -----------------------------
if __name__ == "__main__":
    texts = load_pdfs(PDF_DIR)

    logger.info(f"Total extracted text chunks: {len(texts)}")

    print("\n" + "=" * 80)
    print("SAMPLE EXTRACTED TEXT")
    print("=" * 80)

    for i, text in enumerate(texts[:3], start=1):
        print(f"\n--- Document Chunk {i} ---\n")
        print(text[:1000])  # Print first 1000 chars
