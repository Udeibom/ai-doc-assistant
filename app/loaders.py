import logging
from pathlib import Path
from typing import List

from PyPDF2 import PdfReader
from llama_index.core import Document

logger = logging.getLogger(__name__)


def load_pdfs(pdf_paths: List[Path]) -> List[Document]:
    """
    Load PDFs and extract text page by page.
    """
    documents = []

    for pdf_path in pdf_paths:
        try:
            reader = PdfReader(pdf_path)
            num_pages = len(reader.pages)

            logger.info(f"Loaded {pdf_path.name} with {num_pages} pages")

            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text()

                if not text or not text.strip():
                    continue

                documents.append(
                    Document(
                        text=text,
                        metadata={
                            "source_file": pdf_path.name,
                            "page_number": page_number,
                        }
                    )
                )

        except Exception as e:
            logger.error(f"Failed to load {pdf_path.name}: {e}")

    return documents
