from typing import List, Dict
import uuid

from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core import Document


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 700,
    chunk_overlap: int = 100,
) -> List[Dict]:
    """
    Chunk documents into token-based chunks with metadata.
    """
    splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = []

    for doc in documents:
        nodes = splitter.get_nodes_from_documents([doc])

        for node in nodes:
            chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "text": node.text,
                "source_file": doc.metadata["source_file"],
                "page_number": doc.metadata["page_number"],
            })

    return chunks
