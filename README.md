# Enterprise AI Assistant for Document Intelligence

## Overview

Organizations store critical knowledge across large collections of
PDFs---policies, contracts, manuals, and internal documentation.\
While accessible, extracting precise information is slow, manual, and
inefficient.

This project implements a **production-oriented Retrieval-Augmented
Generation (RAG) system** that enables users to ask natural language
questions and receive **accurate, source-grounded answers strictly
derived from enterprise documents**.

The system is designed with **reliability, traceability, and
hallucination prevention as first-class concerns**, making it suitable
for real-world enterprise use.

------------------------------------------------------------------------

## Key Features

-   📄 End-to-end document ingestion and processing pipeline\
-   🔍 Semantic search using vector embeddings (FAISS / Chroma)\
-   🧠 LLM-based reasoning grounded strictly in retrieved context\
-   📌 Explicit source citation for every response\
-   🚫 Hallucination control with "I don't know" fallback\
-   ⚡ FastAPI-based inference API for real-time interaction\
-   🔁 Persistent vector store with retriever reload support

------------------------------------------------------------------------

## Architecture

The system is designed as a **modular ML pipeline** with clear
separation between ingestion and querying.

### Ingestion Pipeline

    PDF Documents
    → Text Extraction
    → Chunking (with overlap)
    → Embedding Generation
    → Vector Store (FAISS)

### Query Pipeline

    User Query
    → Query Embedding
    → Top-K Retrieval
    → Context Assembly
    → LLM Reasoning
    → Answer + Source Citations

------------------------------------------------------------------------

## Quick Start

### Run Locally

    git clone https://github.com/yourusername/ai-doc-assistant.git
    cd ai-doc-assistant

    pip install -r requirements.txt
    uvicorn app.api:app --reload

### Example API Request

    POST /query
    {
      "question": "What does the policy say about refunds?"
    }

------------------------------------------------------------------------

## Deployment

The system is designed for deployment as an API service and has been
deployed using Render (free tier).

> Note: Free-tier deployments may experience cold-start latency.

------------------------------------------------------------------------

## Tech Stack

-   **Language:** Python\
-   **API:** FastAPI\
-   **ML / RAG:** LlamaIndex\
-   **Vector Store:** FAISS / ChromaDB\
-   **LLM:** OpenAI / Groq (configurable)\
-   **Embeddings:** HuggingFace / OpenAI\
-   **Deployment:** Render, Docker (optional)

------------------------------------------------------------------------

## Key Engineering Decisions

-   Chunk overlap (50--100 tokens) to preserve context continuity\
-   Strict context-only answering to prevent hallucinations\
-   Explicit refusal mechanism for low-confidence queries\
-   Retriever reload after ingestion\
-   Modular pipeline separation (ingestion vs querying)\
-   Confidence scoring using similarity metrics

------------------------------------------------------------------------

## Evaluation

-   Retrieval relevance (similarity scores)\
-   Answer grounding (citation presence)\
-   Hallucination reduction via strict constraints

Future work includes automated evaluation using LLM-based scoring.

------------------------------------------------------------------------

## Project Structure

    ai-doc-assistant/
    │── app/
    │   ├── api.py
    │   ├── ingest.py
    │   ├── qa_engine.py
    │   └── prompts.py
    │── data/
    │── tests/
    │── README.md
    │── requirements.txt
    │── docker-compose.yml

------------------------------------------------------------------------

## Environment Variables

    OPENAI_API_KEY=your_api_key

------------------------------------------------------------------------

## Design Principles

-   Accuracy over creativity\
-   Transparency over convenience\
-   Reliability over novelty\
-   Production-oriented system design

------------------------------------------------------------------------

## Future Improvements

-   Streaming responses\
-   Query rewriting\
-   Role-based access control\
-   Rate limiting\
-   Prompt versioning
