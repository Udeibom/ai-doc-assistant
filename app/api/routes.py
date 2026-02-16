from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional

from app.core.rate_limiter import limiter
from app.core.security import (
    get_role_from_request,
    require_user_or_admin,
    require_admin,
)

router = APIRouter()

# ==========================
# Request / Response Models
# ==========================

class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str


class IngestRequest(BaseModel):
    text: str
    source_file: str


# ==========================
# Guardrails
# ==========================

MAX_QUESTION_LENGTH = 500


def validate_question(question: str):
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if len(question) > MAX_QUESTION_LENGTH:
        raise HTTPException(
            status_code=413,
            detail="Question too long"
        )


# ==========================
# Health / Readiness
# ==========================

@router.get("/health")
def health():
    return {"status": "ok"}


# ==========================
# Query Endpoints (User + Admin)
# ==========================

@router.post("/ask", response_model=AnswerResponse)
@limiter.limit("10/minute")
def ask_question(request: Request, payload: QuestionRequest):
    # Lazy import to prevent startup blocking
    from app.core.qa_engine import answer_question

    role = get_role_from_request(request)
    require_user_or_admin(role)

    validate_question(payload.question)

    answer = answer_question(payload.question)
    return {"answer": answer}


@router.post("/ask/stream")
@limiter.limit("5/minute")
def ask_question_stream(request: Request, payload: QuestionRequest):
    # Lazy import
    from app.core.qa_engine import answer_question_stream

    role = get_role_from_request(request)
    require_user_or_admin(role)

    validate_question(payload.question)

    return StreamingResponse(
        answer_question_stream(payload.question),
        media_type="text/event-stream",
    )


# ==========================
# Ingestion Endpoint (User + Admin)
# ==========================

@router.post("/ingest")
def user_ingest(
    request: Request,
    file: UploadFile = File(None),
    text: str = Form(None),
    source_file: str = Form(None),
):
    """
    Allows users (non-admin) to upload PDF or paste text.
    """
    role = get_role_from_request(request)
    require_user_or_admin(role)

    if file is None and (text is None or text.strip() == ""):
        raise HTTPException(status_code=400, detail="No document or text provided")

    if file:
        from PyPDF2 import PdfReader

        try:
            reader = PdfReader(file.file)
            doc_text = "\n\n".join(
                [page.extract_text() or "" for page in reader.pages]
            )
            doc_name = source_file or file.filename
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"PDF processing failed: {e}"
            )
    else:
        doc_text = text
        doc_name = source_file or "user_text"

    from app.core.ingestion import ingest_text

    ingest_text(text=doc_text, source_file=doc_name)

    return {"status": f"Document ingested successfully: {doc_name}"}
