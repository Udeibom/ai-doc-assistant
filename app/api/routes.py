from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.qa_engine import (
    answer_question,
    answer_question_stream
)
from app.core.rate_limiter import limiter
from app.core.security import (
    get_role_from_request,
    require_user_or_admin,
    require_admin
)
from app.core.ingestion import ingest_text

router = APIRouter()


# Request / Response Models

class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str


class IngestRequest(BaseModel):
    text: str
    source_file: str


# Guardrails

MAX_QUESTION_LENGTH = 500


def validate_question(question: str):
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if len(question) > MAX_QUESTION_LENGTH:
        raise HTTPException(
            status_code=413,
            detail="Question too long"
        )


# Query Endpoints (User + Admin)

@router.post("/ask", response_model=AnswerResponse)
@limiter.limit("10/minute")
def ask_question(request: Request, payload: QuestionRequest):
    role = get_role_from_request(request)
    require_user_or_admin(role)

    validate_question(payload.question)

    answer = answer_question(payload.question)
    return {"answer": answer}


@router.post("/ask/stream")
@limiter.limit("5/minute")
def ask_question_stream(request: Request, payload: QuestionRequest):
    role = get_role_from_request(request)
    require_user_or_admin(role)

    validate_question(payload.question)

    return StreamingResponse(
        answer_question_stream(payload.question),
        media_type="text/event-stream"
    )


# Admin-Only Ingestion

@router.post("/admin/ingest")
def ingest_document(request: Request, payload: IngestRequest):
    role = get_role_from_request(request)
    require_admin(role)

    ingest_text(
        text=payload.text,
        source_file=payload.source_file
    )

    return {"status": "Document ingested successfully"}
