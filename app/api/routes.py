from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.qa_engine import answer_question, answer_question_stream
from app.core.rate_limiter import limiter

router = APIRouter()


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str


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


@router.post("/ask", response_model=AnswerResponse)
@limiter.limit("10/minute")
def ask_question(request: Request, payload: QuestionRequest):
    validate_question(payload.question)

    answer = answer_question(payload.question)
    return {"answer": answer}


@router.post("/ask/stream")
@limiter.limit("5/minute")
def ask_question_streaming(request: Request, payload: QuestionRequest):
    validate_question(payload.question)

    return StreamingResponse(
        answer_question_stream(payload.question),
        media_type="text/event-stream"
    )
