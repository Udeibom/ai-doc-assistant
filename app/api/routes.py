from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.qa_engine import answer_question

router = APIRouter()


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str


@router.post("/ask", response_model=AnswerResponse)
def ask_question(payload: QuestionRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    answer = answer_question(payload.question)

    return {"answer": answer}
