from fastapi import APIRouter, HTTPException, Request, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import shutil
import os
import threading

from app.core.rate_limiter import limiter
from app.core.security import get_role_from_request, require_user_or_admin
from app.core.ingestion import ingest_text
from app.core.model_manager import get_models

router = APIRouter()

# ==========================
# Request / Response Models
# ==========================

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

# ==========================
# Guardrails
# ==========================

MAX_QUESTION_LENGTH = 500

def validate_question(question: str):
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    if len(question) > MAX_QUESTION_LENGTH:
        raise HTTPException(status_code=413, detail="Question too long")

# ==========================
# Health / Readiness
# ==========================

@router.get("/health")
def health():
    return {"status": "ok"}

# ==========================
# QA Endpoints (User)
# ==========================

@router.post("/ask", response_model=AnswerResponse)
@limiter.limit("10/minute")
def ask_question(request: Request, payload: QuestionRequest):
    from app.core.qa_engine import answer_question

    role = get_role_from_request(request)
    require_user_or_admin(role)

    validate_question(payload.question)
    answer = answer_question(payload.question)
    return {"answer": answer}


@router.post("/ask/stream")
@limiter.limit("5/minute")
def ask_question_stream(request: Request, payload: QuestionRequest):
    from app.core.qa_engine import answer_question_stream

    role = get_role_from_request(request)
    require_user_or_admin(role)

    validate_question(payload.question)

    return StreamingResponse(
        answer_question_stream(payload.question),
        media_type="text/event-stream",
    )

# ==========================
# User PDF / Text Ingestion
# ==========================

UPLOAD_DIR = Path(__file__).resolve().parents[2] / "storage" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/ingest")
def ingest_document(request: Request, file: UploadFile = File(...)):
    """
    Public-friendly PDF/text upload endpoint.
    Accepts PDF or plain text files.
    """
    role = get_role_from_request(request)
    require_user_or_admin(role)

    # Save uploaded file
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract text from PDF or TXT
    if file.filename.lower().endswith(".pdf"):
        from PyPDF2 import PdfReader
        pdf = PdfReader(str(file_path))
        text = "\n\n".join([page.extract_text() or "" for page in pdf.pages])
    elif file.filename.lower().endswith(".txt"):
        text = file_path.read_text(encoding="utf-8")
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    # Ingest into vector store
    ingest_text(text=text, source_file=file.filename)

    return {"status": f"File '{file.filename}' ingested successfully."}

# ==========================
# Auto Warmup on Startup
# ==========================

@router.on_event("startup")
def auto_warmup():
    def warmup():
        try:
            get_models()
            print("✅ Models warmed up successfully")
        except Exception as e:
            print(f"⚠️ Model warmup failed: {e}")

    threading.Thread(target=warmup).start()

