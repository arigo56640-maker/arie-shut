"""
FastAPI HTTP wrapper around RAGEngine.

Run locally:
    uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload

On Railway it is launched with:
    uvicorn backend.api:app --host 0.0.0.0 --port $PORT
"""
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.rag_engine import RAGEngine


app = FastAPI(title="Kitzur Shulchan Aruch RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnswerRequest(BaseModel):
    question: str
    history: list[dict[str, Any]] = []
    clarification_already_used: bool = False


_engine: RAGEngine | None = None


def get_engine() -> RAGEngine:
    global _engine
    if _engine is None:
        _engine = RAGEngine()
    return _engine


@app.on_event("startup")
def startup_event() -> None:
    get_engine()


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        eng = get_engine()
        return {
            "status": "ok",
            "embeddings_rows": int(eng.embeddings.shape[0]),
            "embeddings_dim": int(eng.embeddings.shape[1]),
            "metadata_count": len(eng.metadata),
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/answer")
def answer(req: AnswerRequest) -> dict[str, Any]:
    try:
        result = get_engine().answer(
            question=req.question,
            history=req.history,
            clarification_already_used=req.clarification_already_used,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
