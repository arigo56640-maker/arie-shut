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
_engine_load_error: str | None = None


def get_engine(strict: bool = True) -> RAGEngine | None:
    """Lazy-load the RAG engine. Returns None on failure when strict=False."""
    global _engine, _engine_load_error
    if _engine is not None:
        return _engine
    try:
        _engine = RAGEngine()
        _engine_load_error = None
        return _engine
    except Exception as e:
        _engine_load_error = str(e)
        if strict:
            raise
        return None


@app.get("/health")
def health() -> dict[str, Any]:
    """Always returns 200 so the container stays up while the volume is being populated."""
    eng = get_engine(strict=False)
    if eng is None:
        return {"status": "not_ready", "error": _engine_load_error}
    return {
        "status": "ok",
        "embeddings_rows": int(eng.embeddings.shape[0]),
        "embeddings_dim": int(eng.embeddings.shape[1]),
        "metadata_count": len(eng.metadata),
    }


@app.post("/answer")
def answer(req: AnswerRequest) -> dict[str, Any]:
    eng = get_engine(strict=False)
    if eng is None:
        raise HTTPException(
            status_code=503,
            detail=f"RAG engine not ready: {_engine_load_error}",
        )
    try:
        return eng.answer(
            question=req.question,
            history=req.history,
            clarification_already_used=req.clarification_already_used,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
