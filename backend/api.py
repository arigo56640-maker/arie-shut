"""
FastAPI HTTP wrapper around RAGEngine.

Run locally:
    uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload

On Railway it is launched with:
    uvicorn backend.api:app --host 0.0.0.0 --port $PORT

On first deploy the volume at /app/backend/vector_store/ is empty.
A background thread runs ingest automatically so the volume populates
while uvicorn is already serving (health returns "loading" until done).
Subsequent deploys find the files and skip ingest entirely.
"""
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.rag_engine import RAGEngine, EMBEDDINGS_PATH, METADATA_PATH


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
_engine_error: str | None = None
_status: str = "starting"  # "starting" | "loading" | "ok" | "error"


def _bootstrap() -> None:
    """Run in a background thread: ingest if needed, then load engine."""
    global _engine, _engine_error, _status
    _status = "loading"
    try:
        if not EMBEDDINGS_PATH.exists() or not METADATA_PATH.exists():
            print("[api] Vector store missing — running ingest on mounted volume...")
            from backend.ingest import main as ingest_main
            ingest_main()
            print("[api] Ingest complete.")
        _engine = RAGEngine()
        _status = "ok"
        print(f"[api] Engine ready: {len(_engine.metadata)} chunks loaded.")
    except Exception as exc:
        _engine_error = str(exc)
        _status = "error"
        print(f"[api] Engine failed to load: {exc}")


@app.on_event("startup")
def startup_event() -> None:
    threading.Thread(target=_bootstrap, daemon=True).start()


@app.get("/health")
def health() -> dict[str, Any]:
    if _status in ("starting", "loading"):
        return {"status": _status}
    if _status == "error":
        return {"status": "error", "error": _engine_error}
    eng = _engine
    return {
        "status": "ok",
        "embeddings_rows": int(eng.embeddings.shape[0]),
        "embeddings_dim": int(eng.embeddings.shape[1]),
        "metadata_count": len(eng.metadata),
    }


@app.post("/answer")
def answer(req: AnswerRequest) -> dict[str, Any]:
    if _status in ("starting", "loading"):
        raise HTTPException(status_code=503, detail=f"Backend {_status} — try again in a moment.")
    if _status == "error" or _engine is None:
        raise HTTPException(status_code=503, detail=f"Engine error: {_engine_error}")
    try:
        return _engine.answer(
            question=req.question,
            history=req.history,
            clarification_already_used=req.clarification_already_used,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
