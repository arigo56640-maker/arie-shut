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


@app.get("/debug-embed")
def debug_embed(q: str = "שבת") -> dict[str, Any]:
    import numpy as npy
    from openai import OpenAI
    client = OpenAI()
    resp = client.embeddings.create(model="text-embedding-3-large", input=[q])
    qv = npy.array(resp.data[0].embedding, dtype=npy.float32)
    qv_norm = float(npy.linalg.norm(qv))
    qv_n = qv / qv_norm if qv_norm > 0 else qv
    result: dict[str, Any] = {"query": q, "query_norm": qv_norm, "query_first3": qv[:3].tolist()}
    if _engine is not None:
        emb = _engine.embeddings
        norms = npy.linalg.norm(emb, axis=1)
        scores = emb @ qv_n
        idx = int(npy.argmax(scores))
        # Also use engine's own embed_query for comparison
        engine_qv = _engine.embed_query(q)
        engine_scores = emb @ engine_qv
        engine_idx = int(npy.argmax(engine_scores))
        result.update({
            "store_shape": list(emb.shape),
            "store_dtype": str(emb.dtype),
            "store_norm_mean": float(norms.mean()),
            "store_first3": emb[0][:3].tolist(),
            "top_score_direct": float(scores[idx]),
            "top_score_engine_qv": float(engine_scores[engine_idx]),
            "engine_qv_first3": engine_qv[:3].tolist(),
            "engine_qv_norm": float(npy.linalg.norm(engine_qv)),
            "top_ref": _engine.metadata[idx].get("full_reference", "?"),
        })
    return result


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


@app.get("/echo")
def echo(q: str = "test") -> dict[str, Any]:
    """Debug: verify that the server receives Hebrew text correctly."""
    return {"received": q, "repr": repr(q), "len": len(q)}


@app.post("/answer")
def answer(req: AnswerRequest) -> dict[str, Any]:
    if _status in ("starting", "loading"):
        raise HTTPException(status_code=503, detail=f"Backend {_status} — try again in a moment.")
    if _status == "error" or _engine is None:
        raise HTTPException(status_code=503, detail=f"Engine error: {_engine_error}")
    print(f"[api] /answer question={req.question!r}")
    try:
        result = _engine.answer(
            question=req.question,
            history=req.history,
            clarification_already_used=req.clarification_already_used,
        )
        top_score = result.get("retrieved", [{}])[0].get("score", 0) if result.get("retrieved") else 0
        print(f"[api] /answer type={result.get('type')} top_score={top_score:.4f} rewritten={result.get('rewritten')!r}")
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
