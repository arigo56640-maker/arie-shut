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
import os
import threading
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.rag_engine import RAGEngine, EMBEDDINGS_PATH, METADATA_PATH
from backend.whatsapp import GreenAPIClient, SessionStore, handle_incoming


load_dotenv()


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

_wa_client: GreenAPIClient | None = None
_wa_store: SessionStore | None = None


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


def _setup_whatsapp() -> None:
    """Initialize the WhatsApp adapter if Green API credentials are present.
    Stays disabled (and the /webhook/whatsapp endpoint returns 503) when
    GREENAPI_INSTANCE_ID / GREENAPI_API_TOKEN are not configured."""
    global _wa_client, _wa_store
    instance_id = os.getenv("GREENAPI_INSTANCE_ID")
    api_token = os.getenv("GREENAPI_API_TOKEN")
    if instance_id and api_token:
        _wa_client = GreenAPIClient(instance_id, api_token)
        _wa_store = SessionStore()
        print("[api] WhatsApp adapter ready (Green API instance configured).")
    else:
        print("[api] WhatsApp adapter disabled (set GREENAPI_INSTANCE_ID and GREENAPI_API_TOKEN to enable).")


@app.on_event("startup")
def startup_event() -> None:
    threading.Thread(target=_bootstrap, daemon=True).start()
    _setup_whatsapp()


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


@app.post("/webhook/whatsapp")
async def whatsapp_webhook(request: Request, token: str = "") -> dict[str, Any]:
    """Green API delivers inbound WhatsApp messages here.

    The full webhook URL must be:  https://<host>/webhook/whatsapp?token=<GREENAPI_WEBHOOK_TOKEN>
    Always returns HTTP 200 — Green API retries on non-2xx, which would create
    a feedback loop on transient errors. Failures are logged server-side."""
    expected = os.getenv("GREENAPI_WEBHOOK_TOKEN", "")
    if expected and token != expected:
        # Wrong/missing token — refuse but don't 4xx (Green API would retry).
        return {"ok": False, "reason": "invalid_token"}

    if _engine is None or _wa_client is None or _wa_store is None:
        return {"ok": False, "reason": "not_ready", "status": _status}

    try:
        payload = await request.json()
    except Exception as exc:
        print(f"[whatsapp] failed to parse webhook JSON: {exc}")
        return {"ok": False, "reason": "bad_json"}

    try:
        await handle_incoming(payload, _wa_store, _engine, _wa_client)
    except Exception as exc:
        print(f"[whatsapp] handler error: {exc}")
    return {"ok": True}
