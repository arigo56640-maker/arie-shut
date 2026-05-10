# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Hebrew RAG system answering halachic questions from "קיצור שולחן ערוך" (Kitzur Shulchan Aruch). Strict closed-book — answers must come **only** from the corpus, never from LLM training knowledge. Built as a course final project; currently deployed to Railway and accessible via both a Chainlit web UI and a WhatsApp number (Green API gateway).

## Commands

All commands assume the conda env `Arie_RAG` is active (`conda activate Arie_RAG`). The project root contains a Hebrew-named directory with two leading U+200F (RTL marks) — see "RTL path caveat" below.

Local dev runs as **two services**: FastAPI (the engine + WhatsApp webhook) on one port, Chainlit (browser UI) on another. Chainlit talks to FastAPI over HTTP — it does **not** import `RAGEngine` directly anymore.

```powershell
# One-time: build the NumPy vector store from the JSON corpus (~$0.06)
python -m backend.ingest

# Terminal 1 — backend (FastAPI, RAGEngine, /webhook/whatsapp)
uvicorn backend.api:app --host 0.0.0.0 --port 8000

# Terminal 2 — frontend (Chainlit). BACKEND_URL must point at the FastAPI port.
chainlit run frontend/app.py -w --port 8001

# Smoke tests (top-K retrieval scores for sample queries → smoke_output.txt)
python smoke_test.py

# Full pipeline test (rewrite + retrieve + decide + generate, full JSON answers)
python full_pipeline_test.py

# Demo of the used_sources mechanism end-to-end on one query
python demo_used_sources.py
```

There is no test framework, no linter, no build step. Single-process Python on each service.

## Architecture

The flow for one user message lives across these files. Read them in this order:

1. **`backend/ingest.py`** (run once) — reads `backend/data/kitzur_json.json`. Sub-chunks the few outliers >2,000 chars while preserving `full_reference`. Embeds via `text-embedding-3-large` (3,072-dim) in batches of 100, L2-normalizes, writes `backend/vector_store/embeddings.npy` (shape `(2780, 3072)`) and a parallel `metadata.json` aligned by row.

2. **`backend/rag_engine.py`** — the `RAGEngine` class. `answer(question, history, clarification_already_used)` orchestrates:
   - `rewrite_query()` — gated by `FOLLOWUP_ENABLED` (currently `False` → returns the question unchanged, every query is treated as standalone).
   - `embed_query()` → L2-normalized vector.
   - `retrieve()` — `embeddings @ query_vec` (cosine since both normalized), returns top-K with score attached. **K=5**.
   - `decide_path()` — three outcomes:
     - `top_score < THRESHOLD_MIN (0.42)` → `no_info` fallback.
     - Otherwise → if `CLARIFICATION_ENABLED` (currently `False`) skipped; else Router LLM decides `clarification` vs `answer`.
     - If a clarification was already shown for this question, skip Router and answer directly.
   - `generate_answer()` — sends top-K chunks to `gpt-4o-mini` in **JSON mode**. Prompt enforces strict closed-book + literal grounding. LLM returns `{"answer": "...", "used_sources": [N]}`. The code then builds the citation block in Python from **only the chunks listed in `used_sources`** — never from retrieval rank.
   - Return value: `dict` with `type: "no_info" | "clarification" | "answer"`. For `answer` type also includes `text`, `chunks`, `last_prompt`, `last_raw_json` (the last two are debug-only, surfaced by the frontend in admin mode).

3. **`backend/api.py`** — FastAPI wrapper. Two endpoints:
   - `POST /answer` — what the Chainlit frontend hits. Pydantic body matches `engine.answer()`'s args.
   - `POST /webhook/whatsapp` — Green API webhook. Returns 200 always (Green API retries on non-2xx, which would create a feedback loop). Optional `?token=...` query param checked against `GREENAPI_WEBHOOK_TOKEN` env var. WhatsApp adapter only initializes when `GREENAPI_INSTANCE_ID` + `GREENAPI_API_TOKEN` are set, otherwise the endpoint returns `{"ok": false, "reason": "not_ready"}` — meaning a deploy without Green API credentials still works fine for the browser path.
   - On startup, `_bootstrap()` runs `ingest()` if the volume is empty (handles first Railway deploy), then loads the engine. Engine loads in a background thread so the HTTP server is available immediately and `/health` reports `loading` until ready.

4. **`backend/whatsapp.py`** — Green API adapter. Mirrors the **flow** of `frontend/app.py:on_message` (admin gateway → clarification one-shot → regular question) without duplicating any RAG logic. Per-chat sessions kept in memory keyed by Green API `chatId` (e.g. `972500000000@c.us`), with an `asyncio.Lock` per chat to avoid race conditions when a user sends two messages quickly. Markdown converted from `**bold**` (Chainlit-style) to `*bold*` (WhatsApp-style); long messages chunked under 3500 chars.

5. **`backend/shared.py`** — constants and pure helpers used by both channels: `ADMIN_TRIGGER`, `NO_INFO_MESSAGE`, `parse_clarification_choice`, `format_clarification_message`. Single source of truth — anything channel-agnostic lives here so changes propagate to both interfaces.

6. **`frontend/app.py`** — Chainlit. `on_chat_start` initializes per-session state (history + clarification flags + admin flag). `on_message` does (in order): admin gateway (`"מנהל"` prefix → menu, then waits for the next message as the choice), clarification follow-up (one-shot via `clarification_used_for_current_question` flag), regular question. Calls `call_backend_answer()` which is a thin `httpx.AsyncClient.post(f"{BACKEND_URL}/answer", ...)`. **Does not import `RAGEngine` directly** — this is what enables both Chainlit and WhatsApp to share one engine without process coupling.

### Key design choices (do not undo without thinking)

- **Threshold calibrated for Hebrew, not English.** `text-embedding-3-large` produces lower cosines on Hebrew (legit on-topic queries score 0.50–0.66, off-topic 0.34–0.40). `THRESHOLD_MIN=0.42` reflects this. Using English-typical thresholds (0.65+) makes legit questions fall to "no info".
- **Citation block built in Python, not by LLM.** LLM returns `used_sources` (1-indexed positions in the supplied chunks); code maps to `retrieved[i-1]` and reads `score`/`full_reference`/`content` from the in-memory metadata. The user-visible % is **always** from the NumPy retrieval, never from the LLM.
- **Score is intentionally NOT sent to the LLM** in the prompt — it should pick by content alone, not by retrieval rank. (We've seen the right chunk be #4 by score; LLM picked it correctly because content > score.)
- **Anti-hallucination prompt is strict and Hebrew-specific.** It includes a literal forbidden example ("פוקח עורים" / "מלביש ערומים" — blessing names from the standard prayer book that the LLM "knows" from training but aren't in the cited Seif). When tightening rules, preserve this concrete example.
- **Debug data is plumbed but gated.** `engine.answer()` always returns `last_prompt` and `last_raw_json` for `answer`-type responses. The frontend caches them in the session and only renders them when the user types `מנהל 1`. Don't re-prepend to answers; don't delete the plumbing.
- **One source of truth for cross-channel logic.** Anything a Chainlit user and a WhatsApp user should both see (no_info text, clarification message format, parse logic, admin trigger word) lives in `backend/shared.py` and is imported by both `frontend/app.py` and `backend/whatsapp.py`. Don't duplicate.
- **Clarification flow is gated.** `CLARIFICATION_ENABLED = False` currently → Router is skipped, every question that scores above threshold is answered directly. Re-enable by flipping the flag; the Router prompt is conservative (only true ambiguity like "נרות" → Shabbat/Hanukkah/Yom Tov triggers it).

### Data shape

Each entry in `kitzur_json.json` is a flat object: `book`, `siman_id` (Hebrew letter), `siman_title`, `seif_id` (Hebrew letter), `content`, `metadata.full_reference`, `metadata.context_header`. **Use `metadata.full_reference` verbatim** for user-facing citations — it's already pre-formatted ("קיצור שולחן ערוך, סימן א - דיני השכמת הבוקר, סעיף ב").

## Deployment

Deployed to Railway as **two services** in project `arie-shut`, both auto-deploying from `main`:

- `backend` — `uvicorn backend.api:app`, public at `backend-production-cb89.up.railway.app`. Required env vars: `OPENAI_API_KEY`, `GREENAPI_INSTANCE_ID`, `GREENAPI_API_TOKEN`. Has a Railway volume mounted at `/app/backend/vector_store` so the embeddings survive redeploys.
- `frontend` — `chainlit run frontend/app.py --host 0.0.0.0 --port $PORT --headless`, public at `frontend-production-b648.up.railway.app`. Required env vars: `BACKEND_URL=http://backend.railway.internal:8080`, **`PYTHONPATH=/app`** (without this, Chainlit's loader can't find `backend.shared` — the shim doesn't add cwd to `sys.path`).

Green API webhook URL points at the Railway backend: `https://backend-production-cb89.up.railway.app/webhook/whatsapp`. To switch back to local dev, expose the local FastAPI via cloudflared/ngrok and update `webhookUrl` via Green API's `setSettings` endpoint.

### Local WhatsApp dev (cloudflared)

`cloudflared tunnel --url http://localhost:8000` gives a temporary `*.trycloudflare.com` URL. Use the helper script pattern from prior sessions: `httpx.post` to `https://api.green-api.com/waInstance{id}/setSettings/{token}` with `{"webhookUrl": ..., "incomingWebhook": "yes"}`. Note: Green API instances reboot after `setSettings` and may take ~30s to a few minutes to actually apply the new URL. Their service is flaky — `getStateInstance` returning `notAuthorized` is sometimes transient, sometimes a real unlink that needs a QR rescan.

## RTL path caveat

The project root path is `c:\Users\Arie\OneDrive\AI_Dev6\\u200F‏Arie_RAG_System3` — the directory name **starts with two U+200F (RIGHT-TO-LEFT MARK) characters**. Consequences:

- The `Write` tool **fails when creating** at this path: it interprets RTL marks as the literal text `‏‏` and creates a junk folder `AI_Dev6‏‏Arie_RAG_System3\` instead. **Editing existing files works** — the editor harness already has the file path resolved.
- **Workflow that works:** for new files, stage in `C:\temp\rag_staging\` (mirroring the project structure) using `Write`, then `Copy-Item -LiteralPath` via PowerShell to the real path. For edits to existing files, use `Edit` directly.
- PowerShell handles the path correctly via `Get-ChildItem -LiteralPath` + wildcard match: `Get-ChildItem ... | Where-Object { $_.Name -like "*Arie_RAG_System3*" -and $_.Name -notlike "*גיבוי*" }` (the `-notlike "גיבוי"` filter excludes the user's backup directory which also matches the wildcard).
- `Bash` rejects the path entirely (cannot resolve the Unicode escapes). Use PowerShell + Python for anything that touches the project root.
- `New-Item -ItemType Directory` rejects `-LiteralPath`; use `[System.IO.Directory]::CreateDirectory($path)` instead.
- For Python, no issue — `Path(__file__).resolve().parent.parent` resolves correctly because Python receives the path natively from the OS.

## Chainlit hot-reload caveat

`-w` mode reloads modules on save, but **`cl.user_session` retains state** that may reference the old session bindings. After meaningful changes to anything imported at session start, the user's existing chat may keep using stale wiring. Two workarounds: (1) start a **new chat** in the UI, or (2) `TaskStop` + relaunch the server (full process restart).

This matters less now that Chainlit only holds session state, not the engine itself — the engine lives in the FastAPI process and is shared. Engine changes take effect after restarting `uvicorn`, not Chainlit.

## Threshold and parameter knobs

Located at the top of `backend/rag_engine.py`:

- `TOP_K = 5` — chunks sent to the LLM. Was 10; lowered for focused answers + less hallucination opportunity.
- `THRESHOLD_MIN = 0.42` — below this, return "no info". Empirically calibrated for Hebrew embeddings.
- `CLARIFICATION_ENABLED = False` — Router is currently off; flip to re-enable.
- `FOLLOWUP_ENABLED = False` — query rewriting from history currently off; flip to re-enable.
- `SHOW_MATCH_PERCENT = False` — citations don't show "התאמה: X%"; flip to re-enable.
- `LLM_MODEL = "gpt-4o-mini"` — used for rewrite, Router, and generation.
- `EMBEDDING_MODEL = "text-embedding-3-large"` — fixed; the vector store dim (3,072) depends on it.

If you change `EMBEDDING_MODEL` or chunking logic, you must rerun `python -m backend.ingest` to rebuild `embeddings.npy`. On Railway the volume is sticky, so a one-time delete or a manual ingest run is needed to force a rebuild.

## Reference docs in repo

- `README.md` — Hebrew, user-facing. Mermaid architecture diagram, local dev quickstart, Railway deployment notes, WhatsApp setup for both local and production.
- `IMPLEMENTATION_PLAN.md` — original design + decisions (chunking, thresholds, Router, citations, admin roadmap). Some details have evolved since (Router currently disabled; thresholds re-calibrated; citation block built in Python; JSON-mode answers with `used_sources`; Chainlit/FastAPI split). Treat the plan as historical context, the code as truth.
- `chainlit.md` — Hebrew welcome screen rendered by Chainlit.
