# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Hebrew RAG system answering halachic questions from "קיצור שולחן ערוך" (Kitzur Shulchan Aruch). Strict closed-book — answers must come **only** from the corpus, never from LLM training knowledge. Built as a course final project.

## Commands

All commands assume the conda env `Arie_RAG` is active (`conda activate Arie_RAG`). Project root contains a Hebrew-named directory with RTL Unicode marks (U+200F) — see "RTL path caveat" below.

```powershell
# One-time: build the NumPy vector store from the JSON corpus (~$0.06)
python -m backend.ingest

# Run the Chainlit dev server (auto-reload on file changes)
python -m chainlit run frontend/app.py -w --host 0.0.0.0 --port 8000

# Smoke tests (top-K retrieval scores for sample queries → smoke_output.txt)
python smoke_test.py

# Full pipeline test (rewrite + retrieve + decide + generate, full JSON answers)
python full_pipeline_test.py

# Demo of the used_sources mechanism end-to-end on one query
python demo_used_sources.py
```

There is no test framework, no linter, no build step. Single-process Python.

## Architecture

The flow for a single user message lives in three files. Read them in this order to understand the system:

1. **`backend/ingest.py`** (run once) — reads `backend/data/kitzur_json.json` (2,758 Seifim). Sub-chunks the 2 outliers >2,000 chars while preserving `full_reference`. Embeds via `text-embedding-3-large` (3,072-dim) in batches of 100, L2-normalizes, writes `backend/vector_store/embeddings.npy` (shape `(N, 3072)`) and a parallel `metadata.json` aligned by row.

2. **`backend/rag_engine.py`** — the `RAGEngine` class. `answer()` orchestrates:
   - `rewrite_query()` — if there's chat history, condense the question to a standalone form via `gpt-4o-mini`. Otherwise pass-through.
   - `embed_query()` → L2-normalized vector.
   - `retrieve()` — `embeddings @ query_vec` (cosine since both normalized), returns top-K with score attached. **K=5**.
   - `decide_path()` — three outcomes:
     - `top_score < THRESHOLD_MIN (0.42)` → `no_info` fallback.
     - Otherwise → **always** call the Router LLM (`gpt-4o-mini`, JSON mode, conservative prompt). Router decides `clarification` vs `answer`.
     - If a clarification was already shown for this question (flag from frontend), skip Router and answer directly.
   - `generate_answer()` — sends top-K chunks to `gpt-4o-mini` in **JSON mode**. Prompt enforces strict closed-book + literal grounding. LLM returns `{"answer": "...", "used_sources": [N]}`. The code then builds the citation block in Python from **only the chunks listed in `used_sources`** — never from retrieval rank.

3. **`frontend/app.py`** — Chainlit. `on_chat_start` instantiates a single `RAGEngine` per session. `on_message` does (in order): admin gateway (`"מנהל"` prefix → placeholder), clarification follow-up handling (one-shot via `clarification_used_for_current_question` flag), then `engine.answer()`.

### Key design choices (do not undo without thinking)

- **Threshold calibrated for Hebrew, not English.** `text-embedding-3-large` produces lower cosines on Hebrew (legit on-topic queries score 0.50–0.66, off-topic 0.34–0.40). `THRESHOLD_MIN=0.42` reflects this. Using English-typical thresholds (0.65+) makes legit questions fall to "no info".
- **Router runs on every query** (not just gray-zone scores). Reason: a semantically ambiguous query like "מתי מותר להדליק נרות?" can score 0.66 (high) because the corpus is dominated by one meaning (Shabbat). Score-based gating misses this; semantic gating catches it.
- **Citation block built in Python, not by LLM.** LLM returns `used_sources` (1-indexed positions in the supplied chunks); code maps to `retrieved[i-1]` and reads `score`/`full_reference`/`content` from the in-memory metadata. The user-visible % is **always** from the NumPy retrieval, never from the LLM.
- **Score is intentionally NOT sent to the LLM** in the prompt — it should pick by content alone, not by retrieval rank. (We've seen the right chunk be #4 by score; LLM picked it correctly because content > score.)
- **Anti-hallucination prompt is strict and Hebrew-specific.** It includes a literal forbidden example ("פוקח עורים" / "מלביש ערומים" — blessing names from the standard prayer book that the LLM "knows" from training but aren't in the cited Seif). When tightening rules, preserve this concrete example.
- **JSON debug block is prepended to every answer for now.** Marked `# DEBUG:` in `generate_answer`. Strip the `debug_block` declaration and the four `debug_block +` returns to remove.

### Data shape

Each entry in `kitzur_json.json` is a flat object: `book`, `siman_id` (Hebrew letter), `siman_title`, `seif_id` (Hebrew letter), `content`, `metadata.full_reference`, `metadata.context_header`. **Use `metadata.full_reference` verbatim** for user-facing citations — it's already pre-formatted ("קיצור שולחן ערוך, סימן א - דיני השכמת הבוקר, סעיף ב").

## RTL path caveat

The project root path is `c:\Users\Arie\OneDrive\AI_Dev6\\u200F\u200FArie_RAG_System3` — the directory name **starts with two U+200F (RIGHT-TO-LEFT MARK) characters**. Consequences:

- The `Write` tool **fails** on this path: it interprets RTL marks as the literal text `\u200F\u200F` and creates a junk folder `AI_Dev6\u200F\u200FArie_RAG_System3\` instead.
- **Workflow that works:** stage files in `C:\temp\rag_staging\` (mirroring the project structure) using the `Write` tool, then `Copy-Item -LiteralPath` via PowerShell to the real path.
- PowerShell handles the path correctly via `Get-ChildItem -LiteralPath` + wildcard match: `Get-ChildItem ... | Where-Object { $_.Name -like "*Arie_RAG_System3*" }`.
- `Bash` rejects the path entirely (cannot resolve the Unicode escapes).
- `New-Item -ItemType Directory` rejects `-LiteralPath`; use `[System.IO.Directory]::CreateDirectory($path)` instead.
- For Python, no issue — `Path(__file__).resolve().parent.parent` resolves correctly because Python receives the path natively from the OS.

## Chainlit hot-reload caveat

`-w` mode reloads modules on save, but **`cl.user_session` retains the old `RAGEngine` instance** — methods bound at session start keep their old code. After meaningful changes to `rag_engine.py`, the user's existing chat keeps using stale code. Two workarounds: (1) start a **new chat** in the UI, or (2) `TaskStop` + relaunch the server (full process restart).

## Threshold and parameter knobs

Located at top of `backend/rag_engine.py`:

- `TOP_K = 5` — chunks sent to the LLM. Was 10; lowered for focused answers + less hallucination opportunity.
- `THRESHOLD_MIN = 0.42` — below this, return "no info". Empirically calibrated.
- `LLM_MODEL = "gpt-4o-mini"` — used for rewrite, Router, and generation.
- `EMBEDDING_MODEL = "text-embedding-3-large"` — fixed; the vector store dim (3,072) depends on it.

If you change `EMBEDDING_MODEL` or chunking logic, you must rerun `python -m backend.ingest` to rebuild `embeddings.npy`.

## Reference docs in repo

- `IMPLEMENTATION_PLAN.md` — original design + decisions (chunking, thresholds, Router, citations, admin roadmap). Some details have evolved since (Router runs always, not gray-zone-only; thresholds re-calibrated; citation block built in Python; JSON-mode answers with `used_sources`). Treat the plan as historical context, the code as truth.
- `README.md` — Hebrew quickstart for setup and run.
- `chainlit.md` — Hebrew welcome screen rendered by Chainlit.
