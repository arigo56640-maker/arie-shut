"""
WhatsApp adapter for the Hebrew RAG system, via Green API.

Mirrors the *flow* of frontend/app.py (admin gateway → clarification follow-up
→ regular question), but renders for WhatsApp instead of Chainlit. The actual
RAG logic (retrieval, router, generation) is **not** duplicated — every call
goes to the same RAGEngine singleton living in backend/api.py.

Per-chat session state is kept in memory keyed by Green API chatId
(e.g. "972500000000@c.us"). State is lost on process restart — acceptable
for the current scope.

Webhook payload reference (Green API):
    https://green-api.com/en/docs/api/receiving/notifications-format/
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import httpx

from backend.shared import (
    ADMIN_TRIGGER,
    NO_INFO_MESSAGE,
    format_clarification_message,
    parse_clarification_choice,
)


# WhatsApp accepts up to 4096 chars per message. Keep a safety margin so that
# an emoji, RTL mark, or trailing newline never tips us over.
WA_MAX_LEN = 3500


def format_for_whatsapp(text: str) -> str:
    """Convert standard Markdown bold (`**x**`) to WhatsApp bold (`*x*`).

    WhatsApp's lightweight markup uses single asterisks for bold; double
    asterisks render literally. Italics (`_x_`) and code fences are already
    compatible.
    """
    return re.sub(r"\*\*([^*\n]+)\*\*", r"*\1*", text)


def _split_for_whatsapp(text: str, limit: int = WA_MAX_LEN) -> list[str]:
    """Split a long message into chunks under WhatsApp's limit, on paragraph
    boundaries when possible, otherwise on line boundaries."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        # Prefer to break on a blank line, then a single newline, then a hard cut.
        cut = remaining.rfind("\n\n", 0, limit)
        if cut == -1:
            cut = remaining.rfind("\n", 0, limit)
        if cut == -1:
            cut = limit
        chunks.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


# ---------------------------------------------------------------------------
# WhatsApp-flavoured admin views (Chainlit uses its own HTML-rich versions).
# ---------------------------------------------------------------------------

ADMIN_MENU_WA = (
    "🛠️ *מצב מנהל — אפשרויות:*\n\n"
    "1. הצג נתונים על השאלה האחרונה (prompt + JSON גולמי)\n"
    "2. בדיקת תקינות backend (health)\n"
    "3. הצגת סטטיסטיקות שיחה — _בקרוב_\n"
    "4. ייצוא היסטוריית שיחה — _בקרוב_\n\n"
    "להפעלה: שלח `מנהל 1` עד `מנהל 4`."
)


def format_last_debug_wa(debug: dict) -> str:
    return (
        "🛠️ *נתוני השאלה האחרונה*\n\n"
        f"*שאלה (לאחר שכתוב):* {debug.get('question', '—')}\n\n"
        "*📤 הפרומפט שנשלח ל-LLM:*\n"
        f"```\n{debug.get('last_prompt', '')}\n```\n\n"
        "*📥 JSON שהתקבל מה-LLM:*\n"
        f"```\n{debug.get('last_raw_json', '')}\n```"
    )


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

@dataclass
class WhatsAppSession:
    """Per-chat state, mirroring the six fields cl.user_session keeps in Chainlit
    plus last_debug for the `מנהל 1` view."""
    history: list[dict] = field(default_factory=list)
    awaiting_clarification: bool = False
    pending_clarification_options: list[str] | None = None
    pending_original_question: str | None = None
    clarification_used_for_current_question: bool = False
    awaiting_admin_choice: bool = False
    last_debug: dict | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, WhatsAppSession] = {}
        self._global_lock = asyncio.Lock()

    async def get(self, chat_id: str) -> WhatsAppSession:
        async with self._global_lock:
            session = self._sessions.get(chat_id)
            if session is None:
                session = WhatsAppSession()
                self._sessions[chat_id] = session
            return session


# ---------------------------------------------------------------------------
# Green API client (thin)
# ---------------------------------------------------------------------------

class GreenAPIClient:
    def __init__(
        self,
        instance_id: str,
        api_token: str,
        base_url: str = "https://api.green-api.com",
    ) -> None:
        self.instance_id = instance_id
        self.api_token = api_token
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=30.0)

    async def send_message(self, chat_id: str, message: str) -> None:
        url = (
            f"{self.base_url}/waInstance{self.instance_id}"
            f"/sendMessage/{self.api_token}"
        )
        try:
            response = await self._client.post(
                url, json={"chatId": chat_id, "message": message}
            )
            response.raise_for_status()
        except Exception as exc:
            print(f"[whatsapp] send_message failed for {chat_id}: {exc}")

    async def aclose(self) -> None:
        await self._client.aclose()


# ---------------------------------------------------------------------------
# Webhook entry point
# ---------------------------------------------------------------------------

def _extract_message(payload: dict) -> tuple[str | None, str | None]:
    """Return (chat_id, text) for an inbound text message, or (None, None)
    for any other notification type (status updates, outgoing acks, media)."""
    if payload.get("typeWebhook") != "incomingMessageReceived":
        return None, None
    chat_id = (payload.get("senderData") or {}).get("chatId")
    msg_data = payload.get("messageData") or {}
    text = (msg_data.get("textMessageData") or {}).get("textMessage")
    if not text:
        text = (msg_data.get("extendedTextMessageData") or {}).get("text")
    if not chat_id or not text:
        return None, None
    return chat_id, text


async def handle_incoming(
    payload: dict,
    store: SessionStore,
    engine: Any,
    client: GreenAPIClient,
) -> None:
    chat_id, text = _extract_message(payload)
    if not chat_id or not text:
        return
    session = await store.get(chat_id)
    async with session.lock:
        await _handle_message(chat_id, text.strip(), session, engine, client)


# ---------------------------------------------------------------------------
# Send helper (always converts markdown + chunks long messages)
# ---------------------------------------------------------------------------

async def _send(client: GreenAPIClient, chat_id: str, text: str) -> None:
    converted = format_for_whatsapp(text)
    for chunk in _split_for_whatsapp(converted):
        await client.send_message(chat_id, chunk)


# ---------------------------------------------------------------------------
# Core flow — mirrors frontend/app.py:on_message exactly
# ---------------------------------------------------------------------------

async def _handle_message(
    chat_id: str,
    text: str,
    session: WhatsAppSession,
    engine: Any,
    client: GreenAPIClient,
) -> None:
    # 1. Admin gateway
    if text.startswith(ADMIN_TRIGGER):
        command = text[len(ADMIN_TRIGGER):].strip()
        session.awaiting_admin_choice = not command
        await _handle_admin(command, session, client, chat_id)
        return

    if session.awaiting_admin_choice:
        session.awaiting_admin_choice = False
        await _handle_admin(text, session, client, chat_id)
        return

    # 2. Clarification follow-up
    if session.awaiting_clarification:
        options = session.pending_clarification_options or []
        original_question = session.pending_original_question or ""
        choice = parse_clarification_choice(text, options)

        session.awaiting_clarification = False
        session.pending_clarification_options = None
        session.pending_original_question = None
        session.clarification_used_for_current_question = True

        merged = (
            f"{original_question} (הבהרה: {choice})" if choice else original_question
        )
        await _answer_and_send(
            merged, session, client, chat_id, engine,
            history_user_text=original_question,
        )
        return

    # 3. Regular question
    session.clarification_used_for_current_question = False
    await _answer_and_send(
        text, session, client, chat_id, engine,
        history_user_text=text,
    )


async def _answer_and_send(
    question: str,
    session: WhatsAppSession,
    client: GreenAPIClient,
    chat_id: str,
    engine: Any,
    history_user_text: str,
) -> None:
    clarification_already_used = session.clarification_used_for_current_question

    try:
        # engine.answer is a sync method (OpenAI calls + numpy). Run it in a
        # worker thread so we don't block the asyncio event loop while the
        # webhook coroutine is suspended.
        result = await asyncio.to_thread(
            engine.answer,
            question,
            list(session.history),
            clarification_already_used,
        )
    except Exception as exc:
        print(f"[whatsapp] engine.answer failed for {chat_id}: {exc}")
        await _send(client, chat_id, f"❌ שגיאה פנימית: {exc}")
        return

    rtype = result["type"]
    rewritten = result.get("rewritten", question)
    if rewritten and rewritten != question:
        print(f"[whatsapp][{chat_id}] rewritten: {rewritten}")

    if rtype == "no_info":
        await _send(client, chat_id, NO_INFO_MESSAGE)
        session.history.append({"role": "user", "content": history_user_text})
        session.history.append({"role": "assistant", "content": NO_INFO_MESSAGE})
        return

    if rtype == "clarification":
        options = result["options"]
        msg = format_clarification_message(options)
        session.awaiting_clarification = True
        session.pending_clarification_options = options
        session.pending_original_question = history_user_text
        await _send(client, chat_id, msg)
        return

    # type == "answer"
    text_out = result["text"]
    if "last_prompt" in result and "last_raw_json" in result:
        session.last_debug = {
            "question": result.get("rewritten") or question,
            "last_prompt": result["last_prompt"],
            "last_raw_json": result["last_raw_json"],
        }
    await _send(client, chat_id, text_out)
    session.history.append({"role": "user", "content": history_user_text})
    session.history.append({"role": "assistant", "content": text_out})


# ---------------------------------------------------------------------------
# Admin handler
# ---------------------------------------------------------------------------

async def _handle_admin(
    command: str,
    session: WhatsAppSession,
    client: GreenAPIClient,
    chat_id: str,
) -> None:
    cmd = command.strip()

    if not cmd:
        await _send(client, chat_id, ADMIN_MENU_WA)
        return

    if cmd in ("1", "debug"):
        if not session.last_debug:
            msg = "🛠️ לא נשאלה עדיין שאלה בשיחה הזו — אין נתונים להציג."
        else:
            msg = format_last_debug_wa(session.last_debug)
        await _send(client, chat_id, msg)
        return

    if cmd in ("2", "health"):
        await _send(
            client, chat_id,
            "🛠️ Backend health: ✅ engine in-process (WhatsApp adapter)",
        )
        return

    if cmd in ("3", "4"):
        await _send(client, chat_id, "🛠️ אפשרות זו עדיין לא מומשה (בקרוב).")
        return

    await _send(
        client, chat_id,
        f"🛠️ פקודה לא מוכרת: `{cmd}`\n\n{ADMIN_MENU_WA}",
    )
