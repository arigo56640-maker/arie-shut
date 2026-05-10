"""
Chainlit app for the Kitzur Shulchan Aruch RAG system.

This is the *frontend* service. It calls the backend FastAPI service
(at $BACKEND_URL) over HTTP — it does not import the RAG engine directly.

Run locally:
    BACKEND_URL=http://localhost:8000 chainlit run frontend/app.py -w
"""
import os

import chainlit as cl
import httpx
from dotenv import load_dotenv

from backend.shared import (
    ADMIN_TRIGGER,
    NO_INFO_MESSAGE,
    format_clarification_message,
    parse_clarification_choice,
)


load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
HTTP_TIMEOUT = float(os.getenv("BACKEND_TIMEOUT", "60"))


async def call_backend_answer(
    question: str,
    history: list[dict],
    clarification_already_used: bool,
) -> dict:
    payload = {
        "question": question,
        "history": history,
        "clarification_already_used": clarification_already_used,
    }
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        response = await client.post(f"{BACKEND_URL}/answer", json=payload)
        response.raise_for_status()
        return response.json()


async def backend_health_ok() -> tuple[bool, str]:
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"{BACKEND_URL}/health")
            response.raise_for_status()
            data = response.json()
            status = data.get("status")
            if status == "ok":
                return True, f"backend ok ({data.get('metadata_count')} chunks)"
            # "loading" or "starting" — backend is warming up
            return False, f"backend {status} (vector store initializing, ~3 min on first deploy)"
    except Exception as e:
        return False, str(e)


ADMIN_MENU = (
    "🛠️ **מצב מנהל — אפשרויות:**\n\n"
    "1. הצג נתונים על השאלה האחרונה (prompt + JSON גולמי)\n"
    "2. בדיקת תקינות backend (health)\n"
    "3. הצגת סטטיסטיקות שיחה — *בקרוב*\n"
    "4. ייצוא היסטוריית שיחה — *בקרוב*\n\n"
    "להפעלה: הקלד `מנהל 1` עד `מנהל 4`."
)


def _format_last_debug(debug: dict) -> str:
    return (
        f"🛠️ **נתוני השאלה האחרונה**\n\n"
        f"**שאלה (לאחר שכתוב):** {debug.get('question', '—')}\n\n"
        "<details>\n"
        "<summary><b>📤 הפרומפט שנשלח ל-LLM (לחץ לפתיחה)</b></summary>\n\n"
        f"{debug.get('last_prompt', '')}\n\n"
        "</details>\n\n"
        "**📥 JSON שהתקבל מה-LLM:**\n"
        f"```json\n{debug.get('last_raw_json', '')}\n```"
    )


async def handle_admin(command: str) -> None:
    cmd = command.strip()

    if not cmd:
        await cl.Message(content=ADMIN_MENU).send()
        return

    if cmd in ("1", "debug"):
        debug = cl.user_session.get("last_debug")
        if not debug:
            msg = "🛠️ לא נשאלה עדיין שאלה בשיחה הזו — אין נתונים להציג."
        else:
            msg = _format_last_debug(debug)
        await cl.Message(content=msg).send()
        return

    if cmd in ("2", "health"):
        ok, info = await backend_health_ok()
        msg = f"🛠️ Backend health: {'✅' if ok else '❌'} {info}\nBACKEND_URL = `{BACKEND_URL}`"
        await cl.Message(content=msg).send()
        return

    if cmd in ("3", "4"):
        await cl.Message(content="🛠️ אפשרות זו עדיין לא מומשה (בקרוב).").send()
        return

    await cl.Message(
        content=f"🛠️ פקודה לא מוכרת: `{cmd}`\n\n{ADMIN_MENU}"
    ).send()


@cl.on_chat_start
async def on_chat_start():
    ok, info = await backend_health_ok()
    if not ok:
        loading_msg = "loading" in info or "starting" in info
        await cl.Message(
            content=(
                f"⏳ **Backend מתחיל...** ({info})\n\n"
                "המערכת טוענת את מאגר ההטמעות לראשונה. תהליך זה אורך כ-3 דקות. "
                "רענן את הדף בעוד מספר דקות."
                if loading_msg else
                f"❌ לא ניתן להתחבר ל-backend בכתובת `{BACKEND_URL}`.\n\n"
                f"```\n{info}\n```\n\n"
                "ודא ש-service ה-backend רץ ושמשתנה הסביבה `BACKEND_URL` מוגדר נכון."
            )
        ).send()
        return

    cl.user_session.set("history", [])
    cl.user_session.set("awaiting_clarification", False)
    cl.user_session.set("pending_clarification_options", None)
    cl.user_session.set("pending_original_question", None)
    cl.user_session.set("clarification_used_for_current_question", False)
    cl.user_session.set("awaiting_admin_choice", False)

    await cl.Message(
        content=(
            "📖 **ברוכים הבאים למערכת בינה מלאכותית - שאלות ותשובות בהלכה (שו\"ת AI)**\n\n"
            "> 🎓 *פרוייקט סיום קורס בניית מערכות מבוססות AI מגיש* — 👤 **אריה גולדמן**\n\n"
            "שאלו שאלה הלכתית בעברית ואני אתן תשובה המבוססת אך ורק על הספר "
            "קיצור שולחן ערוך, עם הפניה לסימן ולסעיף הרלוונטי בספר ממנו שלפתי את התשובה."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    text = message.content.strip()

    if text.startswith(ADMIN_TRIGGER):
        command = text[len(ADMIN_TRIGGER):].strip()
        # Bare "מנהל" opens the menu and waits for the next message as the choice.
        cl.user_session.set("awaiting_admin_choice", not command)
        await handle_admin(command)
        return

    if cl.user_session.get("awaiting_admin_choice"):
        cl.user_session.set("awaiting_admin_choice", False)
        await handle_admin(text)
        return

    history: list[dict] = cl.user_session.get("history") or []

    if cl.user_session.get("awaiting_clarification"):
        options = cl.user_session.get("pending_clarification_options") or []
        original_question = cl.user_session.get("pending_original_question") or ""
        choice = parse_clarification_choice(text, options)

        cl.user_session.set("awaiting_clarification", False)
        cl.user_session.set("pending_clarification_options", None)
        cl.user_session.set("pending_original_question", None)
        cl.user_session.set("clarification_used_for_current_question", True)

        merged = (
            f"{original_question} (הבהרה: {choice})" if choice else original_question
        )
        await _answer_and_send(
            question=merged,
            history=history,
            history_user_text=original_question,
        )
        return

    cl.user_session.set("clarification_used_for_current_question", False)
    await _answer_and_send(
        question=text,
        history=history,
        history_user_text=text,
    )


async def _answer_and_send(
    question: str,
    history: list[dict],
    history_user_text: str,
) -> None:
    clarification_already_used = bool(
        cl.user_session.get("clarification_used_for_current_question")
    )

    try:
        result = await call_backend_answer(
            question=question,
            history=history,
            clarification_already_used=clarification_already_used,
        )
    except httpx.HTTPStatusError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", "")
        except Exception:
            detail = e.response.text
        await cl.Message(
            content=f"❌ שגיאה מ-backend ({e.response.status_code}): {detail}"
        ).send()
        return
    except Exception as e:
        await cl.Message(content=f"❌ שגיאת תקשורת מול ה-backend: {e}").send()
        return

    rtype = result["type"]
    rewritten = result.get("rewritten", question)
    if rewritten and rewritten != question:
        print(f"[rewritten query] {rewritten}")

    if rtype == "no_info":
        msg = NO_INFO_MESSAGE
        await cl.Message(content=msg).send()
        history.append({"role": "user", "content": history_user_text})
        history.append({"role": "assistant", "content": msg})
        cl.user_session.set("history", history)
        return

    if rtype == "clarification":
        options = result["options"]
        msg = format_clarification_message(options)
        cl.user_session.set("awaiting_clarification", True)
        cl.user_session.set("pending_clarification_options", options)
        cl.user_session.set("pending_original_question", history_user_text)
        await cl.Message(content=msg).send()
        return

    text_out = result["text"]
    if "last_prompt" in result and "last_raw_json" in result:
        cl.user_session.set(
            "last_debug",
            {
                "question": result.get("rewritten") or question,
                "last_prompt": result["last_prompt"],
                "last_raw_json": result["last_raw_json"],
            },
        )
    await cl.Message(content=text_out).send()
    history.append({"role": "user", "content": history_user_text})
    history.append({"role": "assistant", "content": text_out})
    cl.user_session.set("history", history)
