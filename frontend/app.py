"""
Chainlit app for the Kitzur Shulchan Aruch RAG system.

Run: chainlit run frontend/app.py -w
"""
import sys
from pathlib import Path

import chainlit as cl

# Make backend importable when running via `chainlit run frontend/app.py`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.rag_engine import RAGEngine  # noqa: E402

ADMIN_TRIGGER = "מנהל"
CLARIFICATION_LETTERS = ["א", "ב", "ג", "ד"]


def parse_clarification_choice(text: str, options: list[str]) -> str | None:
    text = text.strip()
    if not text:
        return None
    first_char = text[0]
    if first_char in CLARIFICATION_LETTERS:
        idx = CLARIFICATION_LETTERS.index(first_char)
        if idx < len(options):
            return options[idx]
    return text


async def handle_admin(command: str) -> None:
    if not command:
        msg = (
            "🛠️ **מצב מנהל**\n\n"
            "פקודות מתוכננות לעתיד: `verbose`, `timing`, `cost`, `top_k=N`, "
            "`threshold=X`, `search`, `show`, `stats`, `history`, `export`, `health`.\n\n"
            "כרגע אף פקודה לא מומשה."
        )
    else:
        msg = (
            f"🛠️ מצב מנהל זיהה את הפקודה: `{command}`\n\n"
            "בשלב הראשון אין מימוש פעיל. ראה `IMPLEMENTATION_PLAN.md` לרעיונות עתידיים."
        )
    await cl.Message(content=msg).send()


@cl.on_chat_start
async def on_chat_start():
    try:
        engine = RAGEngine()
    except Exception as e:
        await cl.Message(
            content=(
                f"❌ שגיאה בטעינת המערכת:\n```\n{e}\n```\n\n"
                "ודא שהרצת `python -m backend.ingest` ושיש קובץ `.env` עם `OPENAI_API_KEY`."
            )
        ).send()
        return

    cl.user_session.set("engine", engine)
    cl.user_session.set("history", [])
    cl.user_session.set("awaiting_clarification", False)
    cl.user_session.set("pending_clarification_options", None)
    cl.user_session.set("pending_original_question", None)
    cl.user_session.set("clarification_used_for_current_question", False)

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
    engine: RAGEngine | None = cl.user_session.get("engine")
    if engine is None:
        await cl.Message(content="❌ המערכת לא נטענה. רענן את הדף.").send()
        return

    # Step 0 - Admin gateway
    if text.startswith(ADMIN_TRIGGER):
        command = text[len(ADMIN_TRIGGER):].strip()
        await handle_admin(command)
        return

    history: list[dict] = cl.user_session.get("history") or []

    # Was the previous turn a clarification request?
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
            engine,
            question=merged,
            history=history,
            history_user_text=original_question,
        )
        return

    # Fresh question
    cl.user_session.set("clarification_used_for_current_question", False)
    await _answer_and_send(
        engine,
        question=text,
        history=history,
        history_user_text=text,
    )


async def _answer_and_send(
    engine: RAGEngine,
    question: str,
    history: list[dict],
    history_user_text: str,
) -> None:
    clarification_already_used = bool(
        cl.user_session.get("clarification_used_for_current_question")
    )

    try:
        result = await cl.make_async(engine.answer)(
            question,
            history=history,
            clarification_already_used=clarification_already_used,
        )
    except Exception as e:
        await cl.Message(content=f"❌ שגיאה בעיבוד השאלה: {e}").send()
        return

    rtype = result["type"]
    rewritten = result.get("rewritten", question)
    if rewritten and rewritten != question:
        print(f"[rewritten query] {rewritten}")

    if rtype == "no_info":
        msg = "לא נמצא מידע מספיק במסמכים כדי לענות על השאלה."
        await cl.Message(content=msg).send()
        history.append({"role": "user", "content": history_user_text})
        history.append({"role": "assistant", "content": msg})
        cl.user_session.set("history", history)
        return

    if rtype == "clarification":
        options = result["options"]
        opts_text = "\n".join(
            f"{CLARIFICATION_LETTERS[i]}. {opt}" for i, opt in enumerate(options)
        )
        msg = (
            "כדי לענות בצורה מדויקת, אנא הבהר:\n\n"
            f"{opts_text}\n\n"
            "ניתן להשיב באות (א/ב/ג) או בטקסט חופשי."
        )
        cl.user_session.set("awaiting_clarification", True)
        cl.user_session.set("pending_clarification_options", options)
        cl.user_session.set("pending_original_question", history_user_text)
        await cl.Message(content=msg).send()
        return

    # rtype == "answer"
    text_out = result["text"]
    await cl.Message(content=text_out).send()
    history.append({"role": "user", "content": history_user_text})
    history.append({"role": "assistant", "content": text_out})
    cl.user_session.set("history", history)
