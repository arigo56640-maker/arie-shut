"""
Shared constants and pure helpers used by both frontend channels
(Chainlit web UI and WhatsApp adapter).

Anything channel-agnostic lives here so changes propagate to both
interfaces from a single source of truth.
"""
from __future__ import annotations

ADMIN_TRIGGER = "מנהל"
CLARIFICATION_LETTERS = ["א", "ב", "ג", "ד"]
NO_INFO_MESSAGE = "לא נמצא מידע מספיק במסמכים כדי לענות על השאלה."


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


def format_clarification_message(options: list[str]) -> str:
    opts_text = "\n".join(
        f"{CLARIFICATION_LETTERS[i]}. {opt}" for i, opt in enumerate(options)
    )
    return (
        "כדי לענות בצורה מדויקת, אנא הבהר:\n\n"
        f"{opts_text}\n\n"
        "ניתן להשיב באות (א/ב/ג) או בטקסט חופשי."
    )
