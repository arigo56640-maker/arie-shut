"""
RAGEngine - retrieval, query rewriting, clarification routing, and answer generation
for the Kitzur Shulchan Aruch corpus.
"""
import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMBEDDINGS_PATH = PROJECT_ROOT / "backend" / "vector_store" / "embeddings.npy"
METADATA_PATH = PROJECT_ROOT / "backend" / "vector_store" / "metadata.json"

EMBEDDING_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4o-mini"

TOP_K = 5
# Calibrated for text-embedding-3-large on Hebrew. Hebrew scores cluster lower
# than English; off-topic queries score ~0.35-0.40, on-topic legit queries
# score ~0.50-0.66 even on direct matches.
THRESHOLD_MIN = 0.42      # below this → "no info" fallback


SYSTEM_PROMPT = """אתה עוזר הלכתי המתמחה אך ורק בספר "קיצור שולחן ערוך".

חוקים מחייבים:

1. **הסתמכות מילולית מוחלטת:** כל פרט בתשובה שלך חייב להופיע במפורש במקורות שיועברו אליך. חל איסור מוחלט על:
   - הזכרת ברכה, שם, אדם, מקום, תאריך, מנהג, או נוסח תפילה שלא מופיע בטקסט המקורות.
   - השלמת פרטים מ"ידע כללי" של עולם הספרות הרבנית או נוסח התפילה - גם אם הם נכונים הלכתית.
   - פירוט רשימה כשהמקור רק מציין "כל ה..." בלי לפרט.
   - הוספת ציטוטים, שמות תפילות, או נוסחאות שאינם מופיעים מילולית במקורות.
   - שימוש בידע מסידור התפילה הסטנדרטי, מהשולחן ערוך המלא, או ממקורות אחרים.

   **דוגמת הפרה (אסור!):** המקור אומר "מברך את כל ברכות השחר" - **אסור** לכתוב בתשובה "כמו 'פוקח עורים' ו'מלביש ערומים'", כי שמות הברכות הללו אינם מופיעים במקור.

   **דוגמה תקינה:** "המברך אומר את כל ברכות השחר כפי שמפורטות בקיצור שולחן ערוך."

2. **בדיקה עצמית לפני ההחזרה (חובה):** עבור על כל מילה בתשובה שכתבת. עבור כל ברכה, שם, ציטוט, נוסח, או פרט ספציפי - חפש אותו **באופן מילולי** במקורות שקיבלת. אם הוא אינו שם:
   - מחק אותו מהתשובה, או
   - החלף אותו בנוסח כללי ("כפי שמפורט בסימן X").

   **שאל את עצמך:** "האם המילה/השם הזה מופיע בטקסט המקורות שלפני, או שאני 'יודע' אותו ממקום אחר?"
   אם התשובה היא "אני יודע מהאימון שלי" - מחק את הפרט.

3. אם המקורות אינם מספקים תשובה - שדה answer יכיל אך ורק את המשפט:
   "לא נמצא מידע מספיק במסמכים כדי לענות על השאלה."
   ושדה used_sources יהיה מערך ריק [].

4. עברית בלבד. תמציתי, מקצועי, ללא מילות מילוי.

5. **חובה** לפתוח את התשובה במשפט חיווי שמשלב את שאלת המשתמש כהיגד שלם (לא לחזור על השאלה כשאלה).
   דוגמאות נכונות:
   - שאלה: "מהן שלוש תפילות יום החול?" → "שלוש תפילות יום החול הן שחרית, מנחה וערבית."
   - שאלה: "מתי מותר להדליק נרות?" → "מותר להדליק נרות..."
   - שאלה: "מה אומרים בברכת מודה אני?" → "בברכת מודה אני אומרים..."

6. החזר JSON תקין בלבד בפורמט הבא:
{
  "answer": "<גוף התשובה במלואו - פתיח חיווי + תוכן קצר ומקצועי>",
  "used_sources": [<מספרי המקורות מהם בנית את התשובה - למשל [2] או [1, 3]>]
}

7. used_sources חייב להכיל **רק** את מספרי המקורות שמהם בנית את התשובה (כפי שמסומנים "מקור 1", "מקור 2" וכו'):
   - אם השתמשת רק במקור 2 - החזר [2].
   - אם בנית מ-1 ו-3 - החזר [1, 3].
   - אל תוסיף מקורות שלא תרמו לתשובה (גם אם יש להם score גבוה).
   - אם המקורות אינם מספקים תשובה - החזר [].
"""


REWRITE_PROMPT = """בהינתן ההיסטוריה של השיחה והשאלה הנוכחית, נסח שאלה standalone בעברית שמכילה את כל ההקשר הנדרש כדי להבין אותה ללא ההיסטוריה.

- אם השאלה כבר standalone (לא מתייחסת להיסטוריה) - החזר אותה כפי שהיא.
- אם היא שאלת המשך - שכתב אותה כך שתעמוד בפני עצמה.
- אל תוסיף מידע שאינו בהיסטוריה.
- החזר אך ורק את השאלה המשוכתבת, ללא הסברים או טקסט נוסף.
"""


ROUTER_PROMPT = """אתה Router שתפקידו להחליט האם שאלת משתמש דורשת הבהרה לפני שניתן לענות עליה.

ענה למשתמש בכל מקרה אפשרי. בקש הבהרה רק במקרה של אי-בהירות סמנטית אמיתית - כלומר, מילה שיש לה משמעויות שונות שאינן קשורות זו לזו (לדוגמה: "נרות" = שבת/חנוכה/יום-טוב).

שאלות רחבות שניתן לענות עליהן ממספר מקורות - אינן מצריכות הבהרה.
ב-doubt, החזר needs_clarification=false.

החזר JSON תקין בלבד בפורמט הבא:
{"needs_clarification": true או false, "options": ["אפשרות 1", "אפשרות 2", "אפשרות 3"]}

- אם needs_clarification=false, options יכול להיות מערך ריק.
- אם needs_clarification=true, ספק עד 3 אפשרויות בעברית.
"""


class RAGEngine:
    def __init__(self) -> None:
        load_dotenv(PROJECT_ROOT / ".env")
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY not set. Create a .env file in the project root."
            )
        self.client = OpenAI()

        if not EMBEDDINGS_PATH.exists() or not METADATA_PATH.exists():
            raise RuntimeError(
                f"Vector store missing in {EMBEDDINGS_PATH.parent}. "
                "Run: python -m backend.ingest"
            )
        self.embeddings: np.ndarray = np.load(EMBEDDINGS_PATH)
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            self.metadata: list[dict] = json.load(f)

        if self.embeddings.shape[0] != len(self.metadata):
            raise RuntimeError(
                f"Mismatch: embeddings={self.embeddings.shape[0]} rows, "
                f"metadata={len(self.metadata)} entries."
            )

    def embed_query(self, text: str) -> np.ndarray:
        response = self.client.embeddings.create(model=EMBEDDING_MODEL, input=[text])
        vector = np.array(response.data[0].embedding, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector

    def retrieve(self, query_vec: np.ndarray, top_k: int = TOP_K) -> list[dict]:
        scores = self.embeddings @ query_vec  # cosine since both normalized
        top_indices = np.argsort(-scores)[:top_k]
        results = []
        for idx in top_indices:
            md = dict(self.metadata[idx])
            md["score"] = float(scores[idx])
            results.append(md)
        return results

    def rewrite_query(self, history: list[dict], current_question: str) -> str:
        if not history:
            return current_question
        recent = history[-6:]
        history_str = "\n".join(
            f"{turn['role']}: {turn['content']}" for turn in recent
        )
        user_content = (
            f"היסטוריית שיחה:\n{history_str}\n\nשאלה נוכחית: {current_question}"
        )
        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": REWRITE_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,
            )
            rewritten = response.choices[0].message.content.strip()
            return rewritten or current_question
        except Exception:
            return current_question

    def decide_path(
        self,
        retrieved: list[dict],
        query: str,
        clarification_already_used: bool,
    ) -> dict:
        if not retrieved:
            return {"path": "no_info"}

        top_score = retrieved[0]["score"]
        if top_score < THRESHOLD_MIN:
            return {"path": "no_info"}

        kept_chunks = [c for c in retrieved if c["score"] >= THRESHOLD_MIN]

        # If a clarification was already given for this question, answer directly.
        if clarification_already_used:
            return {"path": "answer", "chunks": kept_chunks}

        # Always run Router to detect semantic ambiguity in the query.
        # The Router prompt is conservative — it flags only true ambiguity
        # (e.g., "נרות" → Shabbat/Hanukkah/Yom Tov), so most queries pass through.
        headers_block = "\n".join(
            f"{i + 1}. {c['context_header']}" for i, c in enumerate(retrieved[:5])
        )
        user_content = (
            f"שאלה: {query}\n\nכותרות התוצאות העליונות:\n{headers_block}"
        )
        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": ROUTER_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            decision = json.loads(response.choices[0].message.content)
        except Exception:
            return {"path": "answer", "chunks": kept_chunks}

        if decision.get("needs_clarification"):
            options = [opt for opt in (decision.get("options") or []) if opt]
            if options:
                return {"path": "clarification", "options": options[:3]}
        return {"path": "answer", "chunks": kept_chunks}

    def generate_answer(self, query: str, retrieved: list[dict]) -> str:
        # Build numbered sources block. Numbers ARE meaningful - LLM returns them in used_sources.
        # Note: similarity score is intentionally NOT sent to the LLM - it should pick by content,
        # not by retrieval rank. The score is shown to the user only in the final citation block.
        sources_block_parts = []
        for i, c in enumerate(retrieved, start=1):
            sources_block_parts.append(
                f"מקור {i} ({c['full_reference']}):\n{c['content']}"
            )
        sources_block = "\n\n".join(sources_block_parts)
        user_content = f"שאלה: {query}\n\nמקורות:\n{sources_block}"

        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content.strip()

        # DEBUG: prepend the prompt + raw JSON to every answer for testing.
        # Remove this block when no longer needed.
        debug_block = (
            "<details>\n"
            "<summary><b>📤 הפרומפט שנשלח ל-LLM (לחץ לפתיחה)</b></summary>\n\n"
            "**System prompt:**\n"
            f"```\n{SYSTEM_PROMPT}\n```\n\n"
            "**User content:**\n"
            f"```\n{user_content}\n```\n\n"
            "</details>\n\n"
            "**📥 JSON שהתקבל מה-LLM:**\n"
            f"```json\n{raw}\n```\n\n---\n\n"
        )

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            print(f"[generate_answer] JSON parse failed, falling back. Raw: {raw[:200]}")
            return debug_block + self._format_fallback(raw, retrieved)

        answer = (result.get("answer") or "").strip()
        used_raw = result.get("used_sources") or []

        if not answer:
            return debug_block + self._format_fallback(raw, retrieved)

        if "לא נמצא מידע מספיק" in answer:
            return debug_block + answer  # No citation block on fallback message

        # Validate used_sources: must be ints in valid range, deduplicated, in order.
        valid_used: list[int] = []
        seen: set[int] = set()
        for u in used_raw:
            try:
                u_int = int(u)
            except (TypeError, ValueError):
                continue
            if 1 <= u_int <= len(retrieved) and u_int not in seen:
                valid_used.append(u_int)
                seen.add(u_int)

        if not valid_used:
            print(
                f"[generate_answer] LLM returned answer without used_sources. "
                f"Falling back to top-1. Answer: {answer[:80]}"
            )
            valid_used = [1]

        return debug_block + self._format_with_citations(answer, valid_used, retrieved)

    def _format_with_citations(
        self, answer: str, used_sources: list[int], retrieved: list[dict]
    ) -> str:
        lines = [answer, "", "---", ""]
        if len(used_sources) == 1:
            lines.append("📖 **המקור שעליו מבוססת התשובה:**")
        else:
            lines.append("📖 **המקורות שעליהם מבוססת התשובה:**")
        lines.append("")
        for i, src_num in enumerate(used_sources, start=1):
            c = retrieved[src_num - 1]
            percent = round(c["score"] * 100)
            if len(used_sources) == 1:
                heading = f"**{c['full_reference']}** (התאמה: {percent}%)"
            else:
                heading = f"**{i}. {c['full_reference']}** (התאמה: {percent}%)"
            lines.append(heading)
            lines.append("")
            quoted = "\n".join(
                f"> {ln}" if ln.strip() else ">" for ln in c["content"].split("\n")
            )
            lines.append(quoted)
            lines.append("")
        return "\n".join(lines).rstrip()

    def _format_fallback(self, raw_text: str, retrieved: list[dict]) -> str:
        # When JSON parsing fails: use raw text as answer body, top-1 as citation.
        lines = [raw_text]
        if retrieved:
            c = retrieved[0]
            percent = round(c["score"] * 100)
            lines.extend([
                "", "---", "",
                "📖 **מקור (fallback):**", "",
                f"**{c['full_reference']}** (התאמה: {percent}%)", "",
                f"> {c['content']}",
            ])
        return "\n".join(lines).rstrip()

    def answer(
        self,
        question: str,
        history: list[dict],
        clarification_already_used: bool = False,
    ) -> dict:
        rewritten = self.rewrite_query(history, question)
        query_vec = self.embed_query(rewritten)
        retrieved = self.retrieve(query_vec, top_k=TOP_K)
        decision = self.decide_path(retrieved, rewritten, clarification_already_used)

        base = {"rewritten": rewritten, "retrieved": retrieved}

        if decision["path"] == "no_info":
            return {**base, "type": "no_info"}
        if decision["path"] == "clarification":
            return {**base, "type": "clarification", "options": decision["options"]}

        chunks = decision["chunks"]
        text = self.generate_answer(rewritten, chunks)
        return {**base, "type": "answer", "text": text, "chunks": chunks}
