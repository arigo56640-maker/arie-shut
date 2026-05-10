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

# Toggle the Router-based clarification flow ("did you mean X or Y?").
# Disabled per user request — answer directly whenever score >= THRESHOLD_MIN.
# Flip to True to re-enable the Router.
CLARIFICATION_ENABLED = False

# Toggle follow-up handling. When False, rewrite_query ignores chat history and
# returns the current question as-is — every question is treated as standalone.
# Flip to True to re-enable history-aware query rewriting.
FOLLOWUP_ENABLED = False

# Toggle the per-source match percentage shown next to each citation.
# When False, citations are rendered without "(התאמה: X%)". The score is still
# computed and used internally; only the user-visible suffix is suppressed.
# Flip to True to re-enable the percentage in the citation block.
SHOW_MATCH_PERCENT = False


SYSTEM_PROMPT = """אתה עוזר הלכתי המתמחה אך ורק בספר "קיצור שולחן ערוך".

חוקים מחייבים:

1. **הסתמכות מילולית מוחלטת:** כל פרט בתשובה שלך חייב להופיע במפורש במקורות שיועברו אליך. חל איסור מוחלט על:
   - הזכרת ברכה, שם, אדם, מקום, תאריך, מנהג, או נוסח תפילה שלא מופיע בטקסט המקורות.
   - השלמת פרטים מ"ידע כללי" של עולם הספרות הרבנית או נוסח התפילה - גם אם הם נכונים הלכתית.
   - פירוט רשימה כשהמקור רק מציין "כל ה..." בלי לפרט.
   - הוספת ציטוטים, שמות תפילות, או נוסחאות שאינם מופיעים מילולית במקורות.
   - שימוש בידע מסידור התפילה הסטנדרטי, מהשולחן ערוך המלא, או ממקורות אחרים.

   **דוגמת הפרה (אסור!):** המקור אומר "מברך את כל ברכות השחר" - **אסור** לכתוב בתשובה "כמו 'פוקח עורים' ו'מלביש ערומים'", כי שמות הברכות הללו אינם מופיעים במקור.

   **דוגמה תקינה:** "המברך אומר את כל ברכות השחר."

2. **בדיקה עצמית לפני ההחזרה (חובה):** עבור על כל מילה בתשובה שכתבת. עבור כל ברכה, שם, ציטוט, נוסח, או פרט ספציפי - חפש אותו **באופן מילולי** במקורות שקיבלת. אם הוא אינו שם:
   - מחק אותו מהתשובה, או
   - נסח אותו באופן כללי בלי להוסיף פרטים.

   **שאל את עצמך:** "האם המילה/השם הזה מופיע בטקסט המקורות שלפני, או שאני 'יודע' אותו ממקום אחר?"
   אם התשובה היא "אני יודע מהאימון שלי" - מחק את הפרט.

3. **חוזה דו-כיווני מחייב בין answer ל-used_sources:**

   א. אם המקורות אינם מספקים תשובה - שדה answer יכיל אך ורק את המשפט:
      "לא נמצא מידע מספיק במסמכים כדי לענות על השאלה."
      ושדה used_sources יהיה מערך ריק [].

   ב. **חל איסור מוחלט להחזיר תשובה ממשית עם used_sources ריק.** אם אינך יכול להצביע על מקור ספציפי (מספר 1-N) שממנו בנית את התשובה - **זה אומר שאתה עונה מהזיכרון שלך**, וזה אסור. במקרה כזה אתה חייב להחזיר את משפט ה"לא נמצא מידע" כפי שמתואר בסעיף א.

   ג. **בדיקה אחרונה לפני שליחת ה-JSON:** עצור ושאל את עצמך - "אם אני מסיר עכשיו את המקורות שקיבלתי, האם הייתי יכול לכתוב את התשובה הזו?" אם כן - זו תשובה מהזיכרון. החלף את answer במשפט "לא נמצא מידע מספיק במסמכים כדי לענות על השאלה." ואת used_sources ב-[].

   ד. השילוב **answer מלא תוכן + used_sources=[]** הוא הזיה. הוא אסור באופן מוחלט. אין אף מקרה שבו הוא לגיטימי.

4. עברית בלבד. תמציתי, מקצועי, ללא מילות מילוי.

5. **חובה** לפתוח את התשובה במשפט חיווי שמשלב את שאלת המשתמש כהיגד שלם (לא לחזור על השאלה כשאלה).
   דוגמאות נכונות:
   - שאלה: "מהן שלוש תפילות יום החול?" → "שלוש תפילות יום החול הן שחרית, מנחה וערבית."
   - שאלה: "מתי מותר להדליק נרות?" → "מותר להדליק נרות..."
   - שאלה: "מה אומרים בברכת מודה אני?" → "בברכת מודה אני אומרים..."

6. **איסור הזכרת המקור בתוך גוף התשובה.** שדה answer חייב להכיל אך ורק את התשובה ההלכתית עצמה - בלי להזכיר היכן היא נמצאת. אסור להשתמש בביטויים כגון:
   - "כפי שמפורט בקיצור שולחן ערוך"
   - "על פי קיצור שולחן ערוך"
   - "בסימן X נאמר..."
   - "המקור אומר..."
   - "לפי הסעיף..."
   - "כמו שכתוב בספר..."
   - או כל ניסוח אחר שמפנה למקור, לסימן, לסעיף, או לספר.

   **ייחוס המקור נעשה אך ורק דרך השדה used_sources** - לא בתוך הטקסט של answer.

   **דוגמת הפרה (אסור!):** "שלוש תפילות יום החול הן שחרית, מנחה וערבית, כפי שמפורט בקיצור שולחן ערוך."
   **דוגמה תקינה:** "שלוש תפילות יום החול הן שחרית, מנחה וערבית."

7. החזר JSON תקין בלבד בפורמט הבא:
{
  "answer": "<גוף התשובה במלואו - פתיח חיווי + תוכן קצר ומקצועי>",
  "used_sources": [<מספרי המקורות מהם בנית את התשובה - למשל [2] או [1, 3]>]
}

8. used_sources חייב להכיל **רק** את מספרי המקורות שמהם בנית את התשובה (כפי שמסומנים "מקור 1", "מקור 2" וכו'):
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
        if not FOLLOWUP_ENABLED:
            return current_question
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

        # Clarification disabled — answer directly without invoking the Router.
        if not CLARIFICATION_ENABLED:
            return {"path": "answer", "chunks": kept_chunks}

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

    def generate_answer(self, query: str, retrieved: list[dict]) -> dict:
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
        # message.content can legitimately be None (e.g. refusal in JSON mode).
        raw = (response.choices[0].message.content or "").strip()

        # USER_DEBUG_BLOCK: per user requirement, the prompt sent and the raw JSON
        # returned must be exposed for inspection. Previously prepended to every
        # answer; now returned as separate fields so the frontend can show them
        # only when the user asks for them via admin mode. Keep this contract:
        # `last_prompt` and `last_raw_json` MUST be present on every answer dict.
        last_prompt = (
            "**System prompt:**\n"
            f"```\n{SYSTEM_PROMPT}\n```\n\n"
            "**User content:**\n"
            f"```\n{user_content}\n```"
        )
        debug = {"last_prompt": last_prompt, "last_raw_json": raw}

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            print(f"[generate_answer] JSON parse failed, falling back. Raw: {raw[:200]}")
            return {"text": self._format_fallback(raw, retrieved), **debug}

        answer = (parsed.get("answer") or "").strip()
        used_raw = parsed.get("used_sources") or []

        if not answer:
            return {"text": self._format_fallback(raw, retrieved), **debug}

        if "לא נמצא מידע מספיק" in answer:
            return {"text": answer, **debug}

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

        return {
            "text": self._format_with_citations(answer, valid_used, retrieved),
            **debug,
        }

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
            suffix = f" (התאמה: {percent}%)" if SHOW_MATCH_PERCENT else ""
            if len(used_sources) == 1:
                heading = f"**{c['full_reference']}**{suffix}"
            else:
                heading = f"**{i}. {c['full_reference']}**{suffix}"
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
            suffix = f" (התאמה: {percent}%)" if SHOW_MATCH_PERCENT else ""
            lines.extend([
                "", "---", "",
                "📖 **מקור (fallback):**", "",
                f"**{c['full_reference']}**{suffix}", "",
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
        gen = self.generate_answer(rewritten, chunks)
        return {
            **base,
            "type": "answer",
            "text": gen["text"],
            "chunks": chunks,
            "last_prompt": gen["last_prompt"],
            "last_raw_json": gen["last_raw_json"],
        }
