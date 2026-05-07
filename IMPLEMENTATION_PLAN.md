# תוכנית מימוש - מערכת RAG: קיצור שולחן ערוך

## Context
המטרה: בניית מערכת שאלות-ותשובות הלכתיות בעברית מבוססת על ספר "קיצור שולחן ערוך" (קובץ `kitzur_json.json` בן 2,758 סעיפים, 221 סימנים, סה"כ ~1.7MB). המערכת חייבת להיות Closed-book לחלוטין - להסתמך אך ורק על הקורפוס, ללא אינטרנט וללא ידע מוקדם של ה-LLM, כדי למנוע הזיות בתשובות הלכתיות שעלולות להטעות.

הצורך: כלי נגיש שמאפשר חיפוש איכותי ומדויק בתוך הספר עם ציטוט מקורות מדויק (סימן+סעיף) ושקיפות בדבר רמת הביטחון של התשובה.

תוצאה צפויה: אפליקציית Chainlit מקומית, שמקבלת שאלה בעברית, מאתרת את הסעיפים הרלוונטיים, מבקשת הבהרה כשצריך, ומחזירה תשובה מקצועית עם ציטוט המקורות (`full_reference`) ואחוזי התאמה.

## החלטות מרכזיות (לאחר שאלות הכוונה)
| נושא | החלטה |
|-----|-------|
| Chunking | **צ'אנק לפי סעיף (Seif)** - כל ערך ב-JSON הוא צ'אנק. סעיפים ארוכים (>2000 תווים, רק 2 כאלה) יפוצלו ל-sub-chunks תוך שמירת אותו `full_reference`. |
| Embedding | `text-embedding-3-large` של OpenAI (3072 ממדים) |
| Vector store | NumPy - מטריצה `.npy` של embeddings + JSON של מטה-דאטה |
| Retrieval | Top-K=10, similarity threshold=0.60 (קבוע, ללא לוגיקת סינון נוספת) |
| Clarification | היברידי: סף + Router LLM. **לכל היותר שאלת הבהרה אחת לכל מחזור שאלה** (flag בסשן). אזור אפור צר (0.60-0.65) + פרומפט Router שמרני - הבהרה רק בדו-משמעות סמנטית אמיתית. |
| Session Memory | Query Rewriting - שכתוב שאילתה לפני שליפה |
| LLM | `gpt-4o-mini` (גם ל-Router, גם לשכתוב שאילתה, גם לתשובה הסופית) |

## מבנה ה-JSON (אומת מהקובץ)
כל ערך הוא flat object:
```json
{
  "book": "קיצור שולחן ערוך",
  "siman_id": "א",
  "siman_title": "דיני השכמת הבוקר",
  "seif_id": "א",
  "content": "...",
  "metadata": {
    "full_reference": "קיצור שולחן ערוך, סימן א - דיני השכמת הבוקר, סעיף א",
    "context_header": "דיני השכמת הבוקר - סעיף א"
  }
}
```
שימוש: `metadata.full_reference` יוצג למשתמש בדיוק כפי שהוא בקובץ. `content` הוא הטקסט המקודד.

## מבנה הפרויקט
```
‏‏Arie_RAG_System3/                # תיקיית השורש (כבר קיימת עם kitzur_json.json)
│
├── .env                            # OPENAI_API_KEY
├── requirements.txt                # תלויות
├── README.md                       # הנחיות הפעלה
│
├── backend/
│   ├── __init__.py
│   ├── data/
│   │   └── kitzur_json.json        # יועתק/יזוז לכאן מתיקיית השורש
│   ├── vector_store/
│   │   ├── embeddings.npy          # מטריצה (N, 3072) float32
│   │   └── metadata.json           # רשימת ה-metadata לכל וקטור (אינדקס תואם)
│   ├── ingest.py                   # סקריפט one-time לבניית ה-vector store
│   └── rag_engine.py               # מחלקה RAGEngine - retrieval, clarification, generation
│
└── frontend/
    └── app.py                      # אפליקציית Chainlit
```

## פרטי המימוש

### 1. `backend/ingest.py` - סקריפט הזנת נתונים (one-time)
- קורא את `kitzur_json.json`.
- עבור כל entry:
  - אם `len(content) <= 2000` → צ'אנק יחיד.
  - אם ארוך יותר (רק 2 מקרים) → פיצול ל-sub-chunks של ~1500 תווים עם חפיפה של 150, **שומרים אותו `full_reference`** + סיומת `(חלק 1/N)`.
- בונה טקסט לקידוד: `f"{siman_title}\n{context_header}\n{content}"` - הקשר מסייע לאיכות ה-embedding.
- קורא ל-OpenAI API ב-batches של 100 (limit על request size).
- שומר:
  - `vector_store/embeddings.npy` - `np.array` בגודל `(N, 3072)`, `dtype=float32`, מנורמל ל-L2=1 (כדי שמכפלה סקלרית = cosine similarity).
  - `vector_store/metadata.json` - רשימה של dicts: `{"full_reference", "context_header", "content", "siman_id", "seif_id", "siman_title"}` באותו סדר כמו השורות ב-`embeddings.npy`.
- הסקריפט נטען ידנית: `python -m backend.ingest`. ללא caching בשלב זה (לפי בחירת המשתמש).
- עלות משוערת: ~$0.06 (חד-פעמי).

### 2. `backend/rag_engine.py` - מחלקת `RAGEngine`
מחלקה עם המתודות הציבוריות הבאות. **לא משתמשים ב-CrewAI Agents** - ה-PRD הסביר שהשימוש הוא בכלים בלבד, ולמעשה כאן עדיף לעקוף גם זאת ולקרוא ישירות לכלי OpenAI לשליטה מלאה ולמניעת תלויות מיותרות. (אם המשתמש מקפיד על שימוש ב-CrewAI tools - יש להתייעץ; ראה "פתוח להחלטה" למטה.)

**מתודות:**
1. `__init__(self)` - טוען לזיכרון את `embeddings.npy` ו-`metadata.json`. יוצר OpenAI client.
2. `embed_query(text: str) -> np.ndarray` - מקודד שאילתה, מנרמל L2.
3. `retrieve(query_vec, top_k=10) -> list[dict]` - מחשב similarity = `embeddings @ query_vec` (cosine, כי מנורמל), מחזיר Top-10 כולל `score` ו-`metadata`. **לא** מסנן ולא מחיל לוגיקה נוספת - רק מיון לפי score.
4. `rewrite_query(history, current_question) -> str` - קריאה ל-`gpt-4o-mini` עם פרומפט: "בהינתן השיחה הקודמת והשאלה הנוכחית, נסח שאלה standalone בעברית שמתייחסת להקשר". מחזיר שאלה משוכתבת. אם אין היסטוריה - מחזיר את השאלה כפי שהיא.
5. `decide_path(retrieved: list[dict], query: str, clarification_already_used: bool) -> dict` - עץ החלטות פשוט על בסיס הציון העליון:
   - **`top_score < 0.60`** → `{"path": "no_info"}` - fallback.
   - **`top_score >= 0.65`** → `{"path": "answer", "chunks": [c for c in retrieved if c.score >= 0.60]}` - תשובה ישירה.
   - **`0.60 <= top_score < 0.65`** (אזור אפור צר):
     - אם `clarification_already_used=True` → תשובה בכל מקרה (אין הבהרה שנייה).
     - אחרת → קריאה ל-Router LLM (`gpt-4o-mini`) עם השאלה ו-5 הכותרות (`context_header`) המובילות.
   - **פרומפט ה-Router (שמרני, ברירת מחדל "תענה"):**
     > "ענה למשתמש בכל מקרה אפשרי. בקש הבהרה **רק** במקרה של אי-בהירות סמנטית אמיתית (מילה במשמעויות לא קשורות, למשל 'נרות'=שבת/חנוכה/יום-טוב). שאלות רחבות שניתן לענות עליהן ממספר מקורות - אינן מצריכות הבהרה. ב-doubt, החזר `needs_clarification=false`."
   - Router מחזיר JSON `{"needs_clarification": bool, "options": [...]}`:
     - אם `true` → `{"path": "clarification", "options": [...]}`.
     - אם `false` → `{"path": "answer", ...}`.
6. `generate_answer(query, retrieved) -> str` - בונה context block מה-Top-K, קורא ל-`gpt-4o-mini` עם System Prompt (להלן) + השאלה + ה-context. מוודא שהתשובה כוללת את הפתיח החיווי ואת בלוק המקורות עם אחוזי ההתאמה.
7. `answer(question, history) -> dict` - מתודת תזמור (orchestrator) שעוטפת את הזרימה כולה: rewrite → embed → retrieve(K=10) → decide_path → generate (במידת הצורך). מחזירה: `{type: "answer"|"clarification"|"no_info", payload: ...}`.

**המרת similarity לאחוז:** `percent = round(score * 100)`. (נראה כ-60%-95% טיפוסית ב-`text-embedding-3-large` בעברית.)

**זרימת ההחלטה (סיכום ויזואלי):**
```
Top-10 score של #1:
  < 0.60         → "לא נמצא מידע" (fallback)
  ≥ 0.65         → תשובה ישירה (כל הצ'אנקים ≥ 0.60)
  0.60-0.65      → אזור אפור (צר):
    אם הבהרה כבר התבצעה → תשובה בכל זאת
    אחרת → Router LLM (פרומפט שמרני):
      "ספציפית/רחבה?" → תשובה
      "דו-משמעית סמנטית?" → שאלת הבהרה (א/ב/ג)
```

**הגבלת הבהרה - מימוש ב-Chainlit:** flag `clarification_used_for_current_question` נשמר ב-`cl.user_session`. מאופס ב-`@cl.on_chat_start` ולאחר כל **שאלה חדשה** (לא תשובה להבהרה). מובטח: לכל היותר הבהרה אחת לפני תשובה.

**שאלות המשך:** עוקפות את כל הלוגיקה הזו דרך Query Rewriting - השאילתה משוכתבת לטקסט standalone שמכיל את הקשר השיחה, ולכן הציון העליון יהיה כמעט תמיד ≥ 0.65 ותחזור ישר לתשובה.

### 3. System Prompt (נשמר כקבוע ב-`rag_engine.py`)
```text
אתה עוזר הלכתי המתמחה אך ורק בספר "קיצור שולחן ערוך".
אתה חייב לציית לכללים הבאים:

1. הסתמכות בלעדית: ענה אך ורק על בסיס הקטעים שמופיעים תחת "מקורות".
   חל איסור מוחלט להשתמש בידע חיצוני, לנחש, או להמציא.

2. אם המקורות אינם מספקים תשובה לשאלה - החזר אך ורק את המשפט:
   "לא נמצא מידע מספיק במסמכים כדי לענות על השאלה."

3. סגנון:
   - עברית בלבד.
   - תמציתי, מקצועי, ללא מילות מילוי.
   - **חובה לפתוח** את התשובה במשפט חיווי שכולל את שאלת המשתמש כהיגד מלא.
     דוגמה: שאלה="מהן שלוש תפילות יום החול?" → פתיח="שלוש תפילות יום החול הן: ..."

4. פורמט התשובה (חובה):
   [פתיח חיווי]
   [גוף התשובה הקצר]

   התשובה היא על סמך: [full_reference_1] (התאמה: X%), [full_reference_2] (התאמה: Y%)
   [content_1]
   [content_2]
```

### 4. `frontend/app.py` - אפליקציית Chainlit
- `@cl.on_chat_start`: טוען `RAGEngine` (singleton דרך `cl.user_session.set("engine", ...)`), מאתחל `history=[]`, `clarification_used_for_current_question=False`.
- `@cl.on_message`:
  - **שלב 0 - Admin Gateway:** אם הטקסט מתחיל ב-"מנהל" → ניתוב ל-`handle_admin(command)` (פונקציה נפרדת) ויציאה מוקדמת. בשלב ראשון זה יחזיר רק הודעת אישור placeholder. ראה "Admin Mode Roadmap" למטה לרעיונות עתידיים.
  - אם המשתמש משיב לשאלת הבהרה (סשן מסומן `awaiting_clarification=True`): ממיר את האות (א/ב/ג) לטקסט האפשרות, משלב עם השאלה המקורית, מסמן `clarification_used_for_current_question=True`, ומריץ retrieval מחדש (לא תחזור הבהרה שנייה).
  - אחרת (שאלה חדשה): מאפס `clarification_used_for_current_question=False` ומריץ `engine.answer(msg.content, history)`.
  - לפי `result["type"]`:
    - `"clarification"`: שולח הודעת הבהרה עם האפשרויות; מסמן `awaiting_clarification=True`.
    - `"answer"`: שולח את התשובה המעוצבת.
    - `"no_info"`: שולח את משפט ה-fallback הקבוע.
  - מעדכן `history` (שואל + תשובה).
- כל ה-UI בעברית: `cl.Message`, welcome message, הוראות שימוש.
- הגדרת RTL ב-Chainlit: באמצעות `chainlit.md` עם הוראות התקנה ושימוש בעברית.

### 5. `requirements.txt`
```
chainlit>=1.1.0
openai>=1.30.0
numpy>=1.26.0
python-dotenv>=1.0.0
```
**לא נכלל `crewai`** - ראו "פתוח להחלטה" בסוף.

### 6. `.env`
```
OPENAI_API_KEY=sk-...
```

## זרימת ריצה (קצה לקצה)
1. **חד-פעמי:** הקמת סביבת Conda `Arie_RAG`, התקנת `requirements.txt`, יצירת `.env`, הרצת `python -m backend.ingest`.
2. **תפעול:** `chainlit run frontend/app.py -w`.
3. למשתמש: דפדפן נפתח ב-`http://localhost:8000`, ממשק עברית, מקליד שאלה, מקבל תשובה / שאלת הבהרה.

## אימות (Verification)
1. **Ingestion:** ודא ש-`embeddings.npy` נוצר עם shape `(N, 3072)` (כש-N≈2760+) ו-`metadata.json` באורך זהה.
2. **Smoke retrieval:** הרץ סקריפט קטן (`python -c "..."`) ששואל "מהי ברכת השחר?" ומדפיס את 5 התוצאות + ציוני ההתאמה - ודא שהתוצאה הראשונה אכן מסימן רלוונטי (סימן ז או דומה).
3. **Clarification path:** שאל שאלה מעורפלת ("מתי מותר להדליק נרות?") - ודא שהמערכת מחזירה שאלת הבהרה עם אפשרויות (שבת/חנוכה/יום-טוב).
4. **Fallback path:** שאל שאלה מחוץ לקורפוס ("מה מזג האוויר היום?") - ודא שמתקבל המשפט הקבוע "לא נמצא מידע מספיק...".
5. **Session memory:** שאל "מהן ברכות השחר?" → ואחרי התשובה: "ומה הראשונה?" - ודא ש-Query Rewriting עובד (הצג ב-debug log את השאלה המשוכתבת).
6. **פורמט תשובה:** ודא שכל תשובה מתחילה במשפט חיווי שמשקף את השאלה, וסיומה כולל את `התשובה היא על סמך: ... (התאמה: X%)` עם הציטוטים.

## פתוח להחלטה (לפני התחלת מימוש)
1. **CrewAI tools:** ה-PRD הזכיר שימוש בכלי RAG של CrewAI (ללא Agents). בפועל - כלי `RagTool` ו-`JSONSearchTool` של CrewAI מקבעים backend (Chroma) ולא תומכים נטיב ב-NumPy. הצעתי בתוכנית: דילוג מוחלט על CrewAI ועבודה ישירה מול OpenAI + NumPy. **אם חשוב לך לשלב CrewAI דווקא** - יש להתייעץ על אופן ההסבה (אפשר לעטוף `RagEngine` כ-`BaseTool` של CrewAI, אך זה overhead ללא ערך כל עוד אין סוכנים).
2. **כפילויות גרסאות:** האם כדאי שהקובץ `kitzur_json.json` יישאר בשורש או יועבר ל-`backend/data/`? התוכנית מניחה העברה (פעולה קלה ובת-היפוך).

## Admin Mode Roadmap (רעיונות לעתיד)
ה-Hook מוטמע מראש ב-`@cl.on_message` - מילת המפתח `"מנהל"` בתחילת הודעה תנותב ל-`handle_admin()`. בשלב ראשון יחזיר רק placeholder. מועמדים להרחבה עתידית:

| קטגוריה | פקודות לדוגמה | תועלת |
|---------|---------------|-------|
| **Debug** | `מנהל verbose`, `מנהל timing`, `מנהל cost`, `מנהל prompt` | הצגת ציוני Top-10, latency, עלות, הפרומפט המלא |
| **Tuning** | `מנהל top_k=15`, `מנהל threshold=0.55`, `מנהל model=gpt-4o` | שינוי פרמטרים בזמן ריצה ללא הרצה מחדש |
| **Exploration** | `מנהל search "ברכה"`, `מנהל show סימן ז סעיף ה`, `מנהל stats`, `מנהל random` | חיפוש מילולי, גישה ישירה לפי ID, סטטיסטיקות |
| **Quality** | `מנהל feedback wrong`, `מנהל regression`, `מנהל compare` | feedback loop, golden-set testing, A/B |
| **Analytics** | `מנהל history`, `מנהל export`, `מנהל report` | יצוא סשן + מטריקות שימוש |
| **Management** | `מנהל reindex`, `מנהל health`, `מנהל clear` | תחזוקת המערכת |
| **Extension** | `מנהל add_corpus משנה_ברורה`, `מנהל translate en` | הוספת ספרים, תרגום |

ההטמעה הראשונית כוללת רק את ה-routing וה-placeholder; הפקודות עצמן יוטמעו על-פי דרישה.

## קבצים קריטיים שייווצרו
- `backend/ingest.py` (חדש)
- `backend/rag_engine.py` (חדש)
- `frontend/app.py` (חדש)
- `requirements.txt` (חדש)
- `.env` (חדש - בלי לשמור את המפתח ב-git)
- `chainlit.md` (חדש - ברכת פתיחה בעברית)
- העברת `kitzur_json.json` ל-`backend/data/`
