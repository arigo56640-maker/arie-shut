# מערכת RAG - קיצור שולחן ערוך

מערכת שאלות-ותשובות הלכתיות בעברית המבוססת על ספר "קיצור שולחן ערוך" (2,780 סעיפים). חלון תשובה סגור — המנוע מחזיר אך ורק את מה שכתוב בקורפוס, עם הפניה מילולית לסימן ולסעיף.

זמינה דרך **דפדפן (Chainlit)** ודרך **WhatsApp (Green API)** — שניהם מדברים אל אותו `RAGEngine`, כך ששינוי לוגיקה (פרומפט, סף, מודל) משפיע על שני הממשקים בו-זמנית.

## ארכיטקטורה

```mermaid
flowchart LR
    subgraph clients[משתמשי קצה]
        BR[דפדפן]
        WA[WhatsApp]
    end

    subgraph cloud[שירותי ענן]
        GA[Green API<br/>WhatsApp gateway]
    end

    subgraph railway[Railway]
        FE[frontend<br/>Chainlit<br/>:8080]
        BE[backend<br/>FastAPI<br/>:8080]
        VOL[(Volume<br/>vector_store)]
    end

    BR -->|HTTPS| FE
    WA <-->|messages| GA
    GA -->|webhook POST| BE
    GA <--|sendMessage| BE
    FE -->|"POST /answer<br/>(internal)"| BE
    BE --- VOL
    BE -->|embeddings + chat| OAI[OpenAI API]
```

**שלוש יחידות לוגיות, שני שירותים פרוסים:**
- `backend/rag_engine.py` — מנוע ה-RAG (retrieval, router, generation). מקור אמת יחיד.
- `backend/api.py` (FastAPI) — חושף את המנוע דרך `POST /answer` ודרך `POST /webhook/whatsapp`.
- `frontend/app.py` (Chainlit) — UI לדפדפן. קורא ל-backend ב-HTTP, לא מייבא את המנוע ישירות.
- `backend/whatsapp.py` — מתאם Green API. מקבל webhooks, קורא ל-`engine.answer()` בתוך אותו process של ה-FastAPI.
- `backend/shared.py` — קבועים ופונקציות עזר משותפות לשני הממשקים (פיענוח הבהרה, הודעת "אין מידע", מילת מפתח של מנהל).

## דרישות

- Python 3.11+
- Conda (סביבה בשם `Arie_RAG`)
- מפתח API של OpenAI

## התקנה

```powershell
conda create -n Arie_RAG python=3.11 -y
conda activate Arie_RAG
pip install -r requirements.txt
```

## הגדרת משתני סביבה

העתק את `.env.example` ל-`.env` ומלא:

```
OPENAI_API_KEY=sk-...
BACKEND_URL=http://localhost:8000        # אליו פונה ה-Chainlit
# WhatsApp (אופציונלי — אם ריקים, ה-adapter כבוי לחלוטין):
GREENAPI_INSTANCE_ID=
GREENAPI_API_TOKEN=
GREENAPI_WEBHOOK_TOKEN=                  # מחרוזת סודית שאתה ממציא, אופציונלי
```

## הפעלה מקומית

### 1. בניית מאגר וקטורים (חד-פעמי, ~$0.06)

```powershell
python -m backend.ingest
```

יוצר את `backend/vector_store/embeddings.npy` ו-`metadata.json`.

### 2. הפעלת שני השירותים (כל אחד בטרמינל נפרד)

```powershell
# Terminal 1 — backend (FastAPI + RAG + WhatsApp webhook)
uvicorn backend.api:app --host 0.0.0.0 --port 8000

# Terminal 2 — frontend (Chainlit)
chainlit run frontend/app.py -w --port 8001
```

הדפדפן: <http://localhost:8001>.

### 3. בדיקות

```powershell
python smoke_test.py            # K=5 retrieval scores לדוגמאות
python full_pipeline_test.py    # rewrite + retrieve + decide + generate, JSON מלא
```

## פריסה ל-Railway

הפרויקט פרוס ב-Railway (project `arie-shut`) כשני services שמשתפים את אותו ריפו:

| Service | קוד שמופעל | URL ציבורי |
|---|---|---|
| `backend` | `uvicorn backend.api:app --host 0.0.0.0 --port $PORT` | `backend-production-cb89.up.railway.app` |
| `frontend` | `chainlit run frontend/app.py --host 0.0.0.0 --port $PORT --headless` | `frontend-production-b648.up.railway.app` |

**Auto-deploy:** push ל-`main` → Railway בונה ופורס מחדש את שני ה-services אוטומטית.

### משתני סביבה ב-Railway

**backend** דורש:
- `OPENAI_API_KEY`
- `GREENAPI_INSTANCE_ID`, `GREENAPI_API_TOKEN` (כדי ש-`/webhook/whatsapp` יהיה enabled)
- `RAILWAY_VOLUME_MOUNT_PATH=/app/backend/vector_store` (נשמר את ה-vector store בין deploys)

**frontend** דורש:
- `BACKEND_URL=http://backend.railway.internal:8080` (פרטי ל-VPC של Railway)
- `PYTHONPATH=/app` ⚠️ **חובה** — בלי זה, Chainlit לא מצליח לעשות `from backend.shared import ...` מתוך `/app/frontend/app.py` (ה-shim של `chainlit run` לא מוסיף את ה-cwd ל-`sys.path`).

עדכון משתנה סביבה ב-Railway מפעיל deploy חדש אוטומטית.

## ממשק WhatsApp (Green API)

הזרימה זהה ל-Chainlit:
- מילת המפתח `מנהל` פותחת תפריט מנהל. `מנהל 1` מציג את הפרומפט המלא וה-JSON הגולמי של ה-LLM לשאלה האחרונה (לכל משתמש בנפרד).
- שאלות מעורפלות מקבלות הבהרה (א/ב/ג); ניתן להשיב באות או בטקסט חופשי.
- היסטוריית שיחה נשמרת בזיכרון לכל מספר טלפון ומאופסת ב-restart של השרת.

### חיבור Green API ל-Railway (production)

1. הרשם ב-[Green API](https://green-api.com), צור instance, שמור `idInstance` + `apiTokenInstance`.
2. הוסף את שני הערכים כמשתני סביבה ל-`backend` ב-Railway (`railway variable set GREENAPI_INSTANCE_ID=... --service backend`).
3. בקונסולת Green API, הגדר את ה-`webhookUrl` ל:
   ```
   https://backend-production-cb89.up.railway.app/webhook/whatsapp
   ```
   והפעל את ההתראה `incomingMessageReceived`. את שאר ההתראות (סטטוסים, הודעות יוצאות, שיחות) השאר על "לא".
4. סרוק QR פעם אחת מהטלפון (WhatsApp → מכשירים מקושרים).

### חיבור Green API מקומית (dev/debug)

לפיתוח ובדיקה — לפני שדוחפים שינויים לפרודקשן:
1. הפעל מקומית את שני ה-services (ראה "הפעלה מקומית" מעלה).
2. חשוף את ה-backend לאינטרנט: `cloudflared tunnel --url http://localhost:8000` (או `ngrok http 8000`).
3. עדכן את ה-`webhookUrl` ב-Green API לכתובת הזמנית שהכלי החזיר.
4. כשמסיימים — אל תשכח להחזיר את ה-URL ל-Railway.

## פרמטרים מרכזיים

נמצאים בראש [`backend/rag_engine.py`](backend/rag_engine.py):

| פרמטר | ערך | הערה |
|---|---|---|
| `EMBEDDING_MODEL` | `text-embedding-3-large` (3,072 ממדים) | החלפה דורשת re-ingest |
| `LLM_MODEL` | `gpt-4o-mini` | משמש ל-rewrite, router וגנרציה |
| `TOP_K` | `5` | חתיכות שנשלחות ל-LLM |
| `THRESHOLD_MIN` | `0.42` | מתחתיו → "לא נמצא מידע". מכויל לעברית — לא להעלות לערכים של אנגלית |
| `CLARIFICATION_ENABLED` | `False` | Router מושבת; ניתן להפעיל מחדש בדגל |
| `FOLLOWUP_ENABLED` | `False` | בלי שכתוב היסטוריה; כל שאלה standalone |
| `SHOW_MATCH_PERCENT` | `False` | מסתיר את "התאמה: X%" בציטוט |

## מצב מנהל

`מנהל` (לבד) → תפריט עם 4 אפשרויות (1 = debug של השאלה האחרונה, 2 = health check, 3+4 = בקרוב). הפלוט ב-`backend/api.py` ו-`backend/whatsapp.py` שומר את הפרומפט וה-JSON ב-session — לא מודפס לכל משתמש בלי בקשה מפורשת.

## מסמכי רקע

- [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) — תכנון מקורי. חלק מההחלטות התעדכנו (Router תמיד רץ → כעת מושבת; ספים שונו; ציטוט נבנה ב-Python; JSON-mode עם `used_sources`). הקוד הוא ה-truth.
- [`CLAUDE.md`](CLAUDE.md) — הוראות לסביבת Claude Code (RTL path caveat, Chainlit hot-reload, וכו').
- [`chainlit.md`](chainlit.md) — מסך פתיחה של Chainlit.
