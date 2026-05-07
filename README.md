# מערכת RAG - קיצור שולחן ערוך

מערכת שאלות-ותשובות הלכתיות בעברית המבוססת על ספר "קיצור שולחן ערוך".

## דרישות

- Python 3.11.x
- Conda (סביבה בשם `Arie_RAG`)
- מפתח API של OpenAI

## התקנה

```bash
# יצירת סביבת קונדה
conda create -n Arie_RAG python=3.11 -y
conda activate Arie_RAG

# התקנת חבילות
pip install -r requirements.txt
```

## הגדרת מפתח API

צור קובץ `.env` בתיקיית השורש (העתק מ-`.env.example`):

```
OPENAI_API_KEY=sk-your-key-here
```

## שלבי הפעלה

### שלב 1: בניית מאגר הוקטורים (חד-פעמי)

```bash
python -m backend.ingest
```

זה יקודד את כל 2,758 הסעיפים ויצור:
- `backend/vector_store/embeddings.npy`
- `backend/vector_store/metadata.json`

עלות חד-פעמית: **~$0.06** (טקסט-embedding-3-large).

### שלב 2: הפעלת האפליקציה

```bash
chainlit run frontend/app.py -w
```

הדפדפן ייפתח אוטומטית בכתובת `http://localhost:8000`.

## ארכיטקטורה

```
\u200F\u200FArie_RAG_System3/
├── backend/
│   ├── data/kitzur_json.json     # הקורפוס
│   ├── vector_store/             # embeddings + metadata
│   ├── ingest.py                 # סקריפט קידוד חד-פעמי
│   └── rag_engine.py             # מנוע RAG
└── frontend/
    └── app.py                    # אפליקציית Chainlit
```

## פרמטרים מרכזיים

| פרמטר | ערך |
|-------|------|
| Embedding | `text-embedding-3-large` (3072 ממדים) |
| LLM | `gpt-4o-mini` |
| Top-K | 10 |
| Threshold | 0.60 (סף בסיסי), 0.65 (סף לתשובה ישירה) |
| Chunking | סעיף (Seif) |

## מצב מנהל

מילת המפתח `מנהל` בתחילת הודעה מפעילה placeholder להרחבות עתידיות. ראה `IMPLEMENTATION_PLAN.md` לרעיונות לעתיד.
