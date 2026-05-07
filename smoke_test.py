"""Smoke test for retrieval - run from project root. Writes UTF-8 output."""
import sys
import io
from pathlib import Path

# Force UTF-8 output (Windows console can mangle Hebrew otherwise)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.rag_engine import RAGEngine

engine = RAGEngine()
print(f"Loaded {engine.embeddings.shape[0]} chunks, dim={engine.embeddings.shape[1]}")
print()

queries = [
    "מהן ברכות השחר?",
    "כיצד נוטלים ידיים שחרית?",
    "מה אומרים בברכת מודה אני?",
    "מה הם דיני השכמת הבוקר?",
    "מתי מותר להדליק נרות?",
    "מה מזג האוויר היום?",
]

for q in queries:
    print(f"=== Query: {q} ===")
    vec = engine.embed_query(q)
    results = engine.retrieve(vec, top_k=5)
    for i, r in enumerate(results, 1):
        score_pct = round(r["score"] * 100)
        print(f"  {i}. [{score_pct}%] {r['full_reference']}")
    print()
