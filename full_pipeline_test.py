"""Full pipeline test - run from project root."""
import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.rag_engine import RAGEngine

engine = RAGEngine()
print(f"Engine loaded: {engine.embeddings.shape[0]} chunks\n")

scenarios = [
    {"q": "מה אומרים בברכת מודה אני?", "history": []},
    {"q": "מה מזג האוויר היום?", "history": []},
    {"q": "מתי מותר להדליק נרות?", "history": []},
]

for i, sc in enumerate(scenarios, 1):
    print(f"{'='*70}")
    print(f"Scenario {i}: {sc['q']}")
    print(f"{'='*70}")
    result = engine.answer(sc["q"], history=sc["history"])
    print(f"  rewritten: {result['rewritten']}")
    print(f"  type:      {result['type']}")
    if result["type"] == "no_info":
        top = result["retrieved"][0]
        print(f"  top score: {round(top['score']*100)}% - {top['full_reference']}")
    elif result["type"] == "clarification":
        print(f"  options:   {result['options']}")
    elif result["type"] == "answer":
        print(f"  chunks used: {len(result['chunks'])}")
        for c in result["chunks"][:3]:
            print(f"    - [{round(c['score']*100)}%] {c['full_reference']}")
        print(f"\n  ANSWER:\n{result['text']}")
    print()
