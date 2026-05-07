"""Demo - show the full picture of used_sources for a single query."""
import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.rag_engine import RAGEngine, SYSTEM_PROMPT, LLM_MODEL

engine = RAGEngine()
query = "מהן ברכות השחר?"

print(f"שאלה: {query}\n")
print("=" * 70)
print("שלב 1 - retrieval של 10 הצ'אנקים העליונים (לפני ה-LLM):")
print("=" * 70)

vec = engine.embed_query(query)
retrieved = engine.retrieve(vec, top_k=10)

for i, c in enumerate(retrieved, start=1):
    pct = round(c["score"] * 100)
    snippet = c["content"][:60].replace("\n", " ")
    print(f"  מקור {i:2d}: [{pct}%] {c['full_reference']}")
    print(f"           ↳ {snippet}...")

print()
print("=" * 70)
print("שלב 2 - שולחים את 10 המקורות ל-LLM ומבקשים JSON")
print("=" * 70)

sources_block_parts = []
for i, c in enumerate(retrieved, start=1):
    pct = round(c["score"] * 100)
    sources_block_parts.append(
        f"מקור {i} ({c['full_reference']}, התאמה: {pct}%):\n{c['content']}"
    )
sources_block = "\n\n".join(sources_block_parts)
user_content = f"שאלה: {query}\n\nמקורות:\n{sources_block}"

response = engine.client.chat.completions.create(
    model=LLM_MODEL,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ],
    temperature=0.2,
    response_format={"type": "json_object"},
)
raw = response.choices[0].message.content.strip()
parsed = json.loads(raw)

print()
print("ה-LLM החזיר את ה-JSON הבא:")
print(json.dumps(parsed, ensure_ascii=False, indent=2))
print()

print("=" * 70)
print("שלב 3 - הקוד מפרש את used_sources ובונה את בלוק המקורות")
print("=" * 70)

used = parsed["used_sources"]
print(f"\nused_sources = {used}\n")
print(f"זה אומר: ה-LLM הסתמך על המקורות הבאים מתוך 10 ששלחנו:\n")

for src_num in used:
    c = retrieved[src_num - 1]
    pct = round(c["score"] * 100)
    print(f"  ✓ מקור {src_num} = {c['full_reference']} (התאמה: {pct}%)")
    print(f"     ↳ {c['content'][:80]}...")
    print()

print("=" * 70)
print("שלב 4 - התוצאה הסופית למשתמש (גוף התשובה + בלוק מקורות):")
print("=" * 70)
print()
final = engine._format_with_citations(parsed["answer"], used, retrieved)
print(final)
