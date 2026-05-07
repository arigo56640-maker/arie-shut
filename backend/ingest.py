"""
Ingestion script - one-time embedding of the Kitzur Shulchan Aruch corpus.

Run: python -m backend.ingest
"""
import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "backend" / "data" / "kitzur_json.json"
VECTOR_STORE_DIR = PROJECT_ROOT / "backend" / "vector_store"
EMBEDDINGS_PATH = VECTOR_STORE_DIR / "embeddings.npy"
METADATA_PATH = VECTOR_STORE_DIR / "metadata.json"

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072
BATCH_SIZE = 100

MAX_CHUNK_CHARS = 2000
SUB_CHUNK_SIZE = 1500
SUB_CHUNK_OVERLAP = 150


def split_long_content(content: str) -> list[str]:
    if len(content) <= MAX_CHUNK_CHARS:
        return [content]
    chunks = []
    start = 0
    while start < len(content):
        end = start + SUB_CHUNK_SIZE
        chunks.append(content[start:end])
        if end >= len(content):
            break
        start = end - SUB_CHUNK_OVERLAP
    return chunks


def build_chunks(corpus: list[dict]) -> tuple[list[str], list[dict]]:
    texts: list[str] = []
    metadata: list[dict] = []

    for entry in corpus:
        content = entry["content"]
        siman_title = entry["siman_title"]
        context_header = entry["metadata"]["context_header"]
        full_reference = entry["metadata"]["full_reference"]

        sub_chunks = split_long_content(content)
        n_parts = len(sub_chunks)

        for i, sub in enumerate(sub_chunks, start=1):
            ref = full_reference if n_parts == 1 else f"{full_reference} (חלק {i}/{n_parts})"
            embed_text = f"{siman_title}\n{context_header}\n{sub}"
            texts.append(embed_text)
            metadata.append({
                "full_reference": ref,
                "context_header": context_header,
                "content": sub,
                "siman_id": entry["siman_id"],
                "seif_id": entry["seif_id"],
                "siman_title": siman_title,
            })

    return texts, metadata


def embed_batch(client: OpenAI, texts: list[str]) -> np.ndarray:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    vectors = np.array([d.embedding for d in response.data], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY not set. Create a .env file in the project root "
            "with: OPENAI_API_KEY=sk-..."
        )

    print(f"Loading corpus from {DATA_PATH}...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    print(f"Loaded {len(corpus)} raw entries.")

    texts, metadata = build_chunks(corpus)
    n_chunks = len(texts)
    print(f"Built {n_chunks} chunks for embedding.")

    client = OpenAI()
    all_embeddings = np.zeros((n_chunks, EMBEDDING_DIM), dtype=np.float32)

    for batch_start in range(0, n_chunks, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, n_chunks)
        batch_texts = texts[batch_start:batch_end]
        batch_vectors = embed_batch(client, batch_texts)
        all_embeddings[batch_start:batch_end] = batch_vectors
        print(f"  Embedded {batch_end}/{n_chunks}")

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, all_embeddings)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {EMBEDDINGS_PATH}")
    print(f"  shape={all_embeddings.shape}, dtype={all_embeddings.dtype}")
    print(f"Saved {METADATA_PATH} ({len(metadata)} entries)")


if __name__ == "__main__":
    main()
