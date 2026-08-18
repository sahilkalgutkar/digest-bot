"""Chunk articles from data/articles.jsonl, embed them, and (re)build the
LanceDB index at data/lancedb/ with published_at + source metadata on every
chunk, so retrieval can do recency-weighted search later.

Run: python -m index.build
"""

import json
from pathlib import Path

import lancedb
import yaml
from sentence_transformers import SentenceTransformer

CONFIG_PATH = Path("config/feeds.yaml")
DATA_PATH = Path("data/articles.jsonl")
DB_PATH = Path("data/lancedb")
TABLE_NAME = "chunks"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return [c for c in chunks if c.strip()]


def load_articles():
    with open(DATA_PATH) as f:
        return [json.loads(line) for line in f]


def build():
    config = load_config()
    articles = load_articles()
    model = SentenceTransformer(EMBED_MODEL)

    rows = []
    for article in articles:
        chunks = chunk_text(
            article["text"], config["chunk_size"], config["chunk_overlap"]
        )
        for i, chunk in enumerate(chunks):
            rows.append(
                {
                    "text": chunk,
                    "article_id": article["id"],
                    "chunk_index": i,
                    "source": article["source"],
                    "url": article["url"],
                    "title": article["title"],
                    "published_at": article["published_at"],
                }
            )

    if not rows:
        print("No articles found. Run `python -m ingest.poll` first.")
        return

    texts = [r["text"] for r in rows]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    for row, vec in zip(rows, embeddings):
        row["vector"] = vec.tolist()

    db = lancedb.connect(DB_PATH)
    db.create_table(TABLE_NAME, data=rows, mode="overwrite")

    print(f"Indexed {len(rows)} chunks from {len(articles)} articles -> {DB_PATH}")


if __name__ == "__main__":
    build()
