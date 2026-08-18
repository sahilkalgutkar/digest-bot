"""Hybrid retrieval: vector similarity blended with a recency-decay weight,
so a strong semantic match from six months ago doesn't drown out a weaker
match from this morning.

score = cosine_similarity * exp(-age_hours / half_life_hours)
"""

import math
from datetime import datetime, timezone
from pathlib import Path

import lancedb
import yaml
from sentence_transformers import SentenceTransformer

CONFIG_PATH = Path("config/feeds.yaml")
DB_PATH = Path("data/lancedb")
TABLE_NAME = "chunks"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def recency_weight(published_at: str, half_life_hours: float) -> float:
    published = datetime.fromisoformat(published_at)
    age_hours = (datetime.now(tz=timezone.utc) - published).total_seconds() / 3600
    return math.exp(-max(age_hours, 0) / half_life_hours)


def search(query: str, k: int = 5, candidate_pool: int = 25) -> list[dict]:
    config = load_config()
    half_life = config["recency_half_life_hours"]

    db = lancedb.connect(DB_PATH)
    table = db.open_table(TABLE_NAME)

    query_vec = get_model().encode(query, normalize_embeddings=True).tolist()
    candidates = table.search(query_vec).limit(candidate_pool).to_list()

    for row in candidates:
        # LanceDB returns squared L2 distance on normalized vectors;
        # convert to a similarity in [0, 1] before blending with recency.
        similarity = 1 - row["_distance"] / 2
        row["score"] = similarity * recency_weight(row["published_at"], half_life)

    candidates.sort(key=lambda r: r["score"], reverse=True)
    return candidates[:k]


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "What's new?"
    for r in search(query):
        print(f"[{r['score']:.3f}] {r['title']} ({r['source']}, {r['published_at']})")
        print(f"  {r['text'][:150]}...")
