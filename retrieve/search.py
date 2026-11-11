"""Recency-aware retrieval: vector similarity with a small, bounded recency
boost, so a fresher article wins a near-tie against an older one without ever
burying a clearly better older match.

decay = 0.5 ** (age_hours / half_life_hours)            # 1 now, 0.5 at one half-life
score = similarity * ((1 - recency_blend) + recency_blend * decay)

recency_blend caps what recency can do: at 0.05 the oldest article keeps 95%
of its similarity. An earlier version multiplied similarity by the full decay
(exp(-age/72h)), which let the one or two newest posts fill the top 5 for
almost any question -- recall@5 on the eval set was 23%. See README -> Eval.
"""

from datetime import datetime, timezone
from pathlib import Path

import lancedb
import yaml
from sentence_transformers import SentenceTransformer

CONFIG_PATH = Path("config/feeds.yaml")
DB_PATH = Path("data/lancedb")
TABLE_NAME = "chunks"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_RECENCY_BLEND = 0.05

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def recency_weight(
    published_at: str, half_life_hours: float, now: datetime | None = None
) -> float:
    """Halves every half_life_hours: 1 for a brand-new article, 0.5 at one
    half-life. now defaults to the wall clock; eval pins it to the corpus
    snapshot time so a frozen corpus scores the same next month as today. A
    half-life of math.inf turns the decay off (1 at every age)."""
    now = now or datetime.now(tz=timezone.utc)
    published = datetime.fromisoformat(published_at)
    age_hours = (now - published).total_seconds() / 3600
    return 0.5 ** (max(age_hours, 0) / half_life_hours)


def search(
    query: str,
    k: int = 5,
    candidate_pool: int = 25,
    *,
    db_path: Path | None = None,
    now: datetime | None = None,
    half_life_hours: float | None = None,
    recency_blend: float | None = None,
) -> list[dict]:
    """The keyword arguments default to the bot's own index, the wall clock, and
    the recency settings in config/feeds.yaml; eval overrides them."""
    config = load_config()
    if half_life_hours is None:
        half_life_hours = config["recency_half_life_hours"]
    if recency_blend is None:
        recency_blend = config.get("recency_blend", DEFAULT_RECENCY_BLEND)

    db = lancedb.connect(db_path or DB_PATH)
    table = db.open_table(TABLE_NAME)

    query_vec = get_model().encode(query, normalize_embeddings=True).tolist()
    candidates = table.search(query_vec).limit(candidate_pool).to_list()

    for row in candidates:
        # LanceDB returns squared L2 distance on normalized vectors;
        # convert to a similarity in [0, 1] before blending with recency.
        similarity = 1 - row["_distance"] / 2
        decay = recency_weight(row["published_at"], half_life_hours, now)
        row["score"] = similarity * ((1 - recency_blend) + recency_blend * decay)

    candidates.sort(key=lambda r: r["score"], reverse=True)
    return candidates[:k]


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "What's new?"
    for r in search(query):
        print(f"[{r['score']:.3f}] {r['title']} ({r['source']}, {r['published_at']})")
        print(f"  {r['text'][:150]}...")
