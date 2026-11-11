from datetime import datetime, timedelta, timezone

import lancedb

import retrieve.search as search_module
from retrieve.search import get_model, search


class FakeEncodedQuery:
    def __init__(self, vec):
        self._vec = vec

    def tolist(self):
        return self._vec


class FakeModel:
    def encode(self, query, normalize_embeddings=True):
        return FakeEncodedQuery([1.0, 0.0, 0.0])


def _make_table(db_path, rows):
    db = lancedb.connect(db_path)
    db.create_table(search_module.TABLE_NAME, data=rows, mode="overwrite")


def test_search_prefers_recent_article_over_equally_similar_old_one(tmp_path, monkeypatch):
    now = datetime.now(timezone.utc)
    recent = (now - timedelta(hours=1)).isoformat()
    old = (now - timedelta(days=30)).isoformat()

    rows = [
        {
            "text": "old match",
            "article_id": "a2",
            "chunk_index": 0,
            "source": "feed",
            "url": "http://example.com/2",
            "title": "Old",
            "published_at": old,
            "vector": [1.0, 0.0, 0.0],
        },
        {
            "text": "recent match",
            "article_id": "a1",
            "chunk_index": 0,
            "source": "feed",
            "url": "http://example.com/1",
            "title": "Recent",
            "published_at": recent,
            "vector": [1.0, 0.0, 0.0],
        },
    ]

    config_file = tmp_path / "feeds.yaml"
    config_file.write_text("recency_half_life_hours: 24\n")
    db_path = tmp_path / "lancedb"
    _make_table(db_path, rows)

    monkeypatch.setattr(search_module, "CONFIG_PATH", config_file)
    monkeypatch.setattr(search_module, "DB_PATH", db_path)
    monkeypatch.setattr(search_module, "get_model", lambda: FakeModel())

    results = search("anything", k=2)

    assert [r["article_id"] for r in results] == ["a1", "a2"]
    assert results[0]["score"] > results[1]["score"]


def test_search_respects_k_limit(tmp_path, monkeypatch):
    now = datetime.now(timezone.utc).isoformat()
    rows = [
        {
            "text": f"match {i}",
            "article_id": f"a{i}",
            "chunk_index": 0,
            "source": "feed",
            "url": f"http://example.com/{i}",
            "title": f"Article {i}",
            "published_at": now,
            "vector": [1.0, 0.0, 0.0],
        }
        for i in range(5)
    ]

    config_file = tmp_path / "feeds.yaml"
    config_file.write_text("recency_half_life_hours: 24\n")
    db_path = tmp_path / "lancedb"
    _make_table(db_path, rows)

    monkeypatch.setattr(search_module, "CONFIG_PATH", config_file)
    monkeypatch.setattr(search_module, "DB_PATH", db_path)
    monkeypatch.setattr(search_module, "get_model", lambda: FakeModel())

    assert len(search("anything", k=2)) == 2


def _row(article_id, vector, published_at):
    return {
        "text": article_id,
        "article_id": article_id,
        "chunk_index": 0,
        "source": "feed",
        "url": f"http://example.com/{article_id}",
        "title": article_id,
        "published_at": published_at,
        "vector": vector,
    }


def test_a_clearly_better_old_match_beats_a_weaker_fresh_one(tmp_path, monkeypatch):
    """The regression the eval caught: multiplying similarity by the full decay
    let any recent article bury a month-old one that answers the question.
    Recency is a tie-breaker now, not a veto."""
    now = datetime.now(timezone.utc)
    rows = [
        _row("fresh-but-off-topic", [0.8, 0.6, 0.0], now.isoformat()),
        _row("old-and-on-topic", [1.0, 0.0, 0.0], (now - timedelta(days=60)).isoformat()),
    ]
    config_file = tmp_path / "feeds.yaml"
    config_file.write_text("recency_half_life_hours: 72\nrecency_blend: 0.05\n")
    db_path = tmp_path / "lancedb"
    _make_table(db_path, rows)
    monkeypatch.setattr(search_module, "CONFIG_PATH", config_file)
    monkeypatch.setattr(search_module, "DB_PATH", db_path)
    monkeypatch.setattr(search_module, "get_model", lambda: FakeModel())

    assert [r["article_id"] for r in search("q", k=2)] == ["old-and-on-topic", "fresh-but-off-topic"]
    # with the full decay (blend 1) the fresh article wins, which is the old bug
    assert search("q", k=1, recency_blend=1.0)[0]["article_id"] == "fresh-but-off-topic"


def test_search_takes_an_explicit_index_clock_and_recency_settings(tmp_path, monkeypatch):
    snapshot = datetime(2026, 1, 10, tzinfo=timezone.utc)
    rows = [
        _row("a-day-old", [1.0, 0.0, 0.0], (snapshot - timedelta(days=1)).isoformat()),
        _row("a-week-old", [1.0, 0.0, 0.0], (snapshot - timedelta(days=7)).isoformat()),
    ]
    config_file = tmp_path / "feeds.yaml"
    config_file.write_text("recency_half_life_hours: 72\n")
    db_path = tmp_path / "elsewhere"
    _make_table(db_path, rows)
    monkeypatch.setattr(search_module, "CONFIG_PATH", config_file)
    monkeypatch.setattr(search_module, "DB_PATH", tmp_path / "unused")
    monkeypatch.setattr(search_module, "get_model", lambda: FakeModel())

    results = search("q", k=2, db_path=db_path, now=snapshot, half_life_hours=24, recency_blend=1.0)

    assert [r["article_id"] for r in results] == ["a-day-old", "a-week-old"]
    assert results[0]["score"] == 0.5  # similarity 1, exactly one 24h half-life old
    off = search("q", k=2, db_path=db_path, now=snapshot, recency_blend=0.0)
    assert [r["score"] for r in off] == [1.0, 1.0]


def test_get_model_loads_the_embedding_model_only_once(monkeypatch):
    monkeypatch.setattr(search_module, "_model", None)

    instances_created = []

    class FakeSentenceTransformer:
        def __init__(self, model_name):
            self.model_name = model_name
            instances_created.append(self)

    monkeypatch.setattr(search_module, "SentenceTransformer", FakeSentenceTransformer)

    first = get_model()
    second = get_model()

    assert first is second
    assert len(instances_created) == 1
    assert instances_created[0].model_name == search_module.EMBED_MODEL
