from datetime import datetime, timedelta, timezone

import lancedb

import retrieve.search as search_module
from retrieve.search import search


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
