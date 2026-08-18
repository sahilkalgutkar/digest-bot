import json

import lancedb

import index.build as build_module
from index.build import build, load_articles, load_config


class FakeEncoded:
    def __init__(self, vec):
        self._vec = vec

    def tolist(self):
        return self._vec


class FakeModel:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        return [FakeEncoded([0.1, 0.2, 0.3]) for _ in texts]


def test_load_config(tmp_path, monkeypatch):
    config_file = tmp_path / "feeds.yaml"
    config_file.write_text("chunk_size: 500\nchunk_overlap: 50\n")
    monkeypatch.setattr(build_module, "CONFIG_PATH", config_file)

    assert load_config()["chunk_size"] == 500


def test_load_articles(tmp_path, monkeypatch):
    data_file = tmp_path / "articles.jsonl"
    data_file.write_text(json.dumps({"id": "a1", "text": "hello"}) + "\n")
    monkeypatch.setattr(build_module, "DATA_PATH", data_file)

    assert load_articles() == [{"id": "a1", "text": "hello"}]


def _write_article(data_file, **overrides):
    article = {
        "id": "a1",
        "source": "test-feed",
        "url": "http://example.com/a1",
        "title": "Article One",
        "published_at": "2026-01-01T00:00:00+00:00",
        "text": "x" * 45,
    }
    article.update(overrides)
    data_file.write_text(json.dumps(article) + "\n")


def test_build_indexes_chunks_into_lancedb(tmp_path, monkeypatch):
    config_file = tmp_path / "feeds.yaml"
    config_file.write_text("chunk_size: 20\nchunk_overlap: 5\n")
    data_file = tmp_path / "articles.jsonl"
    _write_article(data_file)
    db_path = tmp_path / "lancedb"

    monkeypatch.setattr(build_module, "CONFIG_PATH", config_file)
    monkeypatch.setattr(build_module, "DATA_PATH", data_file)
    monkeypatch.setattr(build_module, "DB_PATH", db_path)
    monkeypatch.setattr(build_module, "SentenceTransformer", FakeModel)

    build()

    db = lancedb.connect(db_path)
    table = db.open_table(build_module.TABLE_NAME)
    rows = table.to_arrow().to_pylist()

    assert table.count_rows() > 1  # 45 chars, chunk_size 20 overlap 5 -> multiple chunks
    assert all(r["article_id"] == "a1" for r in rows)
    assert all(r["source"] == "test-feed" for r in rows)
    assert all(len(r["vector"]) == 3 for r in rows)


def test_build_with_no_articles_prints_message_and_skips_indexing(tmp_path, monkeypatch, capsys):
    config_file = tmp_path / "feeds.yaml"
    config_file.write_text("chunk_size: 20\nchunk_overlap: 5\n")
    data_file = tmp_path / "articles.jsonl"
    data_file.write_text("")
    db_path = tmp_path / "lancedb"

    monkeypatch.setattr(build_module, "CONFIG_PATH", config_file)
    monkeypatch.setattr(build_module, "DATA_PATH", data_file)
    monkeypatch.setattr(build_module, "DB_PATH", db_path)
    monkeypatch.setattr(build_module, "SentenceTransformer", FakeModel)

    build()

    assert "No articles found" in capsys.readouterr().out
    assert not db_path.exists()
