import json

import eval.snapshot as snapshot_module


def test_snapshot_replaces_the_corpus_and_records_when_it_was_taken(tmp_path, monkeypatch):
    articles = tmp_path / "articles.jsonl"
    articles.write_text(json.dumps({"id": "stale", "source": "old-feed"}) + "\n")
    manifest = tmp_path / "manifest.json"
    monkeypatch.setattr(snapshot_module, "ARTICLES_PATH", articles)
    monkeypatch.setattr(snapshot_module, "MANIFEST_PATH", manifest)

    polled = {}

    def fake_poll(config_path, data_path):
        polled.update(config=config_path, data=data_path)
        assert not data_path.exists(), "a snapshot starts from an empty file, not an append"
        data_path.write_text(
            "".join(
                json.dumps({"id": i, "source": s}) + "\n"
                for i, s in [("a", "go-blog"), ("b", "rust-blog"), ("c", "go-blog")]
            )
        )

    monkeypatch.setattr(snapshot_module, "poll", fake_poll)

    snapshot_module.snapshot()

    assert polled == {"config": snapshot_module.FEEDS_PATH, "data": articles}
    written = json.loads(manifest.read_text())
    assert written["articles"] == 3
    assert written["by_source"] == {"go-blog": 2, "rust-blog": 1}
    assert written["as_of"].endswith("+00:00")
