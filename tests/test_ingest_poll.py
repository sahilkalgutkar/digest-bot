import json
import time
from datetime import datetime

import ingest.poll as poll_module
from ingest.poll import extract_text, load_config, load_seen_ids, poll, published_iso


class FakeEntry(dict):
    """Mimics feedparser's FeedParserDict: supports both attribute and dict access."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


def test_published_iso_uses_parsed_time():
    struct = time.struct_time((2026, 1, 15, 12, 0, 0, 0, 0, 0))
    entry = FakeEntry(published_parsed=struct)

    assert published_iso(entry).startswith("2026-01-15T12:00:00")


def test_published_iso_falls_back_to_now_when_missing():
    entry = FakeEntry()
    # should produce a valid ISO timestamp without raising
    datetime.fromisoformat(published_iso(entry))


def test_load_config(tmp_path, monkeypatch):
    config_file = tmp_path / "feeds.yaml"
    config_file.write_text("feeds:\n  - name: test\n    url: http://example.com\n")
    monkeypatch.setattr(poll_module, "CONFIG_PATH", config_file)

    assert load_config()["feeds"][0]["name"] == "test"


def test_load_seen_ids_empty_when_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr(poll_module, "DATA_PATH", tmp_path / "missing.jsonl")
    assert load_seen_ids() == set()


def test_load_seen_ids_reads_existing_ids(tmp_path, monkeypatch):
    data_file = tmp_path / "articles.jsonl"
    data_file.write_text(json.dumps({"id": "a1"}) + "\n" + json.dumps({"id": "a2"}) + "\n")
    monkeypatch.setattr(poll_module, "DATA_PATH", data_file)

    assert load_seen_ids() == {"a1", "a2"}


def test_extract_text_returns_none_when_fetch_fails(monkeypatch):
    monkeypatch.setattr(poll_module.trafilatura, "fetch_url", lambda url: None)
    assert extract_text("http://example.com") is None


def test_extract_text_returns_extracted_text(monkeypatch):
    monkeypatch.setattr(poll_module.trafilatura, "fetch_url", lambda url: "<html></html>")
    monkeypatch.setattr(poll_module.trafilatura, "extract", lambda html: "article body")
    assert extract_text("http://example.com") == "article body"


def test_poll_dedupes_across_runs(tmp_path, monkeypatch):
    config_file = tmp_path / "feeds.yaml"
    config_file.write_text("feeds:\n  - name: test-feed\n    url: http://example.com/rss\n")
    data_file = tmp_path / "articles.jsonl"

    monkeypatch.setattr(poll_module, "CONFIG_PATH", config_file)
    monkeypatch.setattr(poll_module, "DATA_PATH", data_file)

    entry = FakeEntry(id="article-1", link="http://example.com/a1", title="Article One")
    fake_parsed = type("FakeParsed", (), {"entries": [entry]})()

    monkeypatch.setattr(poll_module.feedparser, "parse", lambda url: fake_parsed)
    monkeypatch.setattr(poll_module, "extract_text", lambda url: "body text")

    poll()
    records = [json.loads(line) for line in data_file.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["id"] == "article-1"
    assert records[0]["source"] == "test-feed"

    poll()  # second run, same entry: should not duplicate
    assert len(data_file.read_text().splitlines()) == 1


def test_poll_skips_entries_where_extraction_fails(tmp_path, monkeypatch):
    config_file = tmp_path / "feeds.yaml"
    config_file.write_text("feeds:\n  - name: test-feed\n    url: http://example.com/rss\n")
    data_file = tmp_path / "articles.jsonl"

    monkeypatch.setattr(poll_module, "CONFIG_PATH", config_file)
    monkeypatch.setattr(poll_module, "DATA_PATH", data_file)

    entry = FakeEntry(id="article-2", link="http://example.com/a2", title="Unreachable")
    fake_parsed = type("FakeParsed", (), {"entries": [entry]})()

    monkeypatch.setattr(poll_module.feedparser, "parse", lambda url: fake_parsed)
    monkeypatch.setattr(poll_module, "extract_text", lambda url: None)

    poll()
    assert data_file.read_text() == ""


def _feed_config(tmp_path, extra=""):
    config_file = tmp_path / "feeds.yaml"
    config_file.write_text(
        "feeds:\n  - name: test-feed\n    url: http://example.com/rss\n" + extra
    )
    return config_file


def _entry(n, published=None):
    kwargs = {"id": f"article-{n}", "link": f"http://example.com/a{n}", "title": f"Article {n}"}
    if published is not None:
        kwargs["published_parsed"] = time.struct_time((2026, 1, published, 12, 0, 0, 0, 0, 0))
    return FakeEntry(**kwargs)


def test_poll_takes_only_the_newest_n_entries_per_feed(tmp_path, monkeypatch):
    """A feed that publishes its whole history costs one page fetch per entry.
    Without a cap the first poll is a fifteen minute one."""
    config_file = _feed_config(tmp_path, "max_articles_per_feed: 2\n")
    data_file = tmp_path / "articles.jsonl"
    monkeypatch.setattr(poll_module, "CONFIG_PATH", config_file)
    monkeypatch.setattr(poll_module, "DATA_PATH", data_file)

    entries = [_entry(1, published=1), _entry(2, published=2), _entry(3, published=3)]
    fake_parsed = type("FakeParsed", (), {"entries": entries})()
    monkeypatch.setattr(poll_module.feedparser, "parse", lambda url: fake_parsed)

    fetched = []
    monkeypatch.setattr(poll_module, "extract_text", lambda url: fetched.append(url) or "body")

    poll()

    records = [json.loads(line) for line in data_file.read_text().splitlines()]
    assert [r["id"] for r in records] == ["article-3", "article-2"], "should take the newest two"
    assert len(fetched) == 2, "an entry beyond the limit should never be fetched at all"


def test_poll_treats_zero_as_no_limit(tmp_path, monkeypatch):
    config_file = _feed_config(tmp_path, "max_articles_per_feed: 0\n")
    data_file = tmp_path / "articles.jsonl"
    monkeypatch.setattr(poll_module, "CONFIG_PATH", config_file)
    monkeypatch.setattr(poll_module, "DATA_PATH", data_file)

    entries = [_entry(n, published=n) for n in (1, 2, 3)]
    fake_parsed = type("FakeParsed", (), {"entries": entries})()
    monkeypatch.setattr(poll_module.feedparser, "parse", lambda url: fake_parsed)
    monkeypatch.setattr(poll_module, "extract_text", lambda url: "body")

    poll()
    assert len(data_file.read_text().splitlines()) == 3


def test_poll_caps_a_feed_whose_config_predates_the_setting(tmp_path, monkeypatch):
    config_file = _feed_config(tmp_path)  # no max_articles_per_feed key at all
    data_file = tmp_path / "articles.jsonl"
    monkeypatch.setattr(poll_module, "CONFIG_PATH", config_file)
    monkeypatch.setattr(poll_module, "DATA_PATH", data_file)

    entries = [_entry(n, published=1) for n in range(poll_module.DEFAULT_MAX_ARTICLES_PER_FEED + 5)]
    fake_parsed = type("FakeParsed", (), {"entries": entries})()
    monkeypatch.setattr(poll_module.feedparser, "parse", lambda url: fake_parsed)
    monkeypatch.setattr(poll_module, "extract_text", lambda url: "body")

    poll()
    assert (
        len(data_file.read_text().splitlines()) == poll_module.DEFAULT_MAX_ARTICLES_PER_FEED
    )


def test_poll_reports_a_feed_that_returned_nothing(tmp_path, monkeypatch, capsys):
    """A feed that has moved or 404s parses cleanly and yields no entries, so
    the only symptom is a lower article count -- invisible on a first run."""
    config_file = _feed_config(tmp_path)
    data_file = tmp_path / "articles.jsonl"
    monkeypatch.setattr(poll_module, "CONFIG_PATH", config_file)
    monkeypatch.setattr(poll_module, "DATA_PATH", data_file)

    fake_parsed = type("FakeParsed", (), {"entries": [], "status": 404})()
    monkeypatch.setattr(poll_module.feedparser, "parse", lambda url: fake_parsed)

    poll()

    out = capsys.readouterr().out
    assert "test-feed: no entries (status 404)" in out
