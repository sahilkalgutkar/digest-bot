"""Poll configured RSS feeds, extract readable article text, and append
new (deduped) articles to data/articles.jsonl.

Run: python -m ingest.poll
"""

import json
from calendar import timegm
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import trafilatura
import yaml

CONFIG_PATH = Path("config/feeds.yaml")
DATA_PATH = Path("data/articles.jsonl")


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_seen_ids():
    if not DATA_PATH.exists():
        return set()
    seen = set()
    with open(DATA_PATH) as f:
        for line in f:
            seen.add(json.loads(line)["id"])
    return seen


def published_iso(entry) -> str:
    """feedparser exposes a normalized time.struct_time when it can parse the
    feed's date; fall back to ingestion time so recency decay always has
    something usable."""
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed:
        dt = datetime.fromtimestamp(timegm(parsed), tz=timezone.utc)
    else:
        dt = datetime.now(tz=timezone.utc)
    return dt.isoformat()


def extract_text(url: str) -> str | None:
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return None
    return trafilatura.extract(downloaded)


def poll():
    config = load_config()
    seen_ids = load_seen_ids()
    DATA_PATH.parent.mkdir(exist_ok=True)

    new_count = 0
    with open(DATA_PATH, "a") as out:
        for feed in config["feeds"]:
            parsed = feedparser.parse(feed["url"])
            for entry in parsed.entries:
                article_id = entry.get("id") or entry.get("link")
                if article_id in seen_ids:
                    continue

                text = extract_text(entry.link)
                if not text:
                    continue

                record = {
                    "id": article_id,
                    "source": feed["name"],
                    "url": entry.link,
                    "title": entry.get("title", ""),
                    "published_at": published_iso(entry),
                    "text": text,
                }
                out.write(json.dumps(record) + "\n")
                seen_ids.add(article_id)
                new_count += 1

    print(f"Ingested {new_count} new articles -> {DATA_PATH}")


if __name__ == "__main__":
    poll()
