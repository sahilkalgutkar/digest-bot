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

# Used when a config predates max_articles_per_feed. A feed that publishes its
# whole history in one document turns an unbounded poll into a page fetch per
# entry, so the default is a cap rather than everything.
DEFAULT_MAX_ARTICLES_PER_FEED = 25


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


def sort_key(entry):
    """Newest first, with undated entries last rather than at an arbitrary
    position -- a feed that dates nothing should not displace one that does."""
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    return timegm(parsed) if parsed else 0


def extract_text(url: str) -> str | None:
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return None
    return trafilatura.extract(downloaded)


def poll():
    config = load_config()
    seen_ids = load_seen_ids()
    DATA_PATH.parent.mkdir(exist_ok=True)
    limit = config.get("max_articles_per_feed", DEFAULT_MAX_ARTICLES_PER_FEED)

    new_count = 0
    with open(DATA_PATH, "a") as out:
        for feed in config["feeds"]:
            parsed = feedparser.parse(feed["url"])
            entries = parsed.entries
            if not entries:
                # A feed that moved or 404s parses fine and yields nothing, so
                # without this the only symptom is a lower article count than
                # expected -- which is invisible on a first run.
                print(f"{feed['name']}: no entries (status {getattr(parsed, 'status', '?')}) — check the URL")
                continue

            # Newest first, so a limit takes the recent end of the feed rather
            # than whatever order the publisher happened to emit.
            entries = sorted(entries, key=sort_key, reverse=True)
            if limit:
                entries = entries[:limit]

            feed_count = 0
            for entry in entries:
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
                feed_count += 1
                new_count += 1

            print(f"{feed['name']}: {feed_count} new from {len(entries)} entries")

    print(f"Ingested {new_count} new articles -> {DATA_PATH}")


if __name__ == "__main__":
    poll()
