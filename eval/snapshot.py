"""Re-snapshot the eval corpus: poll the feeds in eval/fixtures/feeds.yaml into
a fresh eval/fixtures/articles.jsonl and record when the snapshot was taken.

The snapshot time matters as much as the articles. Retrieval weights chunks by
age, so eval scores the corpus as of this moment rather than as of whenever CI
happens to run -- otherwise every score would drift downward day by day as the
fixture ages.

Re-snapshotting replaces the articles eval/questions.yaml points at, so rewrite
the questions against the new corpus before committing it.

Run: python -m eval.snapshot
"""

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from ingest.poll import poll

FIXTURE_DIR = Path("eval/fixtures")
FEEDS_PATH = FIXTURE_DIR / "feeds.yaml"
ARTICLES_PATH = FIXTURE_DIR / "articles.jsonl"
MANIFEST_PATH = FIXTURE_DIR / "manifest.json"


def snapshot():
    as_of = datetime.now(tz=timezone.utc).replace(microsecond=0)
    ARTICLES_PATH.unlink(missing_ok=True)
    poll(config_path=FEEDS_PATH, data_path=ARTICLES_PATH)

    with open(ARTICLES_PATH) as f:
        by_source = Counter(json.loads(line)["source"] for line in f)
    manifest = {
        "as_of": as_of.isoformat(),
        "articles": sum(by_source.values()),
        "by_source": dict(sorted(by_source.items())),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Snapshot as of {manifest['as_of']}: {manifest['articles']} articles -> {ARTICLES_PATH}")


if __name__ == "__main__":
    snapshot()
