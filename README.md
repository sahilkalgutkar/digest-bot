# digest-bot

[![CI](https://github.com/sahilkalgutkar/digest-bot/actions/workflows/ci.yml/badge.svg)](https://github.com/sahilkalgutkar/digest-bot/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/sahilkalgutkar/digest-bot/branch/main/graph/badge.svg)](https://codecov.io/gh/sahilkalgutkar/digest-bot)
[![patch coverage](https://img.shields.io/badge/patch%20coverage-min%2080%25-blue.svg)](codecov.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

I built a RAG chatbot that answers questions over a rolling window of RSS/changelog
feeds instead of a static, one-time-indexed corpus. Most RAG tutorials index a
fixed set of documents once and stop there — I wanted to work through the parts
they skip: keeping the index fresh as new articles arrive, deduping across
re-polls, and weighting retrieval toward what's recent instead of just what's
semantically closest.

## How it works

1. **Ingest** (`ingest/`) — poll RSS feeds, extract readable article text, dedup by GUID/URL.
2. **Index** (`index/`) — chunk articles, embed chunks, store in a local vector DB with `published_at` + `source` metadata.
3. **Retrieve** (`retrieve/`) — vector similarity with a small, bounded recency boost: `score = similarity × ((1 − b) + b × 0.5^(age / half-life))`, with `b = 0.05`. A fresher article wins a near-tie, but an older article that clearly answers the question still wins. The bound is there because the eval showed what happens without it — see [Eval](#eval). This is recency-weighted vector search, not hybrid (keyword + vector) retrieval — there's no BM25/keyword component.
4. **Generate** (`generate/`) — answer questions, forcing citations back to source articles; explicitly say "I don't have recent info" when retrieval is empty.
5. **Eval** (`eval/`) — 43 hand-written questions over a frozen, committed snapshot of 55 articles, scored for recall@1, recall@5 and MRR on every push (CI fails below 90% recall@5), plus citation correctness when an API key is available. "Does this work" is a number rather than vibes — see [Eval](#eval).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # add your ANTHROPIC_API_KEY

python -m ingest.poll       # pull + store new articles
python -m index.build       # chunk + embed + index
python -m generate.chat     # ask questions in a CLI loop
```

Only the API key lives in `.env`; the feeds are in `config/feeds.yaml`. The
first `ingest.poll` takes a minute or two — it fetches the full text of every
article it takes, and prints a per-feed count as it goes. `index.build`
downloads the embedding model on its first run.

## Config

Feeds live in `config/feeds.yaml`. Add/remove RSS URLs there, along with
`max_articles_per_feed` — some feeds publish their whole history in one
document (the OpenAI blog is over a thousand entries), and since every entry
costs a page fetch, an unbounded poll is a fifteen-minute one. The default
takes the 25 newest per feed; set it to 0 for no limit.

`recency_half_life_hours` and `recency_blend` set how much freshness counts in
retrieval. `recency_blend` is the most recency can move a score; raise it and
the bot favours new posts more, at the cost of finding older ones. Measure the
change with `python -m eval.run --recency-blend <value>` before committing it.

## Eval

```bash
python -m eval.run                      # retrieval + citations (citations need ANTHROPIC_API_KEY)
python -m eval.run --retrieval-only     # no model calls, no cost -- what CI runs
python -m eval.run --recency-blend 0.2  # score a different retrieval setting
```

**The corpus is frozen.** `eval/fixtures/articles.jsonl` is a snapshot of 55
posts from the Kubernetes, Go and Rust blogs (openly licensed — see
[ATTRIBUTION](eval/fixtures/ATTRIBUTION.md)), taken by `python -m eval.snapshot`.
`eval/fixtures/manifest.json` records when. The eval indexes the snapshot into a
throwaway directory with the bot's own chunking and embedding model, and scores
retrieval *as of the snapshot time*, so the number only moves when retrieval
code or config changes — not because the corpus got a day older.

**The questions** (`eval/questions.yaml`) are 43 questions I wrote by reading
the articles, each pointing at the one article that answers it, phrased the way
someone would ask rather than by echoing the headline:

- **39 lookups** — a fact from one specific article, of any age. Several sit
  among near-neighbours on purpose: Memory QoS vs the cgroup v2 post, Go's two
  SIMD posts, C variadics in Rust 1.99.0 vs the variadic-generics survey.
- **4 "latest"** — "what's the newest rustup release?", where the corpus also
  holds the previous, near-identical post (rustup 1.29.2 vs 1.29.3). Only the
  newer one is right, so recall@1 is what counts.

**Citation correctness** — for every question whose article was retrieved,
generate an answer with the bot's real prompt and check that a `[n]` citation
points at a chunk of the expected article. It needs `ANTHROPIC_API_KEY`, so CI
doesn't run it on every push; run it locally or from the **Citation eval**
workflow (Actions → Citation eval → Run workflow) once the repository has an
`ANTHROPIC_API_KEY` secret.

### What it found

The first run against the original retrieval scored **23% recall@5**. The
formula was `similarity × exp(−age / 72h)`: most of the corpus is one to three
months old, so its weight was effectively zero, and the two posts from the last
day (Rust's GSoC results and "Go Turns 17") filled 112 of the 215 top-5 slots
regardless of the question. The design intent — let a fresher, *slightly*
less similar article win — wasn't what the multiplier did.

Bounding how much recency can move a score fixed it. Retrieval only,
as of the snapshot:

| Retrieval | lookup recall@5 | lookup MRR | latest recall@1 | all recall@5 |
|---|---|---|---|---|
| Original: `sim × exp(−age/72h)` | 23% | 0.10 | 0% | 23% |
| Bounded, `recency_blend: 0.2` | 82% | 0.67 | 0% | 81% |
| **Bounded, `recency_blend: 0.05` (default)** | **97%** | **0.90** | **75%** | **98%** |
| Recency off, `recency_blend: 0` | 100% | 0.94 | 75% | 100% |

Two honest caveats. Turning recency off entirely scores slightly higher on this
set — the 5% tie-breaker costs one lookup (the GPU/linker compiler groups
post). I kept it because the bot's job is a rolling digest of what's new, and
four "latest" questions are too few to show that side of the trade; growing
that set is the next thing to add. And the weak "latest" question ("most recent
stable Rust release" → 1.99.1) ranks 5th even with recency off: the 1.99.0
announcement is simply a closer semantic match, and a stronger recency boost
pushes 1.99.1 out of the top 5 altogether in favour of newer, unrelated posts.

### Your own feeds

The committed questions only mean something against the frozen corpus. To
score your own index, copy `eval/questions.template.yaml`, fill it in from
`data/articles.jsonl` after an index build, and run
`python -m eval.run --live --questions <your file>`. Until every placeholder id
is replaced it refuses to report a score rather than printing a 0% that looks
like a retrieval failure.

### Refreshing the snapshot

`python -m eval.snapshot` re-polls the feeds in `eval/fixtures/feeds.yaml` and
replaces the corpus, which retires the articles the questions point at —
rewrite `eval/questions.yaml` in the same change. A test fails if any question
points at an article that isn't in the corpus.
