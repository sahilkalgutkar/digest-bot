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
3. **Retrieve** (`retrieve/`) — I blend vector similarity with a recency-decay weight so a fresher, slightly-less-similar chunk can outrank an older, closer one. This is recency-weighted vector search, not hybrid (keyword + vector) retrieval — there's no BM25/keyword component.
4. **Generate** (`generate/`) — answer questions, forcing citations back to source articles; explicitly say "I don't have recent info" when retrieval is empty.
5. **Eval** (`eval/`) — a harness that measures retrieval recall@k and citation correctness against a hand-written question set, so the answer to "does this work" is a number rather than vibes. The question set itself ships as a template, because an article id only means something against the feeds you actually poll — see [Eval](#eval).

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

## Eval

```bash
python -m eval.run
```

Runs the question set in `eval/questions.yaml` against the current index and
reports retrieval recall@k and whether generated answers cited the right
source.

`eval/questions.yaml` ships as a template rather than as my own question set,
because an `expected_article_id` is only meaningful against the feeds it came
from — point the bot at a different beat and every id in it is dead. Fill it in
from `data/articles.jsonl` after the first index build. Until then `eval.run`
says so and exits, instead of scoring the placeholder and reporting a 0% that
looks like a retrieval failure.
