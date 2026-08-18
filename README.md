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
3. **Retrieve** (`retrieve/`) — hybrid search: I blend vector similarity with a recency-decay weight so a fresher, slightly-less-similar chunk can outrank an older, closer one.
4. **Generate** (`generate/`) — answer questions, forcing citations back to source articles; explicitly say "I don't have recent info" when retrieval is empty.
5. **Eval** (`eval/`) — I hand-built a question/answer set to measure retrieval recall and citation correctness, not vibes.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # add your feeds + API key

python -m ingest.poll       # pull + store new articles
python -m index.build       # chunk + embed + index
python -m generate.chat     # ask questions in a CLI loop
```

## Config

Feeds live in `config/feeds.yaml`. Add/remove RSS URLs there.

## Eval

```bash
python -m eval.run
```

Runs the question set in `eval/questions.yaml` against the current index and
reports retrieval recall@k and whether generated answers cited the right source.
