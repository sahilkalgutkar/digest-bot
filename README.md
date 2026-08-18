# digest-bot

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

A RAG chatbot over a rolling window of RSS/changelog feeds. Built as a learning
project for the parts of RAG that static-corpus tutorials skip: freshness,
dedup, incremental indexing, and recency-aware retrieval.

## How it works

1. **Ingest** (`ingest/`) — poll RSS feeds, extract readable article text, dedup by GUID/URL.
2. **Index** (`index/`) — chunk articles, embed chunks, store in a local vector DB with `published_at` + `source` metadata.
3. **Retrieve** (`retrieve/`) — hybrid search: vector similarity blended with a recency-decay weight.
4. **Generate** (`generate/`) — answer questions, forcing citations back to source articles; explicitly say "I don't have recent info" when retrieval is empty.
5. **Eval** (`eval/`) — hand-built question/answer set to measure retrieval recall and citation correctness, not vibes.

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
