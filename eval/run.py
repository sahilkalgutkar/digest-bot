"""Score retrieval -- and, with an API key, citations -- against a hand-written
question set.

By default this runs the committed eval: the frozen corpus in eval/fixtures/
is indexed into a throwaway directory with the bot's own chunking and
embedding model, and retrieval is scored as of the snapshot time recorded in
eval/fixtures/manifest.json. The score therefore moves only when retrieval
code or config changes, never because the calendar did. This is what CI runs.

--live scores your own question set against your own index in data/, as of
now, the way the eval worked before there was a committed corpus.

Reported per question kind (lookup / latest) and overall:
- recall@1, recall@k: the expected article is the top chunk / in the top k
- MRR: mean of 1/rank of the expected article's first chunk (0 if missed)
- citation correctness, when ANTHROPIC_API_KEY is set: of the questions whose
  article was retrieved, how many generated answers cite one of its chunks

Run: python -m eval.run
     python -m eval.run --min-recall 0.8          # exit 1 below the floor
     python -m eval.run --recency-blend 0         # recency off
     python -m eval.run --recency-blend 0.2 --half-life-hours 168
     python -m eval.run --live --questions my-questions.yaml
"""

import argparse
import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

import yaml
from anthropic import Anthropic
from dotenv import load_dotenv

from eval.snapshot import ARTICLES_PATH, MANIFEST_PATH
from generate.chat import generate
from index.build import build
from retrieve.search import search

load_dotenv()

QUESTIONS_PATH = Path("eval/questions.yaml")
K = 5

# eval/questions.template.yaml carries this id. Scoring it would print a
# real-looking 0% that says nothing about retrieval, which is worse than
# printing nothing.
PLACEHOLDER_ARTICLE_ID = "REPLACE_WITH_REAL_ARTICLE_ID"


class UnfilledQuestionSet(Exception):
    pass


def load_questions(path=None):
    path = Path(path or QUESTIONS_PATH)
    with open(path) as f:
        questions = yaml.safe_load(f)["questions"]

    unfilled = [q for q in questions if q["expected_article_id"] == PLACEHOLDER_ARTICLE_ID]
    if unfilled:
        raise UnfilledQuestionSet(
            f"{len(unfilled)} of {len(questions)} questions in {path} still "
            f"carry the placeholder id.\n"
            "Fill them in first: open data/articles.jsonl, pick articles you can "
            "write a real question about, and copy each one's \"id\" field into "
            "expected_article_id."
        )
    return questions


def load_as_of() -> datetime:
    return datetime.fromisoformat(json.loads(MANIFEST_PATH.read_text())["as_of"])


def first_rank(article_id: str, chunks: list[dict]) -> int | None:
    """1-based position of the article's first chunk, or None if absent."""
    for i, chunk in enumerate(chunks, start=1):
        if chunk["article_id"] == article_id:
            return i
    return None


def summarize(ranks: list[int | None], k: int) -> dict:
    n = len(ranks)
    return {
        "n": n,
        "recall@1": sum(r == 1 for r in ranks) / n,
        f"recall@{k}": sum(r is not None for r in ranks) / n,
        "mrr": sum(1 / r for r in ranks if r) / n,
    }


def citation_is_correct(answer_text: str, article_id: str, chunks: list[dict]) -> bool:
    """Sources are numbered in retrieval order, and one article can fill several
    slots; citing any slot that belongs to the expected article counts."""
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer_text)}
    return any(
        chunks[n - 1]["article_id"] == article_id for n in cited if 1 <= n <= len(chunks)
    )


def run(
    questions_path=None,
    live: bool = False,
    min_recall: float | None = None,
    half_life_hours: float | None = None,
    recency_blend: float | None = None,
    retrieval_only: bool = False,
) -> int:
    try:
        questions = load_questions(questions_path)
    except UnfilledQuestionSet as err:
        print(err)
        return 1

    api_key = None if retrieval_only else os.environ.get("ANTHROPIC_API_KEY")
    client = Anthropic(api_key=api_key) if api_key else None

    with tempfile.TemporaryDirectory() as tmp:
        if live:
            db_path, now = None, None
        else:
            db_path, now = Path(tmp) / "lancedb", load_as_of()
            build(data_path=ARTICLES_PATH, db_path=db_path)
            print(f"Scoring the frozen corpus as of {now.isoformat()}\n")

        ranks_by_kind: dict[str, list[int | None]] = {}
        cited_ok = cited_total = 0

        for q in questions:
            kind = q.get("kind", "lookup")
            chunks = search(
                q["question"],
                k=K,
                db_path=db_path,
                now=now,
                half_life_hours=half_life_hours,
                recency_blend=recency_blend,
            )
            rank = first_rank(q["expected_article_id"], chunks)
            ranks_by_kind.setdefault(kind, []).append(rank)

            status = f"HIT @{rank}" if rank else "MISS   "
            print(f"[{status}] ({kind}) {q['question']}")

            if rank and client:
                text = generate(client, q["question"], chunks)
                correct = citation_is_correct(text, q["expected_article_id"], chunks)
                cited_ok += correct
                cited_total += 1
                print(f"    citation {'ok' if correct else 'MISSING/WRONG'}: {text[:120]}...")

    all_ranks = [r for ranks in ranks_by_kind.values() for r in ranks]
    hits = sum(r is not None for r in all_ranks)
    n = len(all_ranks)

    print(f"\n{'':8}{'n':>4}{'recall@1':>10}{f'recall@{K}':>10}{'MRR':>7}")
    for kind, ranks in [*sorted(ranks_by_kind.items()), ("all", all_ranks)]:
        s = summarize(ranks, K)
        print(f"{kind:8}{s['n']:>4}{s['recall@1']:>10.0%}{s[f'recall@{K}']:>10.0%}{s['mrr']:>7.2f}")

    print(f"\nrecall@{K}: {hits}/{n} ({hits/n:.0%})")
    if client:
        print(f"citation correctness (of hits): {cited_ok}/{cited_total or 1}")

    if min_recall is not None and hits / n < min_recall:
        print(f"FAIL: recall@{K} {hits/n:.1%} is below the {min_recall:.0%} floor")
        return 1
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--questions", help=f"question file (default {QUESTIONS_PATH})")
    parser.add_argument(
        "--live", action="store_true",
        help="score your own index in data/ as of now, instead of the frozen corpus",
    )
    parser.add_argument(
        "--min-recall", type=float,
        help=f"exit 1 if overall recall@{K} is below this fraction",
    )
    parser.add_argument(
        "--half-life-hours", type=float,
        help="override recency_half_life_hours from config/feeds.yaml",
    )
    parser.add_argument(
        "--recency-blend", type=float,
        help="override recency_blend from config/feeds.yaml; 0 turns recency off",
    )
    parser.add_argument(
        "--retrieval-only", action="store_true",
        help="skip the citation check even if ANTHROPIC_API_KEY is set (no API cost)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    return run(
        questions_path=args.questions,
        live=args.live,
        min_recall=args.min_recall,
        half_life_hours=args.half_life_hours,
        recency_blend=args.recency_blend,
        retrieval_only=args.retrieval_only,
    )


if __name__ == "__main__":
    raise SystemExit(main())
