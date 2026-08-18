"""Run the hand-built question set against the current index and report:
- recall@k: did the expected article show up in the top-k retrieved chunks
- citation check: did the generated answer cite a source that actually
  points back to the expected article

This is intentionally simple (~20-30 questions, exact match) so it stays
easy to reason about — the point is a real number, not a fancy harness.

Run: python -m eval.run
"""

import os
import re
from pathlib import Path

import yaml
from anthropic import Anthropic
from dotenv import load_dotenv

from generate.chat import MODEL, SYSTEM_PROMPT, format_sources
from retrieve.search import search

load_dotenv()

QUESTIONS_PATH = Path("eval/questions.yaml")


def load_questions():
    with open(QUESTIONS_PATH) as f:
        return yaml.safe_load(f)["questions"]


def run():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    client = Anthropic(api_key=api_key) if api_key else None

    questions = load_questions()
    hits = 0
    citation_correct = 0

    for q in questions:
        chunks = search(q["question"], k=5)
        retrieved_ids = [c["article_id"] for c in chunks]
        hit = q["expected_article_id"] in retrieved_ids
        hits += hit

        status = "HIT " if hit else "MISS"
        print(f"[{status}] {q['question']}")

        if hit and client:
            expected_index = retrieved_ids.index(q["expected_article_id"]) + 1
            sources_block = format_sources(chunks)
            user_message = (
                f"Sources:\n\n{sources_block}\n\nQuestion: {q['question']}\n\n"
                "Answer using only the sources above, with [n] citations."
            )
            response = client.messages.create(
                model=MODEL,
                max_tokens=512,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            text = response.content[0].text
            cited = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
            correct = expected_index in cited
            citation_correct += correct
            print(f"    citation {'ok' if correct else 'MISSING/WRONG'}: {text[:120]}...")

    n = len(questions)
    print(f"\nrecall@5: {hits}/{n} ({hits/n:.0%})")
    if client:
        print(f"citation correctness (of hits): {citation_correct}/{hits or 1}")


if __name__ == "__main__":
    run()
