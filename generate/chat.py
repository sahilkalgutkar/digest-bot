"""CLI chat loop. Retrieves relevant chunks, forces the model to cite them
inline, and explicitly refuses to answer when retrieval comes up empty
instead of letting the model guess from parametric knowledge.

Run: python -m generate.chat
"""

import os

from anthropic import Anthropic
from dotenv import load_dotenv

from retrieve.search import search

load_dotenv()

MODEL = "claude-sonnet-5"

SYSTEM_PROMPT = """You are digest-bot, a news assistant that answers ONLY from \
the provided source excerpts.

Rules:
- Every factual claim must be followed by a citation like [1], [2] referencing \
the numbered sources below.
- If the sources don't contain the answer, say so explicitly. Do not fall back \
on general knowledge.
- Keep answers concise."""


def format_sources(chunks: list[dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, start=1):
        lines.append(
            f"[{i}] {c['title']} ({c['source']}, {c['published_at']})\n{c['text']}"
        )
    return "\n\n".join(lines)


def answer(client: Anthropic, question: str) -> str:
    chunks = search(question, k=5)
    if not chunks:
        return "I don't have any indexed articles to answer that from. Run ingest + index first."

    sources_block = format_sources(chunks)
    user_message = (
        f"Sources:\n\n{sources_block}\n\nQuestion: {question}\n\n"
        "Answer using only the sources above, with [n] citations."
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("Set ANTHROPIC_API_KEY in .env")

    client = Anthropic(api_key=api_key)
    print("digest-bot ready. Ctrl+C to exit.\n")

    while True:
        try:
            question = input("> ")
        except (EOFError, KeyboardInterrupt):
            break
        if not question.strip():
            continue
        print(answer(client, question), "\n")


if __name__ == "__main__":
    main()
