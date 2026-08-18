"""Pure text-chunking logic, kept separate from build.py so it can be
unit tested without pulling in lancedb/sentence-transformers."""


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return [c for c in chunks if c.strip()]
