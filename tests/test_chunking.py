from index.chunking import chunk_text


def test_short_text_returns_single_chunk():
    assert chunk_text("hello world", size=100, overlap=10) == ["hello world"]


def test_chunks_respect_size_and_overlap():
    text = "a" * 250
    chunks = chunk_text(text, size=100, overlap=20)

    assert all(len(c) <= 100 for c in chunks)
    # each chunk after the first should start `size - overlap` chars into the previous one
    assert chunks[1] == text[80:180]


def test_empty_text_returns_no_chunks():
    assert chunk_text("", size=100, overlap=10) == []


def test_whitespace_only_chunks_are_dropped():
    chunks = chunk_text("word " + " " * 200, size=10, overlap=2)
    assert all(c.strip() for c in chunks)
