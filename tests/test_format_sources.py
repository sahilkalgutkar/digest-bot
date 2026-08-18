from generate.chat import format_sources


def test_format_sources_numbers_and_includes_metadata():
    chunks = [
        {"title": "A", "source": "feedA", "published_at": "2026-01-01", "text": "hello"},
        {"title": "B", "source": "feedB", "published_at": "2026-01-02", "text": "world"},
    ]
    result = format_sources(chunks)

    assert "[1] A (feedA, 2026-01-01)" in result
    assert "[2] B (feedB, 2026-01-02)" in result
    assert "hello" in result
    assert "world" in result


def test_format_sources_empty_list_returns_empty_string():
    assert format_sources([]) == ""
