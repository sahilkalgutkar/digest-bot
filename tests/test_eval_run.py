import eval.run as eval_module
from eval.run import load_questions


def test_load_questions(tmp_path, monkeypatch):
    questions_file = tmp_path / "questions.yaml"
    questions_file.write_text(
        'questions:\n  - question: "What happened?"\n    expected_article_id: "a1"\n'
    )
    monkeypatch.setattr(eval_module, "QUESTIONS_PATH", questions_file)

    assert load_questions() == [{"question": "What happened?", "expected_article_id": "a1"}]


def test_run_reports_recall_across_hits_and_misses(tmp_path, monkeypatch, capsys):
    questions_file = tmp_path / "questions.yaml"
    questions_file.write_text(
        "questions:\n"
        '  - question: "hit question"\n    expected_article_id: "a1"\n'
        '  - question: "miss question"\n    expected_article_id: "missing"\n'
    )
    monkeypatch.setattr(eval_module, "QUESTIONS_PATH", questions_file)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def fake_search(question, k=5):
        if question == "hit question":
            return [{"article_id": "a1"}]
        return [{"article_id": "other"}]

    monkeypatch.setattr(eval_module, "search", fake_search)

    eval_module.run()

    out = capsys.readouterr().out
    assert "[HIT ] hit question" in out
    assert "[MISS] miss question" in out
    assert "recall@5: 1/2 (50%)" in out
    # no API key -> citation checks (and the Anthropic client) should never run
    assert "citation" not in out


class FakeTextBlock:
    def __init__(self, text):
        self.text = text


class FakeResponse:
    def __init__(self, text):
        self.content = [FakeTextBlock(text)]


class FakeMessages:
    def __init__(self, texts):
        self._texts = iter(texts)

    def create(self, **kwargs):
        return FakeResponse(next(self._texts))


class FakeAnthropic:
    def __init__(self, texts):
        self._texts = texts

    def __call__(self, api_key=None):
        self.messages = FakeMessages(self._texts)
        return self


def _chunk(article_id):
    return {
        "article_id": article_id,
        "title": "Title",
        "source": "feed",
        "published_at": "2026-01-01",
        "text": "body",
    }


def test_run_checks_citation_correctness_when_client_present(tmp_path, monkeypatch, capsys):
    questions_file = tmp_path / "questions.yaml"
    questions_file.write_text(
        "questions:\n"
        '  - question: "correctly cited"\n    expected_article_id: "a1"\n'
        '  - question: "wrongly cited"\n    expected_article_id: "a1"\n'
    )
    monkeypatch.setattr(eval_module, "QUESTIONS_PATH", questions_file)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")

    # both questions retrieve the expected article as the sole (first) chunk,
    # so the correct citation is always [1]
    monkeypatch.setattr(eval_module, "search", lambda question, k=5: [_chunk("a1")])
    # first response cites [1] correctly, second cites the wrong source
    monkeypatch.setattr(eval_module, "Anthropic", FakeAnthropic(["answer [1]", "answer [2]"]))

    eval_module.run()

    out = capsys.readouterr().out
    assert "citation ok: answer [1]" in out
    assert "citation MISSING/WRONG: answer [2]" in out
    assert "recall@5: 2/2 (100%)" in out
    assert "citation correctness (of hits): 1/2" in out
