import pytest

import eval.run as eval_module
from eval.run import UnfilledQuestionSet, load_questions


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


def test_load_questions_refuses_the_unfilled_template(tmp_path, monkeypatch):
    """Scoring the shipped template prints a real-looking 0% that says nothing
    about retrieval, which is worse than printing nothing."""
    questions_file = tmp_path / "questions.yaml"
    questions_file.write_text(
        "questions:\n"
        '  - question: "real question"\n    expected_article_id: "a1"\n'
        '  - question: "template question"\n'
        '    expected_article_id: "REPLACE_WITH_REAL_ARTICLE_ID"\n'
    )
    monkeypatch.setattr(eval_module, "QUESTIONS_PATH", questions_file)

    with pytest.raises(UnfilledQuestionSet, match="1 of 2 questions"):
        load_questions()


def test_run_exits_nonzero_on_the_unfilled_template(tmp_path, monkeypatch, capsys):
    questions_file = tmp_path / "questions.yaml"
    questions_file.write_text(
        "questions:\n"
        '  - question: "template question"\n'
        '    expected_article_id: "REPLACE_WITH_REAL_ARTICLE_ID"\n'
    )
    monkeypatch.setattr(eval_module, "QUESTIONS_PATH", questions_file)

    called = []
    monkeypatch.setattr(eval_module, "search", lambda *a, **k: called.append(1) or [])

    assert eval_module.run() == 1
    out = capsys.readouterr().out
    assert "still carry the placeholder id" in out
    assert "recall@5" not in out, "no score should be reported at all"
    assert not called, "retrieval should not run against a template question set"


def test_the_shipped_template_is_still_recognised_as_unfilled():
    """If the placeholder id in eval/questions.yaml is ever renamed without
    updating run.py, the guard silently stops guarding."""
    with pytest.raises(UnfilledQuestionSet):
        load_questions()
