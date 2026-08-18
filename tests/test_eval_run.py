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
