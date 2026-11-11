import json
from datetime import datetime, timezone

import pytest
import yaml

import eval.run as eval_module
from eval.run import (
    UnfilledQuestionSet,
    citation_is_correct,
    first_rank,
    load_questions,
    summarize,
)


def _questions_file(tmp_path, *questions):
    path = tmp_path / "questions.yaml"
    path.write_text(yaml.safe_dump({"questions": list(questions)}))
    return path


def _chunk(article_id):
    return {
        "article_id": article_id,
        "title": "Title",
        "source": "feed",
        "published_at": "2026-01-01",
        "text": "body",
    }


def _search_returning(table):
    """Fake retrieval: question -> list of article ids, in rank order. Records
    the keyword arguments so tests can check what eval asked for."""
    calls = []

    def fake_search(question, k=5, **kwargs):
        calls.append(kwargs)
        return [_chunk(a) for a in table[question]][:k]

    return fake_search, calls


# --- question loading --------------------------------------------------------


def test_load_questions(tmp_path):
    path = _questions_file(tmp_path, {"question": "What happened?", "expected_article_id": "a1"})
    assert load_questions(path) == [{"question": "What happened?", "expected_article_id": "a1"}]


def test_load_questions_refuses_the_unfilled_template(tmp_path):
    """Scoring the shipped template prints a real-looking 0% that says nothing
    about retrieval, which is worse than printing nothing."""
    path = _questions_file(
        tmp_path,
        {"question": "real question", "expected_article_id": "a1"},
        {"question": "template question", "expected_article_id": "REPLACE_WITH_REAL_ARTICLE_ID"},
    )
    with pytest.raises(UnfilledQuestionSet, match="1 of 2 questions"):
        load_questions(path)


def test_the_shipped_template_is_still_recognised_as_unfilled():
    """If the placeholder id in the template is ever renamed without updating
    run.py, the guard silently stops guarding."""
    with pytest.raises(UnfilledQuestionSet):
        load_questions("eval/questions.template.yaml")


def test_every_committed_question_points_at_an_article_in_the_frozen_corpus():
    """A question whose expected article isn't in the corpus can never score,
    so it would read as a retrieval miss. Re-snapshotting without rewriting
    the questions is the way this breaks."""
    with open(eval_module.ARTICLES_PATH) as f:
        corpus_ids = {json.loads(line)["id"] for line in f}
    questions = load_questions()

    assert len(questions) >= 20
    assert [q["expected_article_id"] for q in questions if q["expected_article_id"] not in corpus_ids] == []
    assert {q.get("kind", "lookup") for q in questions} <= {"lookup", "latest"}


def test_the_manifest_records_when_the_corpus_was_snapshotted():
    as_of = eval_module.load_as_of()
    assert as_of.tzinfo is not None
    with open(eval_module.ARTICLES_PATH) as f:
        newest = max(json.loads(line)["published_at"] for line in f)
    assert datetime.fromisoformat(newest) <= as_of


# --- scoring helpers --------------------------------------------------------


def test_first_rank_is_one_based_and_uses_the_first_chunk_of_the_article():
    chunks = [_chunk("x"), _chunk("a1"), _chunk("a1")]
    assert first_rank("a1", chunks) == 2
    assert first_rank("missing", chunks) is None


def test_summarize_reports_recall_at_1_and_k_and_mrr():
    s = summarize([1, 2, None, 4], k=5)
    assert s == {"n": 4, "recall@1": 0.25, "recall@5": 0.75, "mrr": (1 + 1 / 2 + 1 / 4) / 4}


def test_citing_any_slot_of_the_expected_article_counts():
    chunks = [_chunk("x"), _chunk("a1"), _chunk("y"), _chunk("a1")]
    assert citation_is_correct("see [4]", "a1", chunks)
    assert citation_is_correct("see [1] and [2]", "a1", chunks)
    assert not citation_is_correct("see [1][3]", "a1", chunks)
    assert not citation_is_correct("see [9]", "a1", chunks), "out-of-range citations are ignored"
    assert not citation_is_correct("no citations at all", "a1", chunks)


# --- run(): live mode --------------------------------------------------------


def test_run_reports_recall_per_kind_and_overall(tmp_path, monkeypatch, capsys):
    path = _questions_file(
        tmp_path,
        {"question": "hit at 1", "expected_article_id": "a1"},
        {"question": "hit at 3", "expected_article_id": "a2"},
        {"question": "miss", "expected_article_id": "a3"},
        {"question": "newest", "expected_article_id": "new", "kind": "latest"},
    )
    fake_search, _ = _search_returning(
        {
            "hit at 1": ["a1", "x"],
            "hit at 3": ["x", "y", "a2"],
            "miss": ["x"],
            "newest": ["old", "new"],
        }
    )
    monkeypatch.setattr(eval_module, "search", fake_search)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert eval_module.run(questions_path=path, live=True) == 0

    out = capsys.readouterr().out
    assert "[HIT @1] (lookup) hit at 1" in out
    assert "[HIT @3] (lookup) hit at 3" in out
    assert "[MISS   ] (lookup) miss" in out
    assert "[HIT @2] (latest) newest" in out
    assert "recall@5: 3/4 (75%)" in out
    lookup_row = next(line for line in out.splitlines() if line.startswith("lookup"))
    assert lookup_row.split() == ["lookup", "3", "33%", "67%", "0.44"]
    # no API key -> citation checks (and the Anthropic client) never run
    assert "citation" not in out


def test_run_exits_nonzero_below_the_recall_floor(tmp_path, monkeypatch, capsys):
    path = _questions_file(
        tmp_path,
        {"question": "hit", "expected_article_id": "a1"},
        {"question": "miss", "expected_article_id": "a2"},
    )
    fake_search, _ = _search_returning({"hit": ["a1"], "miss": ["x"]})
    monkeypatch.setattr(eval_module, "search", fake_search)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert eval_module.run(questions_path=path, live=True, min_recall=0.5) == 0
    assert eval_module.run(questions_path=path, live=True, min_recall=0.6) == 1
    assert "below the 60% floor" in capsys.readouterr().out


def test_run_exits_nonzero_on_the_unfilled_template(tmp_path, monkeypatch, capsys):
    path = _questions_file(
        tmp_path,
        {"question": "template question", "expected_article_id": "REPLACE_WITH_REAL_ARTICLE_ID"},
    )
    called = []
    monkeypatch.setattr(eval_module, "search", lambda *a, **k: called.append(1) or [])

    assert eval_module.run(questions_path=path, live=True) == 1
    out = capsys.readouterr().out
    assert "still carry the placeholder id" in out
    assert "recall@5" not in out, "no score should be reported at all"
    assert not called, "retrieval should not run against a template question set"


class FakeAnthropic:
    def __init__(self, api_key=None):
        self.api_key = api_key


def test_run_checks_citations_of_hits_when_a_key_is_set(tmp_path, monkeypatch, capsys):
    path = _questions_file(
        tmp_path,
        {"question": "correctly cited", "expected_article_id": "a1"},
        {"question": "wrongly cited", "expected_article_id": "a1"},
        {"question": "missed", "expected_article_id": "a1"},
    )
    fake_search, _ = _search_returning(
        {"correctly cited": ["x", "a1"], "wrongly cited": ["x", "a1"], "missed": ["x"]}
    )
    monkeypatch.setattr(eval_module, "search", fake_search)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    monkeypatch.setattr(eval_module, "Anthropic", FakeAnthropic)

    answers = {"correctly cited": "answer [2]", "wrongly cited": "answer [1]"}
    generated_for = []

    def fake_generate(client, question, chunks):
        assert isinstance(client, FakeAnthropic)
        generated_for.append(question)
        return answers[question]

    monkeypatch.setattr(eval_module, "generate", fake_generate)

    eval_module.run(questions_path=path, live=True)

    out = capsys.readouterr().out
    assert generated_for == ["correctly cited", "wrongly cited"], "misses are not sent to the model"
    assert "citation ok: answer [2]" in out
    assert "citation MISSING/WRONG: answer [1]" in out
    assert "citation correctness (of hits): 1/2" in out


def test_retrieval_only_never_calls_the_model(tmp_path, monkeypatch, capsys):
    path = _questions_file(tmp_path, {"question": "q", "expected_article_id": "a1"})
    fake_search, _ = _search_returning({"q": ["a1"]})
    monkeypatch.setattr(eval_module, "search", fake_search)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    monkeypatch.setattr(eval_module, "Anthropic", lambda **kw: pytest.fail("client created"))
    monkeypatch.setattr(eval_module, "generate", lambda *a: pytest.fail("model called"))

    assert eval_module.run(questions_path=path, live=True, retrieval_only=True) == 0
    assert "citation" not in capsys.readouterr().out


# --- run(): frozen-corpus mode -----------------------------------------------


def test_default_mode_indexes_the_fixture_and_scores_as_of_the_snapshot(tmp_path, monkeypatch):
    path = _questions_file(tmp_path, {"question": "q", "expected_article_id": "a1"})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"as_of": "2026-11-11T17:54:10+00:00"}))
    monkeypatch.setattr(eval_module, "MANIFEST_PATH", manifest)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    built = {}
    monkeypatch.setattr(
        eval_module, "build", lambda data_path, db_path: built.update(data=data_path, db=db_path)
    )
    fake_search, calls = _search_returning({"q": ["a1"]})
    monkeypatch.setattr(eval_module, "search", fake_search)

    assert eval_module.run(questions_path=path, half_life_hours=24.0, recency_blend=0.0) == 0

    assert built["data"] == eval_module.ARTICLES_PATH
    assert calls == [
        {
            "db_path": built["db"],
            "now": datetime(2026, 11, 11, 17, 54, 10, tzinfo=timezone.utc),
            "half_life_hours": 24.0,
            "recency_blend": 0.0,
        }
    ]


def test_live_mode_uses_the_bots_own_index_and_the_wall_clock(tmp_path, monkeypatch):
    path = _questions_file(tmp_path, {"question": "q", "expected_article_id": "a1"})
    monkeypatch.setattr(eval_module, "build", lambda **kw: pytest.fail("live mode must not rebuild"))
    fake_search, calls = _search_returning({"q": ["a1"]})
    monkeypatch.setattr(eval_module, "search", fake_search)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    eval_module.run(questions_path=path, live=True)

    assert calls[0]["db_path"] is None and calls[0]["now"] is None


def test_main_passes_cli_flags_through(monkeypatch):
    seen = {}
    monkeypatch.setattr(eval_module, "run", lambda **kw: seen.update(kw) or 0)

    assert eval_module.main(
        ["--live", "--questions", "q.yaml", "--min-recall", "0.9",
         "--half-life-hours", "inf", "--recency-blend", "0", "--retrieval-only"]
    ) == 0
    assert seen == {
        "questions_path": "q.yaml",
        "live": True,
        "min_recall": 0.9,
        "half_life_hours": float("inf"),
        "recency_blend": 0.0,
        "retrieval_only": True,
    }
