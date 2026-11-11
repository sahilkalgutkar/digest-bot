import pytest

import generate.chat as chat_module
from generate.chat import answer


class FakeTextBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class FakeThinkingBlock:
    """Shaped like the SDK's ThinkingBlock: no .text attribute at all."""

    type = "thinking"
    thinking = ""


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeMessages:
    def __init__(self, content):
        self._content = content
        self.last_call = None

    def create(self, **kwargs):
        self.last_call = kwargs
        return FakeResponse(self._content)


class FakeClient:
    def __init__(self, response_text="answer with [1] citation", content=None):
        self.messages = FakeMessages(content or [FakeTextBlock(response_text)])


def test_answer_returns_message_when_no_chunks_found(monkeypatch):
    monkeypatch.setattr(chat_module, "search", lambda question, k=5: [])
    client = FakeClient()

    result = answer(client, "anything?")

    assert "don't have any indexed articles" in result
    assert client.messages.last_call is None  # model should never be called


def test_answer_cites_sources_and_returns_model_text(monkeypatch):
    chunks = [
        {
            "title": "A",
            "source": "feed",
            "published_at": "2026-01-01",
            "text": "hello world",
            "article_id": "a1",
        }
    ]
    monkeypatch.setattr(chat_module, "search", lambda question, k=5: chunks)
    client = FakeClient(response_text="the answer [1]")

    result = answer(client, "what happened?")

    assert result == "the answer [1]"
    call = client.messages.last_call
    assert call["system"] == chat_module.SYSTEM_PROMPT
    assert call["model"] == chat_module.MODEL
    assert "hello world" in call["messages"][0]["content"]
    assert "what happened?" in call["messages"][0]["content"]


def test_answer_skips_a_leading_thinking_block(monkeypatch):
    """With adaptive thinking the first content block can be a thinking block,
    which has no .text -- reading content[0].text used to crash here."""
    monkeypatch.setattr(chat_module, "search", lambda question, k=5: [_chunk()])
    client = FakeClient(
        content=[FakeThinkingBlock(), FakeTextBlock("part one [1]"), FakeTextBlock(" part two")]
    )

    assert answer(client, "what happened?") == "part one [1] part two"


def test_generate_answers_from_the_chunks_it_is_given(monkeypatch):
    monkeypatch.setattr(chat_module, "search", lambda *a, **k: pytest.fail("generate must not search"))
    client = FakeClient(response_text="from the chunk [1]")

    assert chat_module.generate(client, "q?", [_chunk()]) == "from the chunk [1]"
    assert "hello world" in client.messages.last_call["messages"][0]["content"]


def _chunk():
    return {
        "title": "A",
        "source": "feed",
        "published_at": "2026-01-01",
        "text": "hello world",
        "article_id": "a1",
    }


def test_main_exits_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):
        chat_module.main()


def test_main_skips_blank_input_and_answers_real_questions(monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    created_with = {}

    class FakeAnthropic:
        def __init__(self, api_key):
            created_with["api_key"] = api_key

    monkeypatch.setattr(chat_module, "Anthropic", FakeAnthropic)

    inputs = iter(["   ", "what happened?"])

    def fake_input(prompt):
        try:
            return next(inputs)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr("builtins.input", fake_input)

    answer_calls = []

    def fake_answer(client, question):
        answer_calls.append((client, question))
        return f"ANSWER: {question}"

    monkeypatch.setattr(chat_module, "answer", fake_answer)

    chat_module.main()

    # Blank input never reaches answer(); only the real question does.
    assert len(answer_calls) == 1
    assert answer_calls[0][1] == "what happened?"
    assert isinstance(answer_calls[0][0], FakeAnthropic)
    assert created_with["api_key"] == "test-key"

    out = capsys.readouterr().out
    assert "digest-bot ready" in out
    assert "ANSWER: what happened?" in out


def test_main_exits_cleanly_on_keyboard_interrupt(monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(chat_module, "Anthropic", lambda api_key: FakeClient())

    def fake_input(prompt):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", fake_input)

    answer_calls = []
    monkeypatch.setattr(
        chat_module, "answer", lambda client, question: answer_calls.append(question)
    )

    chat_module.main()  # should return normally, not raise

    assert answer_calls == []
