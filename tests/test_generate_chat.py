import generate.chat as chat_module
from generate.chat import answer


class FakeTextBlock:
    def __init__(self, text):
        self.text = text


class FakeResponse:
    def __init__(self, text):
        self.content = [FakeTextBlock(text)]


class FakeMessages:
    def __init__(self, response_text):
        self._response_text = response_text
        self.last_call = None

    def create(self, **kwargs):
        self.last_call = kwargs
        return FakeResponse(self._response_text)


class FakeClient:
    def __init__(self, response_text="answer with [1] citation"):
        self.messages = FakeMessages(response_text)


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
