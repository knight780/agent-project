"""Mock the SDK at its boundary; no paid requests in these tests."""
from types import SimpleNamespace
from unittest.mock import Mock

from app.rag import deepseek_client


def test_default_chinese_answer_and_english_translation_roles(monkeypatch):
    monkeypatch.setattr(deepseek_client, "load_dotenv", Mock())
    monkeypatch.setenv("DEEPSEEK_API_KEY", "unit-test-key")
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(choices=[
        SimpleNamespace(message=SimpleNamespace(content="Mock answer"))])
    factory = Mock(return_value=client)
    monkeypatch.setattr(deepseek_client, "OpenAI", factory)
    assert deepseek_client.call_deepseek("question") == "Mock answer"
    messages = client.chat.completions.create.call_args.kwargs["messages"]
    assert "中文" in messages[0]["content"]
    deepseek_client.call_deepseek("translate", system_prompt="Return English only.")
    messages = client.chat.completions.create.call_args.kwargs["messages"]
    assert messages[0]["content"] == "Return English only."
    assert factory.call_args.kwargs["api_key"] == "unit-test-key"
