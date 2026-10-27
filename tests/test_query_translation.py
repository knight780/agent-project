from unittest.mock import Mock

import pytest

from app import query_translation as translation


@pytest.fixture(autouse=True)
def api(monkeypatch):
    mock = Mock(return_value="Why did the authors choose sinusoidal positional encoding?")
    monkeypatch.setattr(translation, "call_deepseek", mock)
    return mock


def test_translate_chinese_with_real_prompt_and_mock_cloud(api):
    original = "论文为什么选择正弦位置编码？"
    result = translation.retrieval_query(original, True)
    assert result == api.return_value
    assert original in api.call_args.args[0]
    assert "不回答问题" in api.call_args.args[0]
    assert "English" in api.call_args.kwargs["system_prompt"]


@pytest.mark.parametrize("question,enabled", [("English question?", True), ("中文问题", False)])
def test_skip_translation(question, enabled, api):
    assert translation.retrieval_query(question, enabled) == question
    api.assert_not_called()


@pytest.mark.parametrize("response", ["", "翻译失败", "First line\nSecond line", "123", "x" * 1001])
def test_invalid_translation_is_not_used(response, api):
    api.return_value = response
    with pytest.raises(ValueError):
        translation.retrieval_query("中文问题", True)
