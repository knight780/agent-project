"""Offline orchestration and actual Streamlit UI tests; never call DeepSeek."""
from io import BytesIO
from pathlib import Path
from threading import RLock
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document
from streamlit.testing.v1 import AppTest

from app import web_service as service
from app.rag.retrieval import Hit

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tmp_path(tmp_path_factory):
    # Keep Windows temp paths short enough for hash directories + Chroma files.
    return tmp_path_factory.mktemp("web")


@pytest.fixture(autouse=True)
def api(monkeypatch):
    mock = Mock(return_value="Multi-Head Attention 将多个注意力头的结果拼接并投影。[S1]")
    monkeypatch.setattr(service, "call_deepseek", mock)
    return mock


@pytest.fixture
def encoder():
    model = Mock(model_name="web-test-encoder")
    model.embed_documents.side_effect = lambda texts: [[1.0, 0.0, 0.0] for _ in texts]
    model.embed_query.return_value = [1.0, 0.0, 0.0]
    return model


@pytest.fixture
def paper():
    chunks = [Document(page_content="Multi-Head Attention concatenates heads.",
                       metadata={"source": "paper.pdf", "page": 3, "page_number": 4})]
    retriever = Mock()
    retriever.search.return_value = [Hit(7, chunks[0], "Attention", 0.8, 0.2)]
    return service.Paper("hash-a", 15, 1, retriever)


@pytest.mark.parametrize("content", [b"", b"not a PDF", b"%PDF-" + b"x" * service.MAX_UPLOAD_BYTES],
                         ids=["empty", "wrong-content", "oversize"])
def test_invalid_upload(content):
    with pytest.raises(service.WebError):
        service.validate_upload(content)


def test_filename_is_display_only():
    assert service.safe_filename("../../private\\sample.pdf") == "sample.pdf"
    assert service.safe_filename("\x00\n") == "paper.pdf"


def test_sample_real_loader_splitter_and_persistent_index_reuse(tmp_path, encoder):
    content = (ROOT / "data/sample.pdf").read_bytes()
    resource = lambda: (encoder, RLock())
    first = service.prepare_paper(content, tmp_path, resource)
    count = first.retriever.collection.count()
    assert first.page_count == 15
    assert count == first.chunk_count > 0
    encoder.embed_documents.assert_called_once()
    # New Paper/retriever simulates a fresh session with the same upload.
    second = service.prepare_paper(content, tmp_path, resource)
    assert second.paper_id == first.paper_id
    assert second.retriever.collection.name == first.retriever.collection.name
    assert second.retriever.collection.count() == count
    encoder.embed_documents.assert_called_once()
    assert (tmp_path / first.paper_id / "paper.pdf").read_bytes() == content
    assert all(doc.metadata["source"] == "paper.pdf" for doc in second.retriever.chunks)


def test_different_papers_have_isolated_stores(tmp_path, encoder, monkeypatch):
    monkeypatch.setattr(service, "load_pdf", lambda path: [Document(
        page_content="Attention definition", metadata={"page": 0, "page_number": 1})])
    first = service.prepare_paper(b"%PDF-paper-a", tmp_path, lambda: (encoder, RLock()))
    second = service.prepare_paper(b"%PDF-paper-b", tmp_path, lambda: (encoder, RLock()))
    assert first.paper_id != second.paper_id
    assert len(list(tmp_path.glob("*/chroma/chroma.sqlite3"))) == 2
    assert first.retriever.collection.count() == second.retriever.collection.count() == 1


def test_corrupt_pdf_is_safe_and_does_not_load_model(tmp_path):
    resource = Mock()
    with pytest.raises(service.WebError, match="PDF 解析失败") as error:
        service.prepare_paper(b"%PDF-broken", tmp_path, resource)
    assert str(tmp_path) not in str(error.value)
    resource.assert_not_called()


def test_chinese_answer_and_true_sources(paper, encoder, api):
    result = service.answer_question(paper, "What is Multi-Head Attention?", encoder, "../../sample.pdf")
    assert "拼接" in result.answer
    assert result.sources == [{"evidence_label": "[S1]", "filename": "sample.pdf",
                               "page_number": 4, "chunk_id": "chunk_7",
                               "page_content": "Multi-Head Attention concatenates heads."}]
    assert paper.retriever.search.call_args.kwargs == {"hybrid": False, "k": 5}
    assert "请使用中文作答" in api.call_args.args[0]
    service.answer_question(paper, "attention", encoder, "sample.pdf", hybrid=True)
    assert paper.retriever.search.call_args.kwargs["hybrid"] is True


def test_only_cited_sources_are_displayed(paper, encoder, api):
    hit = paper.retriever.search.return_value[0]
    second = Document(page_content="Second cited chunk text.", metadata=hit.document.metadata)
    paper.retriever.search.return_value.append(Hit(9, second, "Attention", 0.7, 0.3))
    api.return_value = "解释 [S2] [S2]"
    result = service.answer_question(paper, "attention", encoder, "sample.pdf")
    assert [source["chunk_id"] for source in result.sources] == ["chunk_9"]
    assert result.sources[0]["page_content"] == "Second cited chunk text."


@pytest.mark.parametrize("answer", ["无法从论文中确认作者月薪。", "论文未提及相关信息。",
                                   "证据不足。", "无引用答案", "错误引用 [S99]", "混合 [S1] [S99]"])
def test_refusal_or_invalid_citations(paper, encoder, api, answer):
    api.return_value = answer
    result = service.answer_question(paper, "What is the monthly salary of the authors?", encoder, "sample.pdf")
    assert "无法确认" in result.answer
    assert result.sources == []


def test_empty_retrieval_skips_api(paper, encoder, api):
    paper.retriever.search.return_value = []
    assert "未检索到" in service.answer_question(paper, "unknown", encoder, "sample.pdf").answer
    api.assert_not_called()


def test_api_failure_never_exposes_secret_or_path(paper, encoder, api):
    api.side_effect = RuntimeError("secret-api-key C:/private/chroma traceback")
    with pytest.raises(service.WebError, match="DeepSeek 服务调用失败") as error:
        service.answer_question(paper, "attention", encoder, "sample.pdf")
    assert "secret" not in str(error.value)
    assert "chroma" not in str(error.value)


def test_retrieval_failure_is_safe(paper, encoder, api):
    paper.retriever.search.side_effect = RuntimeError("internal db path")
    with pytest.raises(service.WebError, match="检索失败"):
        service.answer_question(paper, "attention", encoder, "sample.pdf")
    api.assert_not_called()


def test_paper_switch_and_session_isolation():
    alice = {"paper_id": "a", "messages": ["old"]}
    bob = {"paper_id": "b", "messages": ["private"]}
    assert not service.select_paper(alice, "a")
    assert alice["messages"] == ["old"]
    assert service.select_paper(alice, "b")
    assert alice["messages"] == []
    assert bob["messages"] == ["private"]
    service.select_paper(alice, None)
    assert alice["paper_id"] is None


def test_streamlit_empty_page():
    app = AppTest.from_file(str(ROOT / "streamlit_app.py")).run()
    assert not app.exception
    assert app.chat_input[0].disabled
    assert app.radio[0].value == "纯向量（基线）"


def test_streamlit_upload_chat_rerun_switch(monkeypatch, paper, encoder, api):
    import streamlit
    import streamlit_app

    upload = BytesIO(b"%PDF-test-a")
    upload.name = "../../sample.pdf"
    upload.size = len(upload.getvalue())
    monkeypatch.setattr(streamlit, "file_uploader", lambda *args, **kwargs: upload)
    prepare = Mock(return_value=paper)
    monkeypatch.setattr(streamlit_app, "prepare_paper", prepare)
    monkeypatch.setattr(streamlit_app, "embedding_resource", lambda: (encoder, RLock()))
    # Import the existing module, so patches stay in effect under AppTest.
    app = AppTest.from_string("import streamlit_app\nstreamlit_app.main()").run()
    assert not app.exception
    assert "15 页" in app.success[0].value
    app.chat_input[0].set_value("What is Multi-Head Attention?").run()
    assert not app.exception
    assert len(app.chat_message) == 2
    assert "[S1]" in app.markdown[-1].value
    assert any("chunk_7" in str(element.proto) for element in app.get("html"))
    app.radio[0].set_value("混合检索").run()
    assert len(app.chat_message) == 2
    prepare.assert_called_once()
    next(button for button in app.button if button.label == "清空聊天记录").click().run()
    assert not app.exception
    assert len(app.chat_message) == 0
    assert not app.chat_input[0].disabled
    prepare.assert_called_once()
    # A changed PDF clears the old conversation; a failed new PDF cannot fall back to it.
    upload = BytesIO(b"%PDF-test-b")
    upload.name, upload.size = "other.pdf", len(upload.getvalue())
    app.run()
    assert not app.exception
    assert len(app.chat_message) == 0
    assert prepare.call_count == 2
    upload = BytesIO(b"%PDF-test-c")
    upload.name, upload.size = "broken.pdf", len(upload.getvalue())
    prepare.side_effect = service.WebError("PDF 解析失败")
    app.run()
    assert not app.exception
    assert app.chat_input[0].disabled
    assert app.error[0].value == "PDF 解析失败"
    upload = None
    app.run()
    assert app.chat_input[0].disabled


def test_embedding_resource_is_cached(monkeypatch):
    import streamlit_app
    factory = Mock()
    monkeypatch.setattr(streamlit_app, "LocalEmbeddings", factory)
    streamlit_app.embedding_resource.clear()
    try:
        first = streamlit_app.embedding_resource()
        second = streamlit_app.embedding_resource()
        assert first is second
        factory.assert_called_once()
    finally:
        streamlit_app.embedding_resource.clear()


def test_chinese_query_changes_retrieval_only(monkeypatch, paper, encoder, api):
    translator = Mock(return_value="Why did the authors choose sinusoidal positional encoding?")
    monkeypatch.setattr("app.query_translation.translate_query_to_english", translator)
    question = "论文为什么选择正弦位置编码？"
    service.answer_question(paper, question, encoder, "sample.pdf", translate_chinese=True)
    translator.assert_called_once_with(question)
    encoder.embed_query.assert_called_once_with(translator.return_value)
    assert paper.retriever.search.call_args.args[0] == translator.return_value
    assert f"问题: {question}" in api.call_args.args[0]


def test_baseline_does_not_translate(monkeypatch, paper, encoder, api):
    translator = Mock(side_effect=AssertionError("No translation in baseline"))
    monkeypatch.setattr("app.query_translation.translate_query_to_english", translator)
    question = "论文为什么选择正弦位置编码？"
    service.answer_question(paper, question, encoder, "sample.pdf", translate_chinese=False)
    encoder.embed_query.assert_called_once_with(question)
    translator.assert_not_called()


def test_translation_failure_does_not_silently_use_wrong_query(monkeypatch, paper, encoder, api):
    monkeypatch.setattr("app.query_translation.translate_query_to_english",
                        Mock(side_effect=RuntimeError("secret/internal/path")))
    with pytest.raises(service.WebError, match="查询翻译失败") as error:
        service.answer_question(paper, "中文问题", encoder, "sample.pdf", translate_chinese=True)
    assert "secret" not in str(error.value)
    encoder.embed_query.assert_not_called()
    api.assert_not_called()


def test_supported_answer_with_later_limitation_is_not_whole_refusal(paper, encoder, api):
    # This valid answer was rejected by the old unanchored "未提供" rule.
    api.return_value = "多头注意力会拼接各个头的结果并投影。[S1] 论文未提供实现代码。"
    result = service.answer_question(paper, "How are heads combined?", encoder, "sample.pdf")
    assert result.status == "answered"
    assert len(result.sources) == 1


@pytest.mark.parametrize("answer", ["论文未提供作者月薪。[S1]", "根据提供的论文证据，无法确认相关信息。[S1]",
                                   "证据不足，无法回答。[S1]"])
def test_explicit_refusal_still_rejected_even_with_citation(paper, encoder, api, answer):
    api.return_value = answer
    result = service.answer_question(paper, "salary", encoder, "sample.pdf")
    assert result.status == "model_refusal"
    assert result.sources == []


@pytest.mark.parametrize("label", ["[S0]", "[S01]", "[s1]", "[S 1]", "[S2]"])
def test_malformed_or_unknown_citations_fail_closed(paper, encoder, api, label):
    api.return_value = f"回答 [S1] {label}"
    result = service.answer_question(paper, "attention", encoder, "sample.pdf")
    assert result.status == "invalid_citations"
    assert result.sources == []


def test_evidence_display_preserves_ids_and_escapes_filename():
    from app.web_display import citation_markdown, evidence_html
    text = "**结论** [S2] 与 [S4]"
    assert citation_markdown(text) == "**结论** `[S2]` 与 `[S4]`"
    assert citation_markdown(citation_markdown(text)) == citation_markdown(text)
    markup = evidence_html({"evidence_label": "[S4]", "filename": '<script>alert(1)</script>.pdf',
                            "page_number": 6, "chunk_id": "chunk_36"})
    assert 'translate="no"' in markup and 'class="notranslate"' in markup
    assert "[S4]" in markup and "chunk_36" in markup and "第 6 页" in markup
    assert "<script>" not in markup and "&lt;script&gt;" in markup


def test_source_numbering_and_page_mapping():
    from app.rag.generator import build_context
    document = Document(page_content="Evidence", metadata={"source": "paper.pdf", "page_number": 6})
    _, sources = build_context([Hit(i + 20, document, "Section", 1, 0) for i in range(12)])
    assert [source["evidence_label"] for source in sources] == [f"[S{i}]" for i in range(1, 13)]
    assert sources[3]["chunk_id"] == "chunk_23"
    assert sources[3]["page_number"] == 6
