"""Offline evidence viewer tests with real PDF rendering and Streamlit clicks."""
from io import BytesIO
from pathlib import Path
from threading import RLock
from unittest.mock import Mock

import pymupdf
import pytest
from streamlit.testing.v1 import AppTest

from app.web_evidence import EvidenceError, clear_chat, render_pdf_page, resolve_evidence
from app.web_service import Answer, select_paper, validate_upload


@pytest.fixture
def stored_pdf(tmp_path):
    with pymupdf.open() as pdf:
        for text in ("First page", "Second page"):
            page = pdf.new_page(width=300, height=400)
            page.insert_text((30, 50), text)
        content = pdf.tobytes()
    paper_id = validate_upload(content)
    directory = tmp_path / paper_id
    directory.mkdir()
    (directory / "paper.pdf").write_bytes(content)
    return tmp_path, paper_id, content


def source(label="[S2]", page=2, text="Second page evidence"):
    return {"evidence_label": label, "filename": "sample.pdf", "page_number": page,
            "chunk_id": "chunk_9", "page_content": text}


def message(paper_id, sources):
    return {"role": "assistant", "paper_id": paper_id, "sources": sources}


def test_real_pdf_one_based_physical_page(stored_pdf, monkeypatch):
    root, paper_id, _ = stored_pdf
    original = pymupdf.Document.load_page
    pages = []

    def load_page(pdf, index):
        pages.append(index)
        return original(pdf, index)

    monkeypatch.setattr(pymupdf.Document, "load_page", load_page)
    assert render_pdf_page(root, paper_id, 2).startswith(b"\x89PNG\r\n\x1a\n")
    assert 1 in pages


@pytest.mark.parametrize("page", [0, -1, True, "2", None, 3])
def test_invalid_and_out_of_bounds_pages(stored_pdf, page):
    root, paper_id, _ = stored_pdf
    with pytest.raises(EvidenceError, match="页码"):
        render_pdf_page(root, paper_id, page)


def test_missing_corrupt_and_encrypted_pdf(stored_pdf):
    root, paper_id, _ = stored_pdf
    path = root / paper_id / "paper.pdf"
    path.unlink()
    with pytest.raises(EvidenceError, match="缺失"):
        render_pdf_page(root, paper_id, 1)
    path.write_bytes(b"%PDF-broken")
    with pytest.raises(EvidenceError, match="损坏") as error:
        render_pdf_page(root, paper_id, 1)
    assert str(root) not in str(error.value)
    with pymupdf.open() as pdf:
        pdf.new_page()
        content = pdf.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256,
                             owner_pw="owner", user_pw="reader")
    path.write_bytes(content)
    with pytest.raises(EvidenceError, match="加密"):
        render_pdf_page(root, paper_id, 1)


@pytest.mark.parametrize("paper_id", ["../private", "a" * 63, "C:/secret", "A" * 64, None])
def test_untrusted_paper_paths_rejected(tmp_path, paper_id):
    with pytest.raises(EvidenceError, match="标识无效"):
        render_pdf_page(tmp_path, paper_id, 1)


def test_redirected_path_rejected(stored_pdf, monkeypatch):
    root, paper_id, _ = stored_pdf
    original = Path.resolve

    def redirected(path, *args, **kwargs):
        if path.name == paper_id:
            return root.parent / "private"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", redirected)
    with pytest.raises(EvidenceError, match="路径无效"):
        render_pdf_page(root, paper_id, 1)


def test_answer_scoped_mapping():
    first = message("paper", [source("[S1]", 1, "first answer")])
    second_source = source("[S1]", 2, "second answer")
    second_source["filename"] = "../../private.pdf"
    second_source["source"] = "C:/secret.pdf"
    second = message("paper", [second_source])
    assert resolve_evidence(first, "[S1]", "paper")["page_content"] == "first answer"
    assert resolve_evidence(second, "[S1]", "paper")["page_content"] == "second answer"
    with pytest.raises(EvidenceError, match="当前论文"):
        resolve_evidence(first, "[S1]", "other")


@pytest.mark.parametrize("label", ["[S1]", "[S0]", "[s2]", "[S99]", None])
def test_invalid_or_absent_evidence(label):
    with pytest.raises(EvidenceError):
        resolve_evidence(message("paper", [source()]), label, "paper")


def test_ambiguous_and_legacy_sources():
    with pytest.raises(EvidenceError, match="唯一"):
        resolve_evidence(message("paper", [source(), source()]), "[S2]", "paper")
    legacy = source()
    del legacy["page_content"]
    with pytest.raises(EvidenceError, match="未保存"):
        resolve_evidence(message("paper", [legacy]), "[S2]", "paper")


def test_selection_lifecycle():
    state = {"paper_id": "a", "messages": ["old"], "selected_evidence": {"label": "[S2]"},
             "evidence_button_old": True}
    assert not select_paper(state, "a")
    assert "selected_evidence" in state
    assert select_paper(state, "b")
    assert state == {"paper_id": "b", "messages": []}
    state.update(messages=["new"], selected_evidence={})
    clear_chat(state)
    assert state == {"paper_id": "b", "messages": []}


def test_streamlit_click_rerun_isolation_clear_switch_and_refresh(stored_pdf, monkeypatch):
    import streamlit
    import streamlit_app

    root, paper_id, content = stored_pdf
    upload = BytesIO(content)
    upload.name, upload.size = "sample.pdf", len(content)
    monkeypatch.setattr(streamlit, "file_uploader", lambda *a, **kw: upload)
    monkeypatch.setattr(streamlit_app, "DATA_ROOT", root)
    monkeypatch.setattr(streamlit_app, "prepare_paper", Mock(return_value=Mock(
        paper_id=paper_id, page_count=2, chunk_count=2)))
    monkeypatch.setattr(streamlit_app, "embedding_resource", lambda: (Mock(), RLock()))
    answer = Mock(side_effect=[
        Answer("First [S1] [S2]", [source("[S1]", 1, "first S1"), source()]),
        Answer("Second [S1]", [source("[S1]", 2, "second S1")])])
    monkeypatch.setattr(streamlit_app, "answer_question", answer)
    monkeypatch.setattr("app.web_service.call_deepseek", Mock(side_effect=AssertionError("No API")))
    app = AppTest.from_string("import streamlit_app\nstreamlit_app.main()").run()
    app.chat_input[0].set_value("first question").run()
    next(button for button in app.button if button.label == "[S2]").click().run()
    assert not app.exception
    assert any(text.value == "Second page evidence" for text in app.text)
    assert len(app.get("image")) == 1
    assert any("第 2 页" in str(item.proto) for item in app.get("html"))
    app.run()
    assert any(text.value == "Second page evidence" for text in app.text)
    app.chat_input[0].set_value("second question").run()
    [button for button in app.button if button.label == "[S1]"][1].click().run()
    assert [text.value for text in app.text if "S1" in text.value] == ["second S1"]
    next(button for button in app.button if button.label == "[S1]").click().run()
    assert [text.value for text in app.text if "S1" in text.value] == ["first S1"]
    selection = dict(app.session_state["selected_evidence"])
    selection["label"] = "[S99]"
    app.session_state["selected_evidence"] = selection
    app.run()
    assert app.warning and not app.exception
    selection["label"] = "[S1]"
    app.session_state["selected_evidence"] = selection
    messages = app.session_state["messages"]
    messages[1]["sources"][0]["page_number"] = 3
    app.session_state["messages"] = messages
    app.run()
    assert "页码越界" in app.warning[0].value and not app.exception
    next(button for button in app.button if button.label == "清空聊天记录").click().run()
    assert not app.chat_message and not app.get("image")
    assert "selected_evidence" not in app.session_state
    answer.side_effect = None
    answer.return_value = Answer("New [S2]", [source()])
    app.chat_input[0].set_value("new question").run()
    next(button for button in app.button if button.label == "[S2]").click().run()
    upload = BytesIO(content + b"\n")
    upload.name, upload.size = "other.pdf", len(upload.getvalue())
    app.run()
    assert not app.exception and not app.chat_message and not app.get("image")
    assert "selected_evidence" not in app.session_state
    fresh = AppTest.from_string("import streamlit_app\nstreamlit_app.main()").run()
    assert not fresh.exception and not fresh.chat_message
    assert "selected_evidence" not in fresh.session_state
