from pathlib import Path
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app.rag import pdf_loader


def write_pdf(path: Path, pages: list[str]) -> Path:
    """Create local PDF fixtures without external files or services."""
    writer = PdfWriter()
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    for text in pages:
        page = writer.add_blank_page(width=612, height=792)
        if text:
            page[NameObject("/Resources")] = DictionaryObject(
                {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
            )
            stream = DecodedStreamObject()
            stream.set_data(
                f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("ascii")
            )
            page[NameObject("/Contents")] = writer._add_object(stream)
    with path.open("wb") as output:
        writer.write(output)
    return path


@pytest.fixture
def text_pdf(tmp_path):
    return write_pdf(tmp_path / "example.pdf", ["First page text", "Second page text"])


@pytest.mark.parametrize("as_string", [False, True], ids=["path", "string"])
def test_load_pdf_returns_documents_with_text_and_metadata(text_pdf, as_string):
    documents = pdf_loader.load_pdf(str(text_pdf) if as_string else text_pdf)

    assert isinstance(documents, list)
    assert len(documents) == 2
    for index, (document, expected) in enumerate(
        zip(documents, ["First page text", "Second page text"])
    ):
        assert isinstance(document, Document)
        assert document.page_content.strip() == expected
        assert isinstance(document.metadata, dict)
        assert document.metadata["source"] == str(text_pdf)
        assert document.metadata["page"] == index
        assert document.metadata["page_number"] == index + 1


def test_load_pdf_preserves_relative_source_from_loader(text_pdf, monkeypatch):
    monkeypatch.chdir(text_pdf.parent)

    documents = pdf_loader.load_pdf(text_pdf.name)

    assert all(document.metadata["source"] == text_pdf.name for document in documents)


def test_load_pdf_accepts_uppercase_extension(tmp_path):
    path = write_pdf(tmp_path / "example.PDF", ["Uppercase extension"])

    assert pdf_loader.load_pdf(path)[0].page_content.strip() == "Uppercase extension"


@pytest.mark.parametrize("invalid_path", ["missing.pdf", "missing.txt"])
def test_load_pdf_rejects_missing_file_before_loading(tmp_path, monkeypatch, invalid_path):
    loader = Mock()
    monkeypatch.setattr(pdf_loader, "PyPDFLoader", loader)
    path = tmp_path / invalid_path

    with pytest.raises(FileNotFoundError) as error:
        pdf_loader.load_pdf(path)

    assert str(error.value) == f"文件不存在: {path}"
    loader.assert_not_called()


@pytest.mark.parametrize("filename", ["example.txt", "example", "example.pdf.txt"])
def test_load_pdf_rejects_wrong_extension(tmp_path, monkeypatch, filename):
    path = tmp_path / filename
    path.write_text("Not a PDF", encoding="utf-8")
    loader = Mock()
    monkeypatch.setattr(pdf_loader, "PyPDFLoader", loader)

    with pytest.raises(ValueError) as error:
        pdf_loader.load_pdf(path)

    assert str(error.value) == f"文件不是 PDF 格式: {path}"
    loader.assert_not_called()


def test_load_pdf_rejects_directory(tmp_path, monkeypatch):
    path = tmp_path / "directory.pdf"
    path.mkdir()
    loader = Mock()
    monkeypatch.setattr(pdf_loader, "PyPDFLoader", loader)

    with pytest.raises(ValueError) as error:
        pdf_loader.load_pdf(path)

    assert str(error.value) == f"路径不是文件: {path}"
    loader.assert_not_called()


def test_load_pdf_rejects_real_pdf_without_extractable_text(tmp_path):
    path = write_pdf(tmp_path / "blank.pdf", ["", ""])

    with pytest.raises(ValueError, match="PDF 中没有提取到可用文本，可能需要 OCR"):
        pdf_loader.load_pdf(path)


def test_load_pdf_allows_blank_page_when_another_page_has_text(tmp_path):
    path = write_pdf(tmp_path / "mixed.pdf", ["", "Usable text"])

    documents = pdf_loader.load_pdf(path)

    assert len(documents) == 2
    assert documents[0].page_content.strip() == ""
    assert documents[1].page_content.strip() == "Usable text"
    assert [doc.metadata["page_number"] for doc in documents] == [1, 2]


def test_load_pdf_rejects_whitespace_only_content(text_pdf, monkeypatch):
    loader = Mock()
    loader.return_value.load.return_value = [
        Document(page_content=""),
        Document(page_content=" \n\t "),
    ]
    monkeypatch.setattr(pdf_loader, "PyPDFLoader", loader)

    with pytest.raises(ValueError, match="PDF 中没有提取到可用文本，可能需要 OCR"):
        pdf_loader.load_pdf(text_pdf)


def test_load_pdf_rejects_empty_loader_result(text_pdf, monkeypatch):
    loader = Mock()
    loader.return_value.load.return_value = []
    monkeypatch.setattr(pdf_loader, "PyPDFLoader", loader)

    with pytest.raises(ValueError) as error:
        pdf_loader.load_pdf(text_pdf)

    assert str(error.value) == f"未加载到任何文档: {text_pdf}"
    loader.assert_called_once_with(str(text_pdf))
    loader.return_value.load.assert_called_once_with()


def test_load_pdf_fills_missing_metadata(text_pdf, monkeypatch):
    original = [Document(page_content="First"), Document(page_content="Second")]
    loader = Mock()
    loader.return_value.load.return_value = original
    monkeypatch.setattr(pdf_loader, "PyPDFLoader", loader)
    monkeypatch.chdir(text_pdf.parent)

    documents = pdf_loader.load_pdf(text_pdf.name)

    assert documents is original
    assert [doc.metadata for doc in documents] == [
        {"source": str(text_pdf.resolve()), "page": 0, "page_number": 1},
        {"source": str(text_pdf.resolve()), "page": 1, "page_number": 2},
    ]
    loader.assert_called_once_with(text_pdf.name)
    loader.return_value.load.assert_called_once_with()


def test_load_pdf_preserves_existing_metadata(text_pdf, monkeypatch):
    metadata = {"source": "original.pdf", "page": 7, "page_number": 8, "title": "Paper"}
    document = Document(page_content="Text", metadata=metadata.copy())
    loader = Mock()
    loader.return_value.load.return_value = [document]
    monkeypatch.setattr(pdf_loader, "PyPDFLoader", loader)

    documents = pdf_loader.load_pdf(text_pdf)

    assert documents[0] is document
    assert documents[0].metadata == metadata
