"""Answer-scoped evidence selection and bounded PDF page rendering."""
from pathlib import Path
import re

import pymupdf


class EvidenceError(Exception):
    """Fixed, user-safe evidence viewer errors."""


def clear_chat(state):
    state["messages"] = []
    state.pop("selected_evidence", None)
    for key in list(state):
        if isinstance(key, str) and key.startswith("evidence_button_"):
            del state[key]


def resolve_evidence(message: dict, label: str, paper_id: str) -> dict:
    if message.get("role") != "assistant" or message.get("paper_id") != paper_id:
        raise EvidenceError("证据不属于当前论文或回答，请重新选择引用。")
    if not isinstance(label, str) or not re.fullmatch(r"\[S[1-9]\d*\]", label):
        raise EvidenceError("证据编号无效，请选择本回答下方的实际引用。")
    matches = [source for source in message.get("sources", [])
               if source.get("evidence_label") == label]
    if len(matches) != 1:
        raise EvidenceError("本回答中不存在唯一对应的证据编号，请重新提问。")
    source = matches[0]
    if not isinstance(source.get("page_content"), str) or not source["page_content"].strip():
        raise EvidenceError("该引用未保存 Chunk 原文，请重新提问以生成可查看的证据。")
    return source


def render_pdf_page(root: Path, paper_id: str, page_number: int) -> bytes:
    """Read only root/<SHA-256>/paper.pdf; public page numbers are one-based."""
    if not isinstance(paper_id, str) or not re.fullmatch(r"[0-9a-f]{64}", paper_id):
        raise EvidenceError("论文标识无效，请重新上传论文。")
    if type(page_number) is not int or page_number < 1:
        raise EvidenceError("引用页码无效，页码必须是从 1 开始的整数。")
    try:
        root = Path(root).resolve()
        directory = root / paper_id
        path = directory / "paper.pdf"
        # Reject redirected hash directories and files, including escaping symlinks.
        if directory.resolve() != directory or path.resolve() != path:
            raise EvidenceError("论文文件路径无效，请重新上传论文。")
        if not path.is_file():
            raise EvidenceError("论文 PDF 文件缺失，请重新上传论文。")
        with pymupdf.open(path) as pdf:
            if pdf.needs_pass:
                raise EvidenceError("论文 PDF 已加密，无法预览，请上传未加密的文件。")
            if page_number > pdf.page_count:
                raise EvidenceError(f"引用页码越界：第 {page_number} 页，PDF 共 {pdf.page_count} 页。")
            page = pdf.load_page(page_number - 1)
            # Bound pixel allocation even for PDFs with unusually large page dimensions.
            longest = max(page.rect.width, page.rect.height)
            if longest <= 0:
                raise EvidenceError("PDF 页面尺寸无效，无法预览。")
            scale = min(1.5, 1800 / longest)
            return page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False).tobytes("png")
    except EvidenceError:
        raise
    except FileNotFoundError:
        raise EvidenceError("论文 PDF 文件缺失，请重新上传论文。") from None
    except Exception:
        raise EvidenceError("PDF 页面渲染失败：文件可能损坏或无法读取，请重新上传论文。") from None
