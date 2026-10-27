"""UI orchestration only; reuse the validated RAG algorithms."""
from dataclasses import dataclass
from contextlib import nullcontext
import hashlib
from pathlib import Path
import re

import chromadb
from chromadb.config import Settings
from filelock import FileLock

from app.rag.deepseek_client import call_deepseek
from app.rag.generator import build_context, build_prompt
from app.rag.pdf_loader import load_pdf
from app.rag.retrieval import PaperRetriever, index_chunks
from app.rag.splitter import split_documents
from app.query_translation import retrieval_query
from app.web_evidence import clear_chat

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
REFUSAL = "根据提供的论文证据，无法确认相关信息。请尝试询问论文中明确讨论的内容。"


class WebError(Exception):
    """Only fixed, user-safe messages may cross the UI boundary."""


@dataclass
class Paper:
    paper_id: str
    page_count: int
    chunk_count: int
    retriever: PaperRetriever


@dataclass
class Answer:
    answer: str
    sources: list[dict]
    status: str = "answered"


def safe_filename(name: str) -> str:
    name = str(name).replace("\\", "/").rsplit("/", 1)[-1]
    name = "".join(char for char in name if char.isprintable()).strip()
    return name[:180] or "paper.pdf"


def validate_upload(content: bytes) -> str:
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise WebError("PDF 为空或超过大小上限，请上传不超过 20 MB 的文件。")
    if not content.startswith(b"%PDF-"):
        raise WebError("文件内容不是有效的 PDF，请重新选择文件。")
    return hashlib.sha256(content).hexdigest()


def select_paper(state, paper_id: str | None) -> bool:
    changed = state.get("paper_id") != paper_id
    if changed:
        state["paper_id"] = paper_id
        clear_chat(state)
    return changed


def prepare_paper(content: bytes, root: Path, get_embedding) -> Paper:
    paper_id = validate_upload(content)
    directory = Path(root) / paper_id
    try:
        directory.mkdir(parents=True, exist_ok=True)
        # Serialize concurrent uploads of identical bytes across sessions/processes.
        with FileLock(str(directory / "index.lock"), timeout=120):
            pdf_path = directory / "paper.pdf"
            if not pdf_path.exists() or pdf_path.read_bytes() != content:
                temporary = directory / "paper.pdf.tmp"
                temporary.write_bytes(content)
                temporary.replace(pdf_path)
            try:
                documents = load_pdf(pdf_path)
                for document in documents:
                    document.metadata["source"] = "paper.pdf"
                chunks = split_documents(documents, chunk_size=600, chunk_overlap=120)
            except Exception:
                raise WebError("PDF 解析失败：文件可能损坏、加密或没有可提取文本（扫描件需先 OCR）。") from None
            try:
                encoder, lock = get_embedding()
                client = chromadb.PersistentClient(
                    path=str(directory / "chroma"), settings=Settings(anonymized_telemetry=False))
                with lock:
                    collection = index_chunks(client, chunks, encoder)
                return Paper(paper_id, len(documents), len(chunks), PaperRetriever(collection, chunks))
            except Exception:
                raise WebError("建立知识库失败，请检查本地 Embedding 模型缓存和存储空间后重试。") from None
    except WebError:
        raise
    except Exception:
        raise WebError("论文存储暂时不可用，请稍后重新上传。") from None


def answer_question(paper: Paper, question: str, encoder, filename: str, *, hybrid=False,
                    embedding_lock=None, translate_chinese=False) -> Answer:
    if not question.strip():
        raise WebError("请输入非空问题。")
    try:
        query = retrieval_query(question, translate_chinese)
    except Exception:
        raise WebError("中文检索查询翻译失败，请稍后重试，或关闭中文转英文检索以使用基线模式。") from None
    try:
        with embedding_lock if embedding_lock is not None else nullcontext():
            vector = encoder.embed_query(query)
        hits = paper.retriever.search(query, vector, hybrid=hybrid, k=5)
    except Exception:
        raise WebError("检索失败，请缩短问题后重试，并检查本地模型和知识库。") from None
    if not hits:
        return Answer("未检索到可用证据。" + REFUSAL, [], "empty_evidence")
    context, sources = build_context(hits)
    try:
        answer = call_deepseek(build_prompt(question, context))
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("empty answer")
    except Exception:
        raise WebError("DeepSeek 服务调用失败，请检查服务状态及 .env 中的 API 配置后重试。") from None
    # A limitation in a later sentence is not a refusal of the whole answer.
    # Keep explicit refusal detection, anchored to the opening statement.
    opening = re.sub(r"^[\s#*>`-]+", "", answer).split("。", 1)[0]
    refusal = re.search(
        r"^(?:(?:根据|基于|就|从).{0,40}?[，,:：]|(?:这篇|本篇|该|本)?论文(?:中)?|"
        r"提供的(?:论文)?证据(?:中)?|抱歉[，,]?)?\s*"
        r"(?:无法.{0,16}(?:确认|回答|确定)|(?:证据|信息)不足|未(?:提及|提供)|"
        r"没有.{0,8}(?:提及|相关信息))", opening)
    if refusal:
        return Answer(REFUSAL, [], "model_refusal")
    used = set(re.findall(r"\[S\d+\]", answer))
    available = {source["evidence_label"] for source in sources}
    # Reject malformed variants as well as nonexistent canonical labels.
    tokens = re.findall(r"\[[sS][^\]\n]*\]", answer)
    malformed = any(not re.fullmatch(r"\[S[1-9]\d*\]", token) for token in tokens)
    if malformed or not used or not used <= available:
        return Answer("回答缺少有效的论文引用，暂时无法确认。请重新提问。", [], "invalid_citations")
    selected = [
        {"evidence_label": source["evidence_label"], "filename": safe_filename(filename),
         "page_number": source["page_number"], "chunk_id": source["chunk_id"],
         "page_content": hit.document.page_content}
        for source, hit in zip(sources, hits) if source["evidence_label"] in used
    ]
    return Answer(answer, selected)
