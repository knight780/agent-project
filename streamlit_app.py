"""Run with: python -m streamlit run streamlit_app.py."""
from pathlib import Path
from threading import RLock
import os
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

from app.rag.embedding import LocalEmbeddings
from app.web_service import (MAX_UPLOAD_BYTES, WebError, answer_question,
                             prepare_paper, safe_filename, select_paper, validate_upload)
from app.web_display import citation_markdown, evidence_html
from app.web_evidence import EvidenceError, clear_chat, render_pdf_page, resolve_evidence

DATA_ROOT = Path(__file__).resolve().parent / "data" / "web"


@st.cache_resource(show_spinner=False)
def embedding_resource():
    # Shared model only; serialize inference on the shared tokenizer.
    return LocalEmbeddings(), RLock()


def select_evidence(message_id, label, paper_id):
    st.session_state.selected_evidence = {
        "message_id": message_id, "label": label, "paper_id": paper_id}


def render_message(message):
    message.setdefault("id", uuid4().hex)
    with st.chat_message(message["role"]):
        content = citation_markdown(message["content"]) if message["role"] == "assistant" else message["content"]
        st.markdown(content)
        if message.get("sources"):
            with st.expander("本次实际引用来源", expanded=True):
                for source in message["sources"]:
                    st.html(evidence_html(source))
                    label = source["evidence_label"]
                    st.button(label, key=f"evidence_button_{message['id']}_{label}",
                              help="查看该引用的 Chunk 原文与 PDF 页面",
                              on_click=select_evidence,
                              args=(message["id"], label, st.session_state.paper_id))
            selection = st.session_state.get("selected_evidence")
            if selection and selection.get("message_id") == message["id"]:
                try:
                    if selection.get("paper_id") != st.session_state.paper_id:
                        raise EvidenceError("证据不属于当前论文，请重新选择引用。")
                    source = resolve_evidence(message, selection["label"], st.session_state.paper_id)
                    st.subheader("论文证据")
                    st.html(evidence_html(source))
                    st.text(source["page_content"])
                    page = render_pdf_page(DATA_ROOT, st.session_state.paper_id, source["page_number"])
                    st.image(page, caption=f"{source['filename']} · 第 {source['page_number']} 页")
                except EvidenceError as exc:
                    st.warning(str(exc))


def main():
    load_dotenv(Path(__file__).resolve().parent / ".env")
    st.set_page_config(page_title="AI Research Assistant", page_icon="📚", layout="wide")
    st.title("📚 AI Research Assistant")
    st.caption("基于论文证据的中文问答；专业术语可保留英文。")
    for key, value in {"messages": [], "paper_id": None, "paper": None,
                       "filename": "paper.pdf"}.items():
        st.session_state.setdefault(key, value)
    with st.sidebar:
        st.header("论文知识库")
        uploaded = st.file_uploader("上传单篇 PDF 论文", type=["pdf"], accept_multiple_files=False)
        st.caption(f"仅支持含可提取文本的 PDF，大小上限 {MAX_UPLOAD_BYTES // 1024 // 1024} MB。")
        mode = st.radio("检索模式", ["纯向量（基线）", "混合检索"])
        translate_chinese = st.checkbox(
            "中文问题先转英文检索", value=os.getenv("RAG_TRANSLATE_CHINESE_QUERY", "true").lower() == "true",
            help="保留原问题用于回答；中文问题会额外调用一次 DeepSeek 翻译。关闭可复现原始基线。")
        if uploaded is None:
            select_paper(st.session_state, None)
            st.session_state.paper = None
        else:
            try:
                if uploaded.size > MAX_UPLOAD_BYTES:
                    raise WebError("PDF 超过大小上限，请上传不超过 20 MB 的文件。")
                content = uploaded.getvalue()
                paper_id = validate_upload(content)
                changed = select_paper(st.session_state, paper_id)
                st.session_state.filename = safe_filename(uploaded.name)
                if changed:
                    st.session_state.paper = None
                if st.session_state.paper is None:
                    with st.spinner("正在读取论文并建立或复用索引…"):
                        st.session_state.paper = prepare_paper(content, DATA_ROOT, embedding_resource)
                paper = st.session_state.paper
                st.text(st.session_state.filename)
                st.success(f"知识库就绪 · {paper.page_count} 页 · {paper.chunk_count} Chunks")
            except WebError as exc:
                select_paper(st.session_state, None)
                st.session_state.paper = None
                st.error(str(exc))
            except Exception:
                select_paper(st.session_state, None)
                st.session_state.paper = None
                st.error("知识库暂时不可用，请检查本地模型和运行环境后重新上传。")
        if st.button("清空聊天记录"):
            clear_chat(st.session_state)
    for message in st.session_state.messages:
        render_message(message)
    question = st.chat_input("请输入论文问题，例如：What is Multi-Head Attention?",
                             disabled=st.session_state.paper is None)
    if question and question.strip():
        message = {"role": "user", "content": question.strip()}
        st.session_state.messages.append(message)
        render_message(message)
        try:
            with st.spinner("正在检索论文并生成回答…"):
                encoder, lock = embedding_resource()
                result = answer_question(st.session_state.paper, question.strip(), encoder,
                                         st.session_state.filename, hybrid=mode == "混合检索",
                                         embedding_lock=lock, translate_chinese=translate_chinese)
            reply = {"role": "assistant", "content": result.answer, "sources": result.sources}
        except WebError as exc:
            reply = {"role": "assistant", "content": str(exc), "sources": []}
        except Exception:
            reply = {"role": "assistant", "content": "问答暂时不可用，请稍后重试。", "sources": []}
        st.session_state.messages.append(reply)
        reply["paper_id"] = st.session_state.paper_id
        render_message(reply)


if __name__ == "__main__":
    main()
