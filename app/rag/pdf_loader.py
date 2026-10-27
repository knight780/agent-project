from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from langchain_core.documents import Document
# 1. 创建 PDF 加载器
# 2. 加载文档
def load_pdf(pdf_path: str | Path) -> list[Document]:
    # 1. 将输入路径转换为 Path 对象
    pdf_path = Path(pdf_path)

    # 2. 检查文件是否存在
    if not pdf_path.exists():
        raise FileNotFoundError(f"文件不存在: {pdf_path}")

    # 3. 检查路径是否指向文件
    if not pdf_path.is_file():
        raise ValueError(f"路径不是文件: {pdf_path}")

    # 4. 检查文件扩展名
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"文件不是 PDF 格式: {pdf_path}")

    # 5. 加载 PDF
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    # 6. 检查加载结果
    if not documents:
        raise ValueError(f"未加载到任何文档: {pdf_path}")

    # 检查是否至少有一页提取到了正文
    if not any(doc.page_content.strip() for doc in documents):
        raise ValueError("PDF 中没有提取到可用文本，可能需要 OCR")

    # 7. 补充元数据
    for i, doc in enumerate(documents):
        doc.metadata.setdefault("source", str(pdf_path.resolve()))
        doc.metadata.setdefault("page", i)
        doc.metadata.setdefault("page_number", i + 1)

    # 8. 返回 Document 列表
    return documents