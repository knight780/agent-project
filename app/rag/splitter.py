from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Document]:
    """将输入的 Document 列表切分为多个 Chunk。"""

    # TODO 1：检查 documents 是否为空
    if not documents:
        raise ValueError("输入的 Document 列表为空，无法进行切分。")
    # TODO 2：检查 chunk_size 和 chunk_overlap 是否合理
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须为正整数。")
    if chunk_overlap <0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须为非负整数。")
    # TODO 3：创建 RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )   
    # TODO 4：执行切分，获得 chunks
    chunks = splitter.split_documents(documents)
    # TODO 5：检查结果并返回 chunks
    if not chunks:
        raise ValueError("切分后没有生成任何 Chunk，请检查输入的 Document 内容。") 
    return chunks
    
