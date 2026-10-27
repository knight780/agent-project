from app.rag.pdf_loader import load_pdf
from app.rag.splitter import split_documents

documents = load_pdf("data/sample.pdf")
chunks = split_documents(documents)
documents = load_pdf("data/sample.pdf")
# 第15页的原始文本
print(documents[14].page_content[:1500])
print("原始 Document 数量:", len(documents))
print("切分后 Chunk 数量:", len(chunks))

print("第26个 Chunk 长度:", len(chunks[25].page_content))
print("第26个 Chunk 内容:", chunks[25].page_content)
print("第26个 Chunk 元数据:", chunks[25].metadata)

assert len(chunks[25].page_content) <= 800
assert "page" in chunks[25].metadata
assert "source" in chunks[25].metadata

print("基础切分验证通过")