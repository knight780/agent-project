from app.rag.pdf_loader import load_pdf

documents = load_pdf("data/sample.pdf")

print("文档数量:", len(documents))
print("第一页正文:", documents[0].page_content[:200])
print("第一页元数据:", documents[0].metadata)