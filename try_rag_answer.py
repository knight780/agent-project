
import chromadb

from app.rag.pdf_loader import load_pdf
from app.rag.splitter import split_documents
from app.rag.embedding import LocalEmbeddings
from app.rag.retrieval import PaperRetriever, index_chunks
from app.rag.generator import build_context, build_prompt
from app.rag.deepseek_client import call_deepseek
import re
from pathlib import Path


# 1. 加载 PDF
documents = load_pdf("data/sample.pdf")

# 2. 切分文本
chunks = split_documents(
    documents,
    chunk_size=600,
    chunk_overlap=120
)

# 3. 初始化 Embedding
encoder = LocalEmbeddings()

# 4. 打开 Chroma 数据库
client = chromadb.PersistentClient(
    path="./data/chroma_real"
)

collection = index_chunks(client, chunks, encoder)

# 5. 创建检索器
retriever = PaperRetriever(collection, chunks)

# 6. 用户问题
question = "What is Multi-Head Attention?"
question1 ="Why does Transformer use positional encoding?"
question2 = " What is the monthly salary of the authors?"
# 7. 查询向量化
query_vector = encoder.embed_query(question)
query_vector1 = encoder.embed_query(question1)
query_vector2 = encoder.embed_query(question2)
# 8. 检索 Top-5
hits = retriever.search(
    question,
    query_vector,
    hybrid=False,
    k=5
)
hits1 = retriever.search(
    question1,
    query_vector1,
    hybrid=False,
    k=5
)
hits2 = retriever.search(
    question2,
    query_vector2,
    hybrid=False,
    k=5
)
# 9. 构建证据
context, sources = build_context(hits)
context1, sources1 = build_context(hits1)
context2, sources2 = build_context(hits2)

# 10. 生成 Prompt
prompt = build_prompt(question, context)
prompt1 = build_prompt(question1, context1)
prompt2 = build_prompt(question2, context2)

# 11. 调用 DeepSeek
answer = call_deepseek(prompt)
answer1 = call_deepseek(prompt1)
answer2 = call_deepseek(prompt2)

# 12. 展示结果
def print_rag_result(question, answer, sources, *, case_number=None):
    separator = "─" * 72
    title = "RAG 问答结果" if case_number is None else f"RAG 测试 {case_number:02d}"
    print(f"\n{'═' * 72}\n  {title}\n{'═' * 72}")
    print(f"\n【用户问题】\n{question.strip()}")
    print(f"\n{separator}\n【AI 回答】\n\n{answer.strip()}")

    # 只展示实际被答案引用的来源，按 S1、S2 等编号排序。
    used_refs = set(re.findall(r"\[S\d+\]", answer))
    source_map = {
        source["evidence_label"]: source
        for source in sources
    }
    invalid_refs = used_refs - set(source_map)
    valid_refs = sorted(used_refs & set(source_map), key=lambda label: int(label[2:-1]))
    print(f"\n{separator}\n【实际引用来源 · {len(valid_refs)} 条】")

    for label in valid_refs:
        source = source_map[label]
        print(
            f"\n  {label}  {Path(source['source']).name}  ·  第 {source['page_number']} 页\n"
            f"        Chunk：{source['chunk_id']}\n"
            f"        路径：{source['source']}"
        )
    if not valid_refs:
        print("\n  暂无可展示的已匹配来源。")

    print(f"\n{separator}\n【引用编号检查】")
    print(f"  检索来源 {len(sources)} 条  |  已匹配引用 {len(valid_refs)} 条  |  无效引用 {len(invalid_refs)} 条")
    if invalid_refs:
        labels = sorted(invalid_refs, key=lambda label: int(label[2:-1]))
        print(f"  未匹配到本次来源的编号：{'、'.join(labels)}")
    elif used_refs:
        print("  所有引用编号均能匹配本次检索来源。")
    if not used_refs:
        print("  本次回答未使用引用；若为证据不足的拒答，此情况可能正常。")
    print(f"\n{'═' * 72}\n")


print_rag_result(question, answer, sources, case_number=1)
print_rag_result(question1, answer1, sources1, case_number=2)
print_rag_result(question2, answer2, sources2, case_number=3)
