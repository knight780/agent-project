"""Run the PDF -> chunks -> local embeddings smoke check, without a vector DB."""
import argparse
from pathlib import Path
import numpy as np
from app.rag.embedding import LocalEmbeddings
from app.rag.pdf_loader import load_pdf
from app.rag.splitter import split_documents
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=Path("data/sample.pdf"))
    # MiniLM's configured limit is 256 tokens. Verify actual token lengths in
    # LocalEmbeddings as even 600 characters cannot guarantee this for all PDFs.
    parser.add_argument("--chunk-size", type=int, default=600)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    args = parser.parse_args()

    documents = load_pdf(args.pdf)
    chunks = split_documents(
        documents, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
    )
    embeddings = LocalEmbeddings()
    vectors = embeddings.embed_documents([chunk.page_content for chunk in chunks])
    question = "How does scaled dot-product attention work?"
    query_vector = embeddings.embed_query(question)
    question2="What is multi-head attention?"
    query_vector2 = embeddings.embed_query(question2) 
    question3="Why is positional encoding needed?"  
    query_vector3 = embeddings.embed_query(question3) 
    matrix = np.asarray(vectors)
    assert matrix.ndim == 2 and matrix.shape[0] == len(chunks)
    assert matrix.shape[1] == len(query_vector)
    assert np.isfinite(matrix).all() and np.isfinite(query_vector).all()
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-5)

    print("模型:", embeddings.model_name)
    print("原始页数:", len(documents))
    print("切分参数:", args.chunk_size, args.chunk_overlap)
    print("Chunk 数量:", len(chunks))
    print("向量形状:", matrix.shape)
    print("查询向量维度:", len(query_vector))
    print("第一个向量的前 5 个值:", vectors[0][:5])

    # For this small sample, normalized vector dot products let us inspect
    # semantic matches before adding persistent storage in the next stage.
    scores = matrix @ np.asarray(query_vector)
    print("验证问题:", question)
    for index in np.argsort(-scores)[:3]:
        chunk = chunks[int(index)]
        print(
            f"Chunk {index}, page_number={chunk.metadata.get('page_number')}, "
            f"source={chunk.metadata.get('source')}, similarity={scores[index]:.4f}"
        )
        print(chunk.page_content)
    print("本地 Embedding 基础验证通过")
    print("验证问题:", question2)
    scores2 = matrix @ np.asarray(query_vector2)
    for index in np.argsort(-scores2)[:3]:
        chunk = chunks[int(index)]
        print(
            f"Chunk {index}, page_number={chunk.metadata.get('page_number')}, "
            f"source={chunk.metadata.get('source')}, similarity={scores2[index]:.4f}"
        )
        print(chunk.page_content)
        print("本地 Embedding 基础验证1通过")
    print("验证问题:", question3)
    scores3 = matrix @ np.asarray(query_vector3)
    for index in np.argsort(-scores3)[:3]:
        chunk = chunks[int(index)]
        print(
            f"Chunk {index}, page_number={chunk.metadata.get('page_number')}, "
            f"source={chunk.metadata.get('source')}, similarity={scores3[index]:.4f}"
        )
        print(chunk.page_content)
        print("本地 Embedding 基础验证2通过")
    print("本地 Embedding 基础验证通过")
if __name__ == "__main__":
    main()
