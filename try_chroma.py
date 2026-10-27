import chromadb

# 1. 创建本地持久化数据库
client = chromadb.PersistentClient(
    path="./data/chroma_demo"
)

# 2. 获取或创建 Collection
collection = client.get_or_create_collection(
    name="paper_demo"
)

# 3. 添加三条模拟文本及其向量
# collection.upsert(
#     ids=["chunk_1", "chunk_2", "chunk_3"],
#     documents=[
#         "Transformer uses attention mechanisms.",
#         "Convolutional neural networks process images.",
#         "Gradient descent optimizes model parameters."
#     ],
#     embeddings=[
#         [1.0, 0.0, 0.0],
#         [0.0, 1.0, 0.0],
#         [0.0, 0.0, 1.0]
#     ],
#     metadatas=[
#         {"page": 1, "source": "paper.pdf"},
#         {"page": 2, "source": "paper.pdf"},
#         {"page": 3, "source": "paper.pdf"}
#     ]
# )

print("数据库记录数量:", collection.count())

# 4. 使用模拟问题向量检索
results = collection.query(
    query_embeddings=[[0.1, 0.1, 0.0]],
    n_results=2,
    include=["documents", "metadatas", "distances"]
)

# 5. 输出检索结果
for i, text in enumerate(results["documents"][0]):
    print("文本:", text)
    print("页码:", results["metadatas"][0][i]["page"])
    print("距离:", results["distances"][0][i])
    print("---")
