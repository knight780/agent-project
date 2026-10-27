# 本地 Embedding 验证

本阶段将 Chunk 正文转换为向量，并验证查询与文档使用相同的编码方式。
暂不创建 Chroma 数据库，也不调用 LLM。

## 运行

在项目根目录的 PowerShell 中运行：

```powershell
.\.venv\Scripts\python.exe try_embedding.py
.\.venv\Scripts\python.exe -m pytest -q
```

运行依赖为 `sentence-transformers`，已加入 `requirements.txt`。
模型为本机已缓存的 `sentence-transformers/all-MiniLM-L6-v2`。
`LocalEmbeddings` 设置 `local_files_only=True`，模型缺失时会报错，不会自动下载。
新机器需要提前准备完整的模型缓存，或把本地模型目录传给 `LocalEmbeddings`。

## 输入与输出

```python
from app.rag.embedding import LocalEmbeddings

encoder = LocalEmbeddings()
vectors = encoder.embed_documents([chunk.page_content for chunk in chunks])
query_vector = encoder.embed_query("How does attention work?")
```

- 每个 Chunk 对应一个向量，返回顺序与输入一致。
- 该模型输出 384 维向量；文档与查询都做 L2 归一化。
- 原始 `chunks` 继续保留正文和 metadata，向量不会替代 Document。
- 空批次返回空列表；空白文本与超长文本抛出 `ValueError`。

## 为什么演示脚本使用 600 字符

现有 Splitter 的默认值仍为 800 字符、120 字符重叠。
本地模型配置的输入上限为 256 token（包含特殊 token）。
这份 sample.pdf 按 800 字符切分产生 66 块，其中 3 块超过模型上限；
按 600 字符、120 字符重叠切分产生 89 块，最长为 225 token。
因此新脚本显式使用 600，避免模型静默截断正文。
字符与 token 不等价，更换 PDF 后仍会逐块检查真实 token 数量。

可传参数复现其他配置：

```powershell
.\.venv\Scripts\python.exe try_embedding.py --chunk-size 800
```

对于当前 sample.pdf，此命令应报超长输入错误，而不是悄悄丢弃正文。

## 如何验收

脚本检查向量数量、维度、有限数值和归一化结果，并打印英文问题最相近的三个
Chunk 及原文、来源、页码。这里直接做向量点积，供小样本检查；下一步再接
Chroma 持久化存储。相似度分数不是回答正确的概率，需要人工核对原文。

当前模型用于英文论文流程验证；中文问题检索英文论文的效果需要另行评测。
更换模型后需要重新生成整套文档向量。
