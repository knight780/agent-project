# AI Research Assistant

面向论文的单 PDF RAG 问答应用，包含 Streamlit 前端、PDF 原页证据查看、
本地纯向量/混合检索，以及带人工审核门槛的检索评估工具。

## 安装与运行

使用 Python 3.11，在项目根目录执行（PowerShell）：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
Copy-Item .env.example .env
```

在本地 `.env` 填写自己的 `DEEPSEEK_API_KEY`。如平台不适用现有环境锁定快照，
可使用 `requirements.txt` 安装，再验证测试；不要上传密钥或本机环境目录。

应用使用本地缓存的 `sentence-transformers/all-MiniLM-L6-v2`，运行时不会自动下载。
新机器需预先准备模型。以下是需要联网、显式下载模型的独立命令：

```powershell
.\.venv\Scripts\python.exe -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

上传含可提取文本的 PDF 后即可提问。问答及可选的中文检索翻译会调用 DeepSeek；
点击回答下方的实际引用可查看原文和 PDF 页面。`.streamlit/config.toml` 限制上传 20 MB。

## 离线测试

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

测试使用 Mock LLM/Embedding 或合成向量，部分测试真实读取 PDF、Chroma、Streamlit，
不调用真实云端 API。测试依赖的公开 `data/sample.pdf` 随仓库提供。

## 真实检索评估

```powershell
.\.venv\Scripts\python.exe scripts/run_retrieval_experiment.py --max-papers 1 --include sample.pdf --chunk-size 600 --chunk-overlap 120 --top-k 5
```

该命令使用真实本地 Embedding 和独立 Chroma 数据库，两模式共享问题、查询向量、
语料和 Top-K，仅改变 `hybrid`。输出位于 `data/p3_phase2/runs/<run-id>/`，不覆盖旧报告。
现有五题为待审核开发题，只输出探索性排名，正式 Recall/MRR 为 null。

纯指标框架 `python -m app.retrieval_evaluation` 只校验或消费排名 JSON，不执行检索。
候选生成脚本默认仅预览；只有明确传入 `--execute` 才调用付费 API，生成结果保持待审核。

## 文档与数据

- [架构与模块边界](ARCHITECTURE.md)
- [检索评估协议、批量语料与候选生成命令](docs/RETRIEVAL_EVALUATION.md)
- [Streamlit 与证据查看](docs/STREAMLIT.md)
- [本地 Embedding](docs/EMBEDDING.md)
- [数据来源与重建](data/README.md)
- [开发记录](docs/DEV_LOG.md)

大型 PDF、模型缓存、Chroma、上传文件、运行报告和自动生成数据集不随 Git 上传。
历史文档中的本机运行目录用于追溯，克隆后应重新执行命令生成自己的结果。
