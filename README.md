# AI Research Assistant

面向大学生与科研工作者的 AI 论文研究助手。当前为 T01 工程骨架，所有业务方法均未实现。原有 main.py 为占位脚本，保持原样。

## 项目目标

- 单篇 PDF 解析，支持中英文论文，使用中文回答。
- 基于论文内容进行 RAG 问答，返回真实论文名称、页码及来源片段。
- 检索证据不足时明确拒答。
- 技术栈：Python 3.11、LangChain、Chroma、云端 LLM API、云端 Embedding API、pytest。
- 开发顺序：基础 RAG → 检索增强 → LangGraph Agent → Web 可视化。

本次不实现完整 RAG、不调用云端 API、不实现 Web、不使用 LangGraph。现有依赖清单包含 LangGraph 和 FAISS，均原样保留；基础向量库采用 Chroma。

## 运行环境与验证

在项目根目录使用 Python 3.11，激活现有虚拟环境后运行：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest -q
```

也可直接运行 `.\.venv\Scripts\python.exe -m pytest -q`。
新环境先执行 `python -m venv .venv`，激活后执行 `python -m pip install -r requirements.txt`。完整依赖安装需要网络；T01 模块只依赖标准库，测试只需要 pytest，无需 API 密钥。系统默认 Python 当前未安装 pytest。

requirements.txt 与 requirements-lock.txt 均保留原样。尚无可运行的 RAG 服务命令。

## 目录与职责

```text
app/
  __init__.py
  config/
    __init__.py
    settings.py       # 离线配置数据类
  rag/
    __init__.py
    models.py         # 页面、分块、向量、证据、引用和回答契约
    pdf_loader.py     # PDFLoader
    splitter.py       # Splitter
    embedding.py      # Embedding
    vector_store.py   # VectorStore
    retriever.py      # Retriever
    generator.py      # Generator
    service.py        # RAGService
 tests/test_smoke.py  # 离线冒烟测试
 docs/ARCHITECTURE.md # 数据流与接口说明
 AGENTS.md           # 全项目开发规则
```

接口细节见 [架构文档](docs/ARCHITECTURE.md)。Settings 的分块长度、重叠长度和 top_k 均为待调优占位值，长度单位及证据充分性阈值尚未确定。