# AI Research Assistant 架构

## 应用链路

`streamlit_app.py` → `app/web_service.py` → 现有 RAG 模块。

建库：PDF → `pdf_loader.load_pdf` → `splitter.split_documents` →
`LocalEmbeddings` → `index_chunks` → Chroma。

问答：原问题 → 可选查询翻译 → 查询 Embedding → `PaperRetriever.search` →
`build_context` / `build_prompt` → `call_deepseek` → 引用校验 → 回答。

`app/web_evidence.py` 仅按当前回答的 sources 映射读取证据；文件路径由固定目录和
论文哈希构造，PyMuPDF 按物理页码减 1 渲染，切换论文/清空聊天清除旧证据状态。

## 评估链路

`scripts/run_retrieval_experiment.py` → `app/retrieval_experiment.py` →
真实 PDF、现有 Loader/Splitter/Embedding/Chroma/PaperRetriever →
`app/retrieval_evaluation.py` → JSON 报告。

逐论文检索，模型权重、tokenizer、语料、切块配置均留有指纹。每题只编码一次，
纯向量和混合模式使用同一查询向量。评测数据库与 Web/演示数据库分离。

`scripts/generate_retrieval_dataset.py` → `app/retrieval_candidates.py`，只复用现有
冻结 Chunk 清单；串行、有限重试与预算限制，保留原始响应和审核历史。
默认预览；`--execute` 才调用现有 DeepSeek 配置。

完整且人工批准的标签才产生正式 Recall/MRR；不完整标签仅使用明确命名的锚点指标，
待审核问题不进入正式指标。相似度分数不作为真实相关性标签。

## 运行资产与测试边界

Git 保存源码、配置模板、测试、开发文档及公开测试样例。API 密钥、虚拟环境、
大型语料、模型/数据库缓存、生成数据和运行日志只留在本地。

单元测试的固定排名与 Mock 模型用于校验协议和算法；它们不代表真实模型的检索质量。
`app/retrieval_evaluation.py` 可独立消费排名文件，本身不执行检索。
