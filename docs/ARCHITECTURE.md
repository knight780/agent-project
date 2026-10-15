# 基础 RAG 架构（T01）

## 数据流

索引：PDF 路径 → PDFLoader → PaperPage[] → Splitter → Chunk[] → Embedding.embed_documents → EmbeddedChunk[] → VectorStore（Chroma）→ paper_id。

问答：paper_id + 问题 → Retriever → Embedding.embed_query → VectorStore.search → Evidence[] → Generator → Answer（中文答案、引用或拒答）。

RAGService 未来负责上述编排。T01 仅声明接口，不执行任何业务步骤；组件通过构造参数注入。导入与实例化不初始化客户端、不读取 PDF、不创建数据库。

## 输入输出契约

- PDFLoader.load(Path) → list[PaperPage]：按页输出，保留真实标题、源路径和从 1 开始的 PDF 物理页码。
- Splitter.split(Sequence[PaperPage]) → list[Chunk]：页内切分，保留来源及 chunk_id。
- Embedding.embed_documents(Sequence[str]) → list[tuple[float, ...]]：向量与输入一一对应；embed_query(str) → tuple[float, ...] 使用同一模型。
- VectorStore.add(paper_id, Sequence[EmbeddedChunk]) → None：持久化向量及元数据；search(paper_id, query_vector, top_k) → list[Evidence]，限定论文范围。
- Retriever.retrieve(paper_id, question, top_k) → list[Evidence]：返回排序后的证据。
- Generator.generate(question, Sequence[Evidence]) → Answer：生成中文回答或明确拒答。
- RAGService.ingest(Path) → str：返回稳定 paper_id；ask(paper_id, question) → Answer。

## 来源与拒答

PaperPage 包含 paper_id、paper_title、source_path、page_number、text。Chunk 引用页面并保存分块文本，Evidence 包含分块和检索分数。Citation 显式保存论文标题、页码、源路径、原文片段及 chunk_id。Answer 包含 text、citations、refused、refusal_reason。

未来实现必须保证引用对应实际检索原文，禁止编造标题或来源。空证据或证据不足时明确拒答，不以模型常识补齐论文结论。T01 数据类仅表达契约，尚未验证来源真实性或执行拒答策略；所有业务方法抛出 NotImplementedError。

## 后续待确定

- 云端供应商、模型、凭据、超时配置。
- PDF 真实标题识别及无法识别时的处理策略；物理页码与印刷页码区别。
- 分块长度单位、稳定 ID、重复入库及索引生命周期。
- Chroma 适配方式、向量维度、分数方向和证据充分性判定。
- 网络失败、无效 PDF、扫描件、空文本及配置错误处理。

当前不使用 LangGraph，不实现 Web，不调用真实 API。