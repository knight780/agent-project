# T06 Streamlit 论文问答

在项目根目录运行，使用已有 `.env` 中的 `DEEPSEEK_API_KEY`：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Embedding 继续使用已有的本地缓存 MiniLM 模型，不会下载模型。默认纯向量检索，
可在侧栏切换混合检索。英文论文建议使用英文术语提问，例如
`What is Multi-Head Attention?`；生成器已要求使用中文回答。

Web V1.0 修复增加“中文问题先转英文检索”开关，默认开启；可通过 `.env` 中的
`RAG_TRANSLATE_CHINESE_QUERY=false` 关闭默认值。开启时中文问题额外调用一次 DeepSeek 翻译，
译文仅用于检索，原问题仍用于回答。关闭可复现原始 baseline，翻译错误不会静默降级。
详见 [修复与逐题验收](WEB_V1_ACCEPTANCE.md)。

## 数据交互

1. 侧栏单文件上传；限制 20 MB 并检查 PDF 文件头，再由现有 Loader 实际解析。
   损坏、加密、无文本文件返回友好提示。原始文件名只用于显示，去除路径和控制字符。
2. 内容 SHA-256 决定 `data/web/<hash>/paper.pdf` 和该目录下的 `chroma/`。
   Loader → Splitter（600/120）→ `index_chunks()`，复用已有指纹与缺失记录 upsert。
   同内容更名也使用同一目录，文件锁序列化并发建库，临时文件原子替换上传内容。
   入库 source 统一为 `paper.pdf`，不携带原始路径或跨会话文件名。
3. 全局 `st.cache_resource` 只缓存 Embedding 与推理锁；模型推理加锁，API 请求不占用该锁。
   每个会话独立持有 Paper、Retriever、文件名、历史；不全局缓存上传内容或答案。
   不同论文按哈希隔离；相同论文只共享磁盘索引。普通 rerun 不重新解析或入库。
4. 问题 → `embed_query()` → `PaperRetriever.search(hybrid=False/True, k=5)` →
   `build_context()` → `build_prompt()` → `call_deepseek()`。后者沿用 dotenv 配置读取。
   历史仅供聊天展示，每次检索使用当前问题，不实现依赖上文的指代消解。
5. 将答案引用编号与本次 sources 校验，只展示实际引用的编号、文件名、页码和 Chunk ID。
   Markdown 使用原生组件。空检索直接拒答且不调用 API；模型拒答统一提示；无引用或
   越界引用的回答不予展示。引用页码从 1 开始。
6. 更换、移除或上传无效论文清理旧历史和检索器，失败后不能误用之前的论文。
   API、索引及解析错误显示固定提示，不显示异常文本、内部路径、堆栈或密钥。

## P2：论文证据定位与查看

回答下方“本次实际引用来源”中的 `[S1]`、`[S2]` 按钮可打开该条回答的证据，
展示 PDF 文件名、真实物理页码、Chunk ID、完整 Chunk 原文和 PDF 原始页面。
本阶段不实现像素级文本高亮；回答正文仍保留原有的引用编号展示。

数据流：现有检索 hits → 原有 `build_context()` / 引用校验 → Web 服务层将对应
hit 的 `page_content` 补入实际引用的 `sources` → 与该回答的唯一 ID、论文 ID 一起
保存到会话消息 → 点击按钮记录 `(message_id, label, paper_id)` → 仅从该回答的
`sources` 查找唯一匹配 → PyMuPDF 渲染 `data/web/<paper_id>/paper.pdf` → `st.image()`。
不从模型生成的正文推测文件、页码或 Chunk，不修改原有检索、Prompt 或模型实现。

`page_number` 是从 1 开始的 PDF 物理页序号（并非论文印刷页眉中的编号），渲染时
转换为 `load_page(page_number - 1)`。只在用户打开证据后渲染单页，图像最长边限制
为 1800 像素。PDF 文件名仅用于展示；读取路径只由固定数据根目录、严格校验的
SHA-256 论文 ID 和固定文件名构成，拒绝重定向的目录或文件路径。
非法编号、重复映射、旧回答缺少原文、无效/越界页码、文件缺失、损坏及加密 PDF
均显示明确提示，不暴露服务器路径或异常堆栈。

不同回答即使都包含 `[S1]` 也拥有独立按钮键和来源快照。普通 Streamlit rerun
保留当前选中证据；切换、移除论文或清空聊天同时清除证据选择及旧按钮状态。
浏览器硬刷新开启新会话，不恢复旧证据。热更新之前缺少原文的历史回答会提示重新提问。

P2 离线测试位于 `tests/test_web_evidence.py`，覆盖真实双页 PDF 的渲染与页码转换、
异常文件和路径保护，以及 AppTest 中点击 `[S2]`、跨回答 `[S1]`、重跑、清空、
换论文和新会话。API 与 Embedding 使用 Mock，不调用云端服务。
完整离线回归：`python -m pytest -q`，**109 passed**；保留一条原有的
`langchain-community` 弃用警告。Streamlit AppTest 在支持本地 socket 的环境运行。

## 刷新与复用（基础行为）

浏览器硬刷新会开启新会话，聊天历史和文件选择清空；重新选择同一 PDF 后磁盘索引仍可复用，
无需重新计算文档 Embedding。重新解析与切分用于恢复当前会话 Retriever，`index_chunks()`
跳过已有 Chunk，不增加记录。普通组件操作导致的 rerun 保留 Paper，不重新加载模型、解析或入库。

参考 [Streamlit 缓存说明](https://docs.streamlit.io/develop/concepts/architecture/caching)
与 [Session State](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state)。

## 测试与边界

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

`tests/test_web.py` 使用 Mock API 和 Embedding，真实解析 `data/sample.pdf`，调用真实
Splitter、Chroma、Retriever、Prompt 构造；通过 Streamlit AppTest 验证上传、中文聊天、
引用展示、rerun、模式切换、换论文清空与移除文件。没有真实 DeepSeek 请求。

Web V1.0 修复后的完整回归结果：**86 passed，1 条原有 langchain-community 弃用警告**。
在受限 Windows 沙箱中，AppTest 创建本地 asyncio socketpair 可能阻塞；本次完整回归
在允许本地 socket 的执行环境完成，未调用真实 DeepSeek API。

拒答沿用后端 Prompt 对证据充分性的判断；编号校验不能证明每句话都被证据支持。
没有添加未经评估的相似度阈值或改写检索算法。真实 DeepSeek 回答质量仍需手动验收。
本版是本地单论文应用，没有账户鉴权或磁盘数据保留期限管理。
