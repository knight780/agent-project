# TOP5 检索评估与改进

## P3 第一阶段：离线评测框架

正式评测入口为 `app/retrieval_evaluation.py`。本阶段只消费已经保存的排名，
不加载 Embedding、不操作 Chroma、不检索、不生成问题，也不调用 DeepSeek。
下方原 `try_chroma_real.py --evaluate` 保留为历史开发诊断入口；它的分数不受新审核门槛
保护，不应作为正式 P3 评测结果。后续阶段再接入两模式自动实验。

### 数据协议与旧数据保留

`tests/fixtures/retrieval_gold.json` 已扩展为 `schema_version=2`，版本
`legacy-dev-v2-unreviewed`。旧 question、annotation、grades、rubric、scope 及指纹
原样保留；未将 Agent 制定的标签伪装成人工标签。新增稳定 question_id、论文 PDF SHA-256、
origin、answerability、review、review_history、labels 及 target_evidence。
这五题均为 `legacy_agent / pending / partial`，没有创建任何人工批准记录。

`papers` 保存按论文哈希隔离的冻结 Chunk 清单（完整正文、正文 SHA-256、从 1 开始的
物理页码、固定来源标识）；`chunk_config` 明确为 600/120。迁移时使用现有 Loader 和
Splitter 读取 `data/sample.pdf`，核对原 PDF 哈希及切分指纹，未调用模型。
文件中的 source 仅作为证据身份，不由评测器当作路径打开。

标签规则：0=显式判无直接证据，1=相关背景，2=直接回答证据；正式指标只把 2 视为相关。
未列入 judgments 的 Chunk 是“未知”，不自动补 0。target_evidence 必须与显式的全部
2 分标签对应，且来源、页码、原文必须匹配冻结清单。

审核门槛：approved 必须有 reviewer、合法 reviewed_at 和匹配的 approve 历史记录。
pending/rejected 不产生任何指标。人工修订应保留旧 grades、已有审核事件及原始输出，
追加记录、更新当前 labels/evidence 并提升 dataset_version；框架不会自动修改输入数据。
框架校验审核记录结构，但不能认证审核者身份，也不能证明语义标注本身正确。

### 指标与异常处理

- `complete_reviewed` 要求人工批准且 judgments 显式覆盖该论文的全部 Chunk。
  可回答且至少有一个相关 Chunk 时才计算 Recall@K 和 MRR@K。
- Recall@K = 前 K 条中不同相关 Chunk 数 / 全部相关 Chunk 数。
  每题 mrr_at_k 为首个相关结果排名的倒数；报告均值才是 MRR@K，前 K 无命中计 0。
- 已批准但标签为 `partial` 或 `anchor_only`：仅报告 anchor_hit_at_k 和 anchor_mrr_at_k；
  普通 recall_at_k/mrr_at_k 为 null，避免把已知锚点当成完整相关证据集合。
- 不可回答题单独排除于本阶段指标，并标明需要独立拒答评估；声称可回答却没有相关证据
  属于无效样本。已批准的不可回答标签也不等于已完成拒答实验。
- 显式空列表是有效空检索，指标为 0；缺少该题输出是错误，不与空检索混淆。
- 少于 K 条按实际结果计算，不补齐；超过 K 条只用前 K 条计算，并保留全部排名和提示。
- 重复结果不去重压缩排名，而是整题无效；未知 Chunk、错论文、错来源、错页码或
  正文指纹不一致同样无效。每条结果（包括 K 以外的结果）都必须通过身份校验。
- 数据集版本/指纹/结构错误、重复 question_id、未知输出题号及非有限数值明确报错；
  单题错误记录在 cases 和 reason_counts 中，不静默删除。
- 汇总包含 total/valid/invalid/excluded、formal/anchor 样本数、审核及完整性分布、
  每项指标实际分母；没有适用样本时均值为 null，而非 0。valid 包含正式与锚点样本。

报告保存数据集内容哈希、版本、语料指纹、Top-K、逐题指标、结果排名及调用方传入的
run_metadata。排序分数只作为日志保留，不用作相关性标签。
数据集自身的证据一致性可以校验，但本阶段不会重新读取当前 PDF 或执行检索；
真实语料与排名版本的绑定由后续实验调用方完成。

### 运行与验证

仅检查当前数据集，不生成报告或访问模型：

```powershell
.\.venv\Scripts\python.exe -m app.retrieval_evaluation --dataset tests/fixtures/retrieval_gold.json
.\.venv\Scripts\python.exe -m pytest -q tests/test_retrieval_evaluation.py
```

当前五题应均显示 excluded / review_pending，而不是输出正式指标。

对已有排名输出 JSON 报告：

```powershell
.\.venv\Scripts\python.exe -m app.retrieval_evaluation --dataset tests/fixtures/retrieval_gold.json --rankings <排名文件.json> --top-k 5 --output data/p3_retrieval_evaluation.json
```

排名文件是 `question_id -> 有序结果列表` 的 JSON 对象。每条结果必须包含 paper_id、
chunk_id、source、page_number、text_sha256；可附带 text、score、score_type、distance
等日志字段，列表顺序即排名。论文身份和正文指纹从对应冻结语料版本取得。
缺少符合审核条件问题的排名会标记无效。单题无效时仍输出报告并返回退出码 1；
整体输入错误返回 2；正常完成（包括只有明确排除样本）返回 0。

首次 `data/p3_retrieval_evaluation.json` 仅记录当前五题的框架验证结果：未执行检索、
正式与锚点样本数均为 0，所有指标为 null，不代表任何检索性能。

本阶段测试使用固定六块合成语料与 Mock 排名，不调用真实 API。手算用例：相关块
为 {1,3,5}，前五排名为 [0,1,2,3,4]，Recall@5=2/3，RR@5=1/2。
第一阶段验证结果：新增模块测试 **55 passed**；完整离线回归 **164 passed**，
仅一条已有 langchain-community 弃用警告。Windows 沙箱拒绝临时文件的原子重命名，
且既有 Streamlit 测试需要本地 socket，因此完整回归在获准的沙箱外环境完成。
未调用真实云端 API。

## P3 第二阶段：真实 PDF、独立索引及双模式实验

新增 `scripts/run_retrieval_experiment.py`（同时运行两种模式）、
`app/retrieval_experiment.py`（语料与实验适配）、
`scripts/generate_retrieval_dataset.py` / `app/retrieval_candidates.py`（候选生成）。
现有 Loader、Splitter、Embedding、index_chunks、PaperRetriever 和问答组件均未重写。

### 独立存储与版本绑定

默认检查 `data/` 全部 PDF 路径，但只导入选定的顶层论文；`data/web/` 标记为上传缓存，
其他子目录不参与导入。清单保存相对路径、PDF SHA-256、导入状态、页数、Chunk 数、
arXiv 版本（能识别时）、标题及同论文不同版本/格式的线索。字节相同或完整标准化正文
相同的副本跳过；标题/编号相似只是待复核线索，不自动认为不同版本的证据可互换。
无法确定论文关系时明确标为 unresolved_or_unique。非选中论文的版本信息可能不完整。

数据库固定位于 `<output-dir>/evaluation_indexes/<配置哈希>/`，不会清空原来的
`data/chroma_real`、`data/chroma_demo` 或 `data/web`。复用原 index_chunks 生成的真实
Collection 名称，不假定固定名称；相同内容/config 再运行会复用已有文档向量。
每次执行创建独立 `runs/<时间>-<随机ID>/`，保存 corpus.json、report.json、summary.json；
失败保存 failure.json，已有报告不覆盖。检查出的导入错误保留在语料清单，退出码非零。

显式参数为 600/120、all-MiniLM-L6-v2、Top-K=5，可通过命令参数更改。
语料版本包含论文哈希、正文和页码、切块配置、模型名、可获取的本地模型 revision、
核心实现文件哈希及依赖版本，并包含实际加载权重及 tokenizer 的 SHA-256。
模型 revision 无法获取时为 null，不虚构版本；仍通过实际权重指纹隔离同名模型变更。
全局 Chunk 标识 chunk_uid=`PDF哈希:切块正文版本:chunk_N`；局部 chunk_id 保留原
Chroma 的 chunk_N，二者一起保存。单独拿 chunk_N 不能跨论文判断命中。

实验前逐一核对数据集与实际重载 PDF 的哈希、来源、页数、Chunk ID、原文、页码和指纹。
不匹配即失败，不对旧证据进行猜测映射。每题仅编码一次，保存查询向量及指纹，复制给
两次检索调用；仅 hybrid=False/True 不同。实验范围是逐篇论文检索，不是全库跨论文检索。

每条结果保存问题 ID、模式、配置版本、真实排名、paper_id、chunk_id、chunk_uid、
source、从 0 开始的 page、从 1 开始的 page_number、正文及哈希、score、distance、
score_type、neighbor_from、valid/error。额外验证实际 Document 与 Chunk 清单的一致性。
任一模式失败，两组都不将该题计入指标，原始结果和错误仍保存，避免样本分母不一致。
待审核题可以运行，但只输出 exploratory 排名和 known_target_ranks，不计算正式指标。

### 已完成的本地试运行

`data/` 顶层有 10 个 PDF（字节哈希均不同），另外发现 1 个 Web 缓存副本并排除。

1. 原有 600/120 标签版本：`sample.pdf`，89 个 Chunk，五道历史问题完成 10 次真实检索，
   每个问题均运行纯向量与混合检索，无检索错误。报告位于
   `data/p3_phase2/runs/20261026T145909Z-81be5d11/report.json`。
   两模式正式样本数均为 0、排除 5（review_pending），Recall/MRR 均为 null。
   真实名次示例：定义题 chunk_19 从第 3 到第 1；位置编码题 chunk_34/35/36
   从未进入前五到第 1/2/3。仅描述未审核开发题的排名，不代表正式性能提升。
2. 批量候选试点采用显式新版本 400/120：GAN `1406.2661v1.pdf` 109 个 Chunk、
   BERT `1810.04805v2.pdf` 228 个 Chunk、`sample.pdf` 150 个 Chunk，共 487 个。
   清单位于 `data/p3_phase2/runs/20261026T150006Z-568dbf24/corpus.json`。
   此次为 corpus-only，没有把 600/120 的旧标签迁移到新切块；报告中没有检索样本。
   已预览三篇各 3 题，共 9 个生成任务，未调用付费 API。

600 字符不保证满足 MiniLM 的 256 token 上限。试运行中 BERT、ResNet 等有超限 Chunk，
已保留失败试验目录；不截断、不调整 Embedding、不静默缩小 Chunk。
因此候选试点显式建立 400/120 新版本。全量导入仍可能遇到超限或字体解析问题，
应根据清单处理；调整切块必须建立新版本并重新生成/审核相应标签。

复现五题真实双模式对照：

```powershell
.\.venv\Scripts\python.exe scripts/run_retrieval_experiment.py --pdf-dir data --dataset tests/fixtures/retrieval_gold.json --output-dir data/p3_phase2 --max-papers 1 --include sample.pdf --chunk-size 600 --chunk-overlap 120 --top-k 5
```

建立三篇候选语料，或扩展到全部现有顶层 PDF：

```powershell
.\.venv\Scripts\python.exe scripts/run_retrieval_experiment.py --pdf-dir data --output-dir data/p3_phase2 --corpus-only --max-papers 3 --include sample.pdf 1406.2661v1.pdf 1810.04805v2.pdf --chunk-size 400 --chunk-overlap 120
.\.venv\Scripts\python.exe scripts/run_retrieval_experiment.py --pdf-dir data --output-dir data/p3_phase2 --corpus-only --all-papers --chunk-size 400 --chunk-overlap 120
```

### 候选题：默认预览，显式命令才调用 API

生成器只读取已有 corpus.json，不重新加载 PDF、切分、编码或入库。固定种子先采样论文，
再优先选不同章节、不同页面、非相邻 Chunk；类型轮换 fact/concept/method。
每篇 3～5 题，语料不足时明确报错，不以相邻重复证据凑数。

```powershell
$pilotCorpus = 'data/p3_phase2/runs/20261026T150006Z-568dbf24/corpus.json'
.\.venv\Scripts\python.exe scripts/generate_retrieval_dataset.py --corpus $pilotCorpus --output-dir data/p3_phase2/candidates_pilot --paper-count 3 --per-paper 3
```

上述命令仅写计划。以下命令会实际调用现有 DeepSeek 配置，产生费用；再次运行同一命令
从该目录的 candidates.json 续跑：

```powershell
.\.venv\Scripts\python.exe scripts/generate_retrieval_dataset.py --corpus $pilotCorpus --output-dir data/p3_phase2/candidates_pilot --paper-count 3 --per-paper 3 --max-requests 40 --attempts-per-stage 2 --interval 1 --execute
```

通常每题一次生成、一次语义检查，9 题约 18 个逻辑请求。串行执行，每阶段至多 2 次尝试，
总预算按已持久化的逻辑请求数（包括失败及中断未知请求）限制；现有客户端每次请求可能
内部再重试一次，因此默认 40 个逻辑请求的底层 HTTP 尝试上界是 80，不是实际计费承诺。
预算不足可提高同一计划的总预算后续跑；明确拒绝的坏候选保留原始响应，不自动反复生成。
网络/服务异常只记录异常类别，不记录可能包含密钥的异常文本。

确定性检查 JSON、必需字段、类型、真实 Chunk/页码、原文支持片段和重复/近似重复问题；
再请求模型检查答案支持、歧义和单跳性。模型复核不是人工 Gold，所有接纳的候选仍是
candidate_status=pending_review、review.status=pending、anchor_only。
候选答案、原始证据、配置模型名、提示词版本、原始响应、失败记录和审核历史独立保存。
不会改写 retrieval_gold.json。人工修订应追加 review_history 并提升数据版本；
不建议同时启动多个进程写同一个候选检查点目录。

候选生成后，使用完全相同的 400/120 语料参数运行两模式探索实验：

```powershell
.\.venv\Scripts\python.exe scripts/run_retrieval_experiment.py --pdf-dir data --dataset data/p3_phase2/candidates_pilot/candidates.json --output-dir data/p3_phase2 --max-papers 3 --chunk-size 400 --chunk-overlap 120 --top-k 5
```

### 离线测试

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_retrieval_experiment.py tests/test_retrieval_candidates.py
.\.venv\Scripts\python.exe -m pytest -q
```

新增测试覆盖共享查询向量/参数、真实 PDF 加 Mock Embedding 的独立 Chroma、索引复用、
身份与版本绑定、空/重复/错证据、正式与探索性分流、候选校验/预算/续跑/不污染正式集，
以及候选集接入双模式实验和 JSON 往返。测试不调用真实 LLM API。
最终新增测试 **34 passed**，完整离线回归 **198 passed**（保留原有 164 项），
一条原有 langchain-community 弃用警告。沙箱限制临时文件原子重命名及 Streamlit
本地 socket，回归在获准的沙箱外环境完成。真实检索仅使用本地缓存 Embedding；
付费候选生成已完成 Mock 闭环及真实语料预览，本次未实际调用 DeepSeek。

## 历史开发诊断（以下为 P3 之前的口径）

## 复现

在项目根目录、已激活 `.venv` 的终端运行：

```powershell
python try_chroma_real.py
python try_chroma_real.py --mode hybrid
python try_chroma_real.py --question "Why does the Transformer use positional encoding?"
python try_chroma_real.py --evaluate
python -m pytest -q
```

默认展示同一份索引上的纯向量基线和改进版。`--evaluate` 生成
`data/retrieval_evaluation.json`，其中保留每题两种方法的全部 TOP5 原文、页码、
来源、分数、相邻补召回锚点及指标。模型仍为本地缓存 MiniLM，无云端推理。

## 原脚本的问题

原问题是 `What is the attention mechanism?`。原 TOP5 页码依次为
13、15、3、7、14：第 1、2、5 条是附录中的图示说明，第 4 条同时涉及 attention
可解释性和训练数据，第 3 条才包含 query/key/value 与加权和的定义。
这些结果大多在主题上相关，但并不都能直接回答定义问题。

第 3 页定义最后一句在页尾中断，续句在第 4 页；原脚本没有展示后续文本。
这属于按页切分后的检索上下文问题，不需要改写 Loader 或 Splitter。

原脚本还只保存 `page_number`，缺失来源与块序号；固定 `chunk_0` 等 ID 配合
重复 upsert，在语料变短时可能残留旧记录，在更换论文/模型时也缺少隔离。

原 Collection 默认平方 L2 距离。本模型输出单位向量，平方 L2 等于余弦距离的两倍，
所以切换成余弦距离本身不会改善排序。对照基线复现了原 TOP5 顺序；指标提升来自
召回与排序逻辑，不能归功于距离数值变小。

## 本次改动

- 加入 BM25 关键词召回，与向量候选各取最多 30 条，使用 RRF 融合。
- 根据论文的编号标题推断章节，让匹配章节及其子章节优先；不硬编码问题答案或页码。
- 对含 `<pad>` 的图示文本降低排序权重；明确询问图示、示例的问题保留正常权重。
- 对前三个候选进行一次相邻块补召回，限制同一来源和同章节/子章节，保留锚点信息。
  相邻补召回系数为 0.85，不递归传播。它能带回难以独立匹配的跨页续句，也可能带回脚注。
- 展示续接上下文时单独标注页码，不把展示出来的上下文额外计为 TOP5 命中。
- 保存 source/page/page_number/chunk_index，按正文、metadata 与模型名称生成版本化
  Collection；重跑同一语料复用索引，不删除原 `research_papers` Collection。
- 增加 main 入口与命令行参数；导入脚本不会自动建库或加载模型。

这是一组可解释的启发式规则。标题推断并非论文结构解析器，可能识别错边界；
英文 BM25 不覆盖中文分词。模型权重更换但名称不变时，需要提升索引版本重新建库。

## 评估口径和结果

标注在 `tests/fixtures/retrieval_gold.json`，由本次评估对照本地论文文本制定，供复核。
2 分表示原块含直接回答事实（允许事实不完整），1 分表示相关背景/示例，0 分表示
没有回答证据或未列入标签。PDF 与切分正文的哈希不匹配时，脚本拒绝沿用标注。

直接证据 P@5 只统计 2 分原块；相关性 P@5 统计至少 1 分；nDCG@5 同时考虑分级与名次。
分母固定为 5。某问题的直接证据块不足 5 个时，即使理想检索，直接 P@5 也未必为 100%。

- Attention 定义：直接 P@5 从 20% 到 40%；首个直接命中从第 3 到第 1；
  nDCG@5 从 0.548 到 0.825。改进 TOP5 为 chunk 19、28、20、22、23，
  对应页码 3、5、4、4、4。定义与续句已找回，其余为应用、类型和缩放解释。
- Scaled dot-product：直接 P@5 保持 80%；nDCG@5 从 0.989 到 1.000。
- Positional encoding：直接 P@5 从 0% 到 60%；前三条为第 6 页的相关正文。
- Optimizer / learning rate：直接 P@5 保持 40%；TOP5 尾部仍混入背景内容。
- Multi-head 合并：直接 P@5 从 20% 到 40%；两条直接证据排在前两名，
  但相邻补召回也带入一条方差解释脚注，说明该策略并非总能提高相关性。

五题平均直接 P@5 从 32% 到 52%，平均 nDCG@5 从 0.673 到 0.936。
这五题是改进中使用的单论文英文开发回归集，**不是独立测试集，也不是产品回答准确率**。
需要另外增加未参与调试的问题、多篇论文和标注复核，才能评估泛化效果。

## 验证

`python -m pytest -q`：41 passed，1 条原有 langchain-community 弃用警告。
新增用例覆盖定义/图示排序、小语料与空查询、章节与来源边界、上下文续接、
BM25、相邻补召回不递归传播、真实临时 Chroma 的索引复用/版本隔离及评估指标口径。
