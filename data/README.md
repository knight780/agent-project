# Data and reproducibility

仓库仅包含公开测试样例 `sample.pdf`；其他本地语料、Chroma 数据库、上传缓存、
运行报告和生成数据集由 .gitignore 排除，不会删除本地文件。

## 版本绑定测试样例

- 论文：Vaswani et al., *Attention Is All You Need*。
- 来源：https://arxiv.org/abs/1706.03762v7
- PDF：https://arxiv.org/pdf/1706.03762v7
- 本地样例 SHA-256：`bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697`。
- 600/120 切分为 89 个 Chunk。`tests/fixtures/retrieval_gold.json` 绑定此字节和正文版本。

重新下载的字节若不同，即使是同一论文，也不能直接沿用版本绑定标签；先检查来源版本、
PDF/正文指纹，再创建新语料版本。标签仍是未审核的 Agent 开发标签，不是正式 Gold。

## 可选扩展语料

将公开论文 PDF 自行下载到 data 顶层，示例：

- GAN：https://arxiv.org/pdf/1406.2661v1 → `1406.2661v1.pdf`
- BERT：https://arxiv.org/pdf/1810.04805v2 → `1810.04805v2.pdf`
- ResNet：https://arxiv.org/pdf/1512.03385v1 → `1512.03385v1.pdf`
- RAG：https://arxiv.org/pdf/2005.11401v4 → `2005.11401v4.pdf`
- ViT：https://arxiv.org/pdf/2010.11929v2 → `2010.11929v2.pdf`

示例下载命令（可选、联网）：

```powershell
Invoke-WebRequest https://arxiv.org/pdf/1406.2661v1 -OutFile data/1406.2661v1.pdf
Invoke-WebRequest https://arxiv.org/pdf/1810.04805v2 -OutFile data/1810.04805v2.pdf
```

## 重建本地索引及报告

按 README.md 准备依赖和模型缓存后执行：

```powershell
.\.venv\Scripts\python.exe scripts/run_retrieval_experiment.py --max-papers 1 --include sample.pdf --chunk-size 600 --chunk-overlap 120 --top-k 5
.\.venv\Scripts\python.exe scripts/run_retrieval_experiment.py --corpus-only --all-papers --chunk-size 400 --chunk-overlap 120
```

输出在 `data/p3_phase2/`，每次创建新的运行目录；重建不会下载模型、调用 LLM 或清空旧库。
对任一论文导入失败时查看 corpus.json/failure.json；调整 Chunk 配置需建立新版本。
候选生成复用新运行的 corpus.json，详见 docs/RETRIEVAL_EVALUATION.md。
