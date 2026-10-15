# 项目开发规则

适用于本仓库全部文件。

## 当前技术方案

- 面向大学生与科研工作者，支持单篇中英文 PDF，默认中文回答。
- Python 3.11、LangChain、Chroma、云端 LLM/Embedding API、pytest。
- 开发顺序：基础 RAG → 检索增强 → LangGraph Agent → Web 可视化。
- 当前 T01 仅提供类型契约与工程骨架，业务方法抛出 NotImplementedError。
- 模块职责与数据流见 docs/ARCHITECTURE.md。

## Codex 开发规则

- 修改前检查现有文件和规则，保留用户文件，不覆盖依赖配置。
- 未经任务授权，不调用真实云端 API、不引入 LangGraph、不实现 Web。
- 不自动执行 git commit、push 或 merge。
- 禁止提交 API 密钥、凭据或 .env。
- 使用类型标注与可注入组件；导入模块不得请求网络或创建数据库。
- 保留真实论文标题、从 1 开始的 PDF 物理页码和来源片段，禁止伪造引用。
- 证据不足时明确拒答；后续实现必须测试来源及拒答行为。
- 默认测试离线，使用替身隔离云端依赖；在根目录执行 python -m pytest -q。
- 完成任务后说明修改文件、验证结果、限制和人工审查事项。