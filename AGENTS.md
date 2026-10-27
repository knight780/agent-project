# Project guidance

Applies to this entire repository.

- Read README.md and relevant docs before changing module boundaries.
- Keep Loader, Splitter, Embedding, Chroma, Retriever, Prompt and Generator changes within the user's requested scope. Prefer adapters over unrelated rewrites.
- Run tests with `.venv\Scripts\python.exe -m pytest -q`; use Mock for LLM calls and synthetic or mocked vectors in tests. Do not call paid APIs from tests.
- Real candidate generation requires an explicit command with `--execute`; preserve pending review status, raw responses and review history.
- Never treat similarity scores, unreviewed candidates or incomplete relevance labels as complete gold labels. Report null when no formal samples qualify.
- Never commit .env, credentials, virtual environments, databases, caches or generated runtime datasets. Keep `.env.example` free of real values.
- Preserve local PDFs and databases. Use isolated versioned evaluation storage; do not clear demonstration or user data.
- Do not commit or push unless the user authorizes it. Never force-push or modify main without explicit authorization.
