"""Local, offline embeddings shared by document indexing and query retrieval."""

from langchain_core.embeddings import Embeddings


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _load_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        model_name,
        device="cpu",
        local_files_only=True,
        trust_remote_code=False,
    )


class LocalEmbeddings(Embeddings):
    """Encode English text on CPU using an already cached model.

    Texts and returned vectors have matching order. The caller keeps the
    original Documents (including metadata) alongside those vectors.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = _load_model(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("待向量化的文本必须是非空字符串。")

        # Character limits do not guarantee token limits. Reject rather than
        # silently truncate chunks containing formulas or unusual PDF text.
        tokens = self._model.tokenizer(
            texts, truncation=False, add_special_tokens=True
        )["input_ids"]
        for index, token_ids in enumerate(tokens):
            if len(token_ids) > self._model.max_seq_length:
                raise ValueError(
                    f"文本索引 {index} 有 {len(token_ids)} token，超过模型上限 "
                    f"{self._model.max_seq_length}；请减小 chunk_size 或使用更短的查询。"
                )

        return self._model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
