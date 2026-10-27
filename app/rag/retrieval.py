"""Small local hybrid retriever; no changes to PDF loading or splitting."""

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import re

from langchain_core.documents import Document


# Keep domain nouns (including mechanism and transformer).
STOP_WORDS = set(
    "a an the of in on at to from for with by and or is are was were be been "
    "what how why which does do did can could would should it its this that "
    "these those as we they their our use used work works combine".split()
)
HEADING = re.compile(r"^([1-9](?:\.[1-9]\d?){0,2})[ \t]+([A-Z][^\n]+)$", re.MULTILINE)


def terms(text: str) -> list[str]:
    words = re.findall(r"[a-z]+", text.lower())
    return [
        word[:-1] if word.endswith("s") and not word.endswith("ss") and len(word) > 3 else word
        for word in words if word not in STOP_WORDS
    ]


def infer_section_info(chunks: list[Document]) -> list[tuple[str, str]]:
    """Carry numbered headings forward without treating years as headings."""
    active = ("", "")
    result = []
    previous_source = None
    for chunk in chunks:
        source = chunk.metadata.get("source")
        if source != previous_source:
            active = ("", "")
        previous_source = source
        text = chunk.page_content
        title = active
        for match in HEADING.finditer(text):
            # A heading at the very end describes the next chunk, not this one.
            if match.start() < len(text) * 0.6:
                title = (match.group(1), match.group(2).strip())
            active = (match.group(1), match.group(2).strip())
        if re.search(r"^References\s*$", text, re.MULTILINE):
            active = ("", "References")
        if re.search(r"^Attention Visualizations\s*$", text, re.MULTILINE):
            title = active = ("", "Attention Visualizations")
        result.append(title)
    return result


def infer_sections(chunks: list[Document]) -> list[str]:
    return [title for _, title in infer_section_info(chunks)]


class BM25:
    def __init__(self, texts: list[str]):
        self.counts = [Counter(terms(text)) for text in texts]
        self.lengths = [sum(count.values()) for count in self.counts]
        self.average = sum(self.lengths) / max(len(texts), 1) or 1.0
        self.df = Counter(word for count in self.counts for word in count)

    def scores(self, query: str) -> list[float]:
        scores = []
        for count, length in zip(self.counts, self.lengths):
            score = 0.0
            for word in set(terms(query)):
                frequency = count[word]
                idf = math.log(1 + (len(self.counts) - self.df[word] + 0.5) / (self.df[word] + 0.5))
                score += idf * frequency * 2.5 / (
                    frequency + 1.5 * (0.25 + 0.75 * length / self.average)
                )
            scores.append(score)
        return scores


@dataclass
class Hit:
    index: int
    document: Document
    section: str
    score: float
    distance: float | None
    neighbor_from: int | None = None


class PaperRetriever:
    def __init__(self, collection, chunks: list[Document]):
        self.collection = collection
        self.chunks = chunks
        self.sections = infer_sections(chunks)
        self.section_numbers = [number for number, _ in infer_section_info(chunks)]
        self.lexical = BM25([
            f"{section}\n{section}\n{chunk.page_content}"
            for section, chunk in zip(self.sections, chunks)
        ])

    def search(self, question: str, query_vector: list[float], *, hybrid=True, k=5) -> list[Hit]:
        if not question.strip():
            raise ValueError("查询不能为空。")
        if k <= 0:
            raise ValueError("k 必须为正数。")
        if not self.chunks:
            return []
        candidate_count = min(max(30, k), len(self.chunks)) if hybrid else min(k, len(self.chunks))
        raw = self.collection.query(
            query_embeddings=[query_vector], n_results=candidate_count,
            include=["metadatas", "distances"],
        )
        dense = [int(meta["chunk_index"]) for meta in raw["metadatas"][0]]
        distances = dict(zip(dense, raw["distances"][0]))
        if not hybrid:
            return [
                Hit(i, self.chunks[i], self.sections[i], 1 - distances[i], distances[i])
                for i in dense
            ]

        lexical_scores = self.lexical.scores(question)
        sparse = sorted(
            (i for i, score in enumerate(lexical_scores) if score > 0),
            key=lambda i: (-lexical_scores[i], i),
        )[:candidate_count]
        # Reciprocal rank fusion avoids comparing raw BM25 and cosine scales.
        fused: dict[int, float] = {}
        for ranking in (dense, sparse):
            for rank, index in enumerate(ranking, 1):
                fused[index] = fused.get(index, 0.0) + 1 / (60 + rank)

        query_terms = set(terms(question))
        # Prefer the closest heading and its descendants, rather than unrelated
        # chapters sharing a broad word. No paper-specific topic/page rules.
        def heading_match(index):
            heading_terms = set(terms(self.sections[index]))
            return len(query_terms & heading_terms) / max(len(query_terms | heading_terms), 1)

        numbered = [i for i in range(len(self.chunks)) if self.section_numbers[i]]
        best = max(numbered, key=heading_match, default=None)
        preferred = self.section_numbers[best] if best is not None and heading_match(best) > 0 else ""
        wants_figures = bool(query_terms & {"figure", "diagram", "visualization", "example", "appendix"})
        for index in fused:
            heading_terms = set(terms(self.sections[index]))
            overlap = len(query_terms & heading_terms)
            coverage = overlap / max(len(query_terms), 1)
            precision = overlap / max(len(heading_terms), 1)
            fused[index] *= 1 + coverage + 0.5 * precision
            number = self.section_numbers[index]
            if preferred and (number == preferred or number.startswith(preferred + ".")):
                fused[index] *= 1.75
            # Retain illustrations; lower their priority for non-visual questions.
            if "<pad>" in self.chunks[index].page_content and not wants_figures:
                fused[index] *= 0.4

        # A definition can end at a page boundary, with its continuation ranking
        # poorly in isolation. Expand only the top three seeds by one chunk in
        # the same numbered section/subsection and source, without recursion.
        neighbors = {}
        seeds = sorted(fused, key=lambda i: (-fused[i], i))[:3]
        seed_scores = {seed: fused[seed] for seed in seeds}
        for seed in seeds:
            seed_number = self.section_numbers[seed]
            for index in (seed - 1, seed + 1):
                if not 0 <= index < len(self.chunks) or not seed_number:
                    continue
                number = self.section_numbers[index]
                same_family = number == seed_number or number.startswith(seed_number + ".")
                same_source = self.chunks[index].metadata.get("source") == self.chunks[seed].metadata.get("source")
                if not same_family or not same_source or "<pad>" in self.chunks[index].page_content:
                    continue
                propagated = seed_scores[seed] * 0.85
                if propagated > fused.get(index, 0):
                    neighbors[index] = seed
                    fused[index] = propagated

        order = sorted(fused, key=lambda i: (-fused[i], i))[:k]
        return [
            Hit(i, self.chunks[i], self.sections[i], fused[i], distances.get(i), neighbors.get(i))
            for i in order
        ]

    def continuation(self, hit: Hit) -> Document | None:
        """Return separately cited continuation; never re-embed expanded context."""
        index = hit.index + 1
        if index >= len(self.chunks):
            return None
        next_doc = self.chunks[index]
        if next_doc.metadata.get("source") != hit.document.metadata.get("source"):
            return None
        current_number = self.section_numbers[hit.index]
        next_number = self.section_numbers[index]
        if not current_number or not (next_number == current_number or next_number.startswith(current_number + ".")):
            return None
        return next_doc


def index_chunks(client, chunks: list[Document], encoder):
    """Version collections by content/config; keep old collections untouched."""
    manifest = json.dumps(
        {"version": 1, "model": encoder.model_name,
         "chunks": [(doc.page_content, doc.metadata) for doc in chunks]},
        sort_keys=True, ensure_ascii=False,
    )
    fingerprint = hashlib.sha256(manifest.encode("utf-8")).hexdigest()[:20]
    collection = client.get_or_create_collection(
        name=f"paper_hybrid_{fingerprint}", embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"model": encoder.model_name, "index_version": 1, "fingerprint": fingerprint},
    )
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    existing = set(collection.get(include=[])["ids"])
    pending = [i for i, chunk_id in enumerate(ids) if chunk_id not in existing]
    if pending:
        vectors = encoder.embed_documents([chunks[i].page_content for i in pending])
        collection.upsert(
            ids=[ids[i] for i in pending], documents=[chunks[i].page_content for i in pending],
            embeddings=vectors,
            metadatas=[{
                "chunk_index": i, "source": str(chunks[i].metadata["source"]),
                "page": int(chunks[i].metadata["page"]),
                "page_number": int(chunks[i].metadata["page_number"]),
            } for i in pending],
        )
    return collection
