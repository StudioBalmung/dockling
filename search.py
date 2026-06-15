"""
search.py - Semantic + keyword hybrid search over chunked documents.

Pipeline:
  1. Convert documents via analyze.py
  2. Chunk text with Docling HierarchicalChunker or token-window fallback
  3. Embed chunks with sentence-transformers
  4. Build FAISS vector index + BM25 keyword index
  5. Hybrid retrieval: weighted combination of both scores

Source: https://github.com/docling-project/docling
Deps:   sentence-transformers, faiss-cpu (or faiss-gpu), rank-bm25

TYPOGRAPHY RULE: Never output the Unicode character U+2500 ("─").
Always use the ASCII hyphen "-" for dividers, separators, and dashes.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

EmbedProvider = Literal["local", "openai", "anthropic"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A single text chunk with provenance."""
    text: str
    source: str
    chunk_id: int
    page_no: int = 0
    heading: str = ""
    score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source": self.source,
            "chunk_id": self.chunk_id,
            "page_no": self.page_no,
            "heading": self.heading,
            "score": self.score,
        }


@dataclass
class SearchResult:
    """A retrieved chunk with its score."""
    chunk: Chunk
    score: float
    rank: int


# ---------------------------------------------------------------------------
# SearchIndex
# ---------------------------------------------------------------------------

class SearchIndex:
    """
    Persistent hybrid (dense + sparse) search index.

    Usage:
        idx = build_index(["doc1.pdf", "doc2.docx"])
        results = idx.query("What is the revenue?", top_k=5)
        idx.save("./.docling_index")
        idx2 = SearchIndex.load("./.docling_index")
    """

    def __init__(
        self,
        chunks: list[Chunk],
        embeddings: Any,           # np.ndarray (n, dim)
        faiss_index: Any,          # faiss.Index
        bm25: Any,                 # BM25Okapi
        embed_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self._chunks = chunks
        self._embeddings = embeddings
        self._faiss = faiss_index
        self._bm25 = bm25
        self._embed_model = embed_model

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        *,
        top_k: int = 5,
        alpha: float = 0.5,   # 0 = full BM25, 1 = full dense
    ) -> list[Chunk]:
        """
        Hybrid retrieval: alpha * dense_score + (1-alpha) * bm25_score.

        Args:
            question: Natural language query.
            top_k:    Number of results to return.
            alpha:    Weight between dense (1.0) and sparse (0.0) retrieval.

        Returns:
            List of Chunk objects sorted by descending score.
        """
        n = len(self._chunks)
        if n == 0:
            return []

        dense_scores = self._dense_scores(question, top_k=min(n, top_k * 4))
        sparse_scores = self._sparse_scores(question, n=n)

        # Normalise both to [0, 1]
        combined: dict[int, float] = {}
        for idx, sc in dense_scores.items():
            combined[idx] = combined.get(idx, 0.0) + alpha * sc
        for idx, sc in sparse_scores.items():
            combined[idx] = combined.get(idx, 0.0) + (1.0 - alpha) * sc

        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results: list[Chunk] = []
        for rank, (idx, score) in enumerate(ranked):
            c = self._chunks[idx]
            c.score = score
            results.append(c)

        return results

    def _dense_scores(self, question: str, top_k: int) -> dict[int, float]:
        try:
            import numpy as np
            vec = self._embed_query(question)
            _, I, D = self._faiss.search(vec.reshape(1, -1), top_k)  # type: ignore
            scores = {}
            for idx, dist in zip(I[0], D[0]):
                if idx >= 0:
                    # Convert L2 distance to similarity in [0,1]
                    scores[int(idx)] = float(1.0 / (1.0 + dist))
            # Normalise
            if scores:
                mx = max(scores.values())
                scores = {k: v / mx for k, v in scores.items()}
            return scores
        except Exception:
            return {}

    def _sparse_scores(self, question: str, n: int) -> dict[int, float]:
        try:
            tokens = question.lower().split()
            raw = self._bm25.get_scores(tokens)
            mx = max(raw) if max(raw) > 0 else 1.0
            return {i: float(s / mx) for i, s in enumerate(raw) if s > 0}
        except Exception:
            return {}

    def _embed_query(self, text: str):
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(self._embed_model)
        import numpy as np
        return model.encode([text], show_progress_bar=False)[0].astype(np.float32)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save index to directory."""
        import faiss
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._faiss, str(p / "faiss.index"))
        with open(p / "bm25.pkl", "wb") as f:
            pickle.dump(self._bm25, f)
        meta = {
            "chunks": [c.to_dict() for c in self._chunks],
            "embed_model": self._embed_model,
        }
        (p / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "SearchIndex":
        """Load index from directory."""
        import faiss
        import numpy as np

        p = Path(path)
        faiss_index = faiss.read_index(str(p / "faiss.index"))
        with open(p / "bm25.pkl", "rb") as f:
            bm25 = pickle.load(f)
        meta = json.loads((p / "meta.json").read_text())
        chunks = [Chunk(**c) for c in meta["chunks"]]
        embed_model = meta.get("embed_model", "all-MiniLM-L6-v2")

        return cls(
            chunks=chunks,
            embeddings=None,
            faiss_index=faiss_index,
            bm25=bm25,
            embed_model=embed_model,
        )

    def __len__(self) -> int:
        return len(self._chunks)


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

def build_index(
    sources: list[str],
    *,
    embed_provider: EmbedProvider = "local",
    embed_model: str = "all-MiniLM-L6-v2",
    chunk_strategy: Literal["hierarchical", "hybrid", "token_window"] = "hierarchical",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    ocr: bool = True,
    workers: int = 4,
) -> SearchIndex:
    """
    Build a SearchIndex from a list of document sources.

    Args:
        sources:         File paths or URLs to index.
        embed_provider:  Embedding backend: "local" | "openai" | "anthropic".
        embed_model:     Model name (for "local" provider).
        chunk_strategy:  Chunking strategy.
        chunk_size:      Max tokens per chunk (token_window strategy).
        chunk_overlap:   Overlap between consecutive chunks.
        ocr:             Enable OCR during conversion.
        workers:         Parallel workers for document conversion.

    Returns:
        Populated SearchIndex ready for querying.
    """
    from docling_skill.analyze import analyze_batch

    results = analyze_batch(sources, ocr=ocr, workers=workers)
    all_chunks: list[Chunk] = []

    for result in results:
        chunks = _chunk_document(
            result,
            strategy=chunk_strategy,
            chunk_size=chunk_size,
            overlap=chunk_overlap,
        )
        all_chunks.extend(chunks)

    if not all_chunks:
        raise ValueError("No chunks produced - check that sources are valid documents.")

    texts = [c.text for c in all_chunks]
    embeddings = _embed_texts(texts, provider=embed_provider, model=embed_model)
    faiss_index = _build_faiss(embeddings)
    bm25 = _build_bm25(texts)

    return SearchIndex(
        chunks=all_chunks,
        embeddings=embeddings,
        faiss_index=faiss_index,
        bm25=bm25,
        embed_model=embed_model,
    )


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _chunk_document(
    result: Any,
    *,
    strategy: str,
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    source = result.source
    doc = result.document

    if doc is not None and strategy in ("hierarchical", "hybrid"):
        chunks = _hierarchical_chunk(doc, source)
        if chunks:
            return chunks

    # Fallback: token-window over markdown
    md = result.markdown or ""
    return _token_window_chunk(md, source=source, size=chunk_size, overlap=overlap)


def _hierarchical_chunk(doc: Any, source: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    try:
        from docling.chunking import HybridChunker
        chunker = HybridChunker()
        chunk_iter = chunker.chunk(doc)
        for i, ch in enumerate(chunk_iter):
            text = ch.text if hasattr(ch, "text") else str(ch)
            heading = ""
            try:
                heading = ch.meta.headings[0] if ch.meta.headings else ""
            except Exception:
                pass
            page_no = 0
            try:
                page_no = ch.meta.doc_items[0].prov[0].page_no
            except Exception:
                pass
            chunks.append(Chunk(
                text=text,
                source=source,
                chunk_id=i,
                page_no=page_no,
                heading=heading,
            ))
    except Exception:
        pass
    return chunks


def _token_window_chunk(
    text: str, *, source: str, size: int, overlap: int
) -> list[Chunk]:
    words = text.split()
    chunks: list[Chunk] = []
    i = 0
    idx = 0
    while i < len(words):
        window = words[i: i + size]
        chunks.append(Chunk(
            text=" ".join(window),
            source=source,
            chunk_id=idx,
        ))
        i += size - overlap
        idx += 1
    return chunks


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _embed_texts(
    texts: list[str],
    *,
    provider: EmbedProvider,
    model: str,
):
    import numpy as np

    if provider == "local":
        try:
            from sentence_transformers import SentenceTransformer
            m = SentenceTransformer(model)
            return m.encode(texts, show_progress_bar=True).astype(np.float32)
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers required: pip install sentence-transformers"
            ) from exc

    if provider == "openai":
        try:
            import openai
            client = openai.OpenAI()
            batch_size = 100
            all_embs = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i: i + batch_size]
                resp = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch,
                )
                all_embs.extend([d.embedding for d in resp.data])
            return np.array(all_embs, dtype=np.float32)
        except ImportError as exc:
            raise ImportError("openai package required: pip install openai") from exc

    raise ValueError(f"Unknown embed provider: {provider!r}")


# ---------------------------------------------------------------------------
# FAISS + BM25
# ---------------------------------------------------------------------------

def _build_faiss(embeddings):
    try:
        import faiss
        import numpy as np
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        return index
    except ImportError as exc:
        raise ImportError("faiss-cpu required: pip install faiss-cpu") from exc


def _build_bm25(texts: list[str]):
    try:
        from rank_bm25 import BM25Okapi
        tokenised = [t.lower().split() for t in texts]
        return BM25Okapi(tokenised)
    except ImportError as exc:
        raise ImportError("rank-bm25 required: pip install rank-bm25") from exc
