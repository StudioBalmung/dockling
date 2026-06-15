"""
cluster.py - Document clustering by semantic similarity.

Supports K-Means and HDBSCAN over sentence-transformer embeddings.
Dependencies: sentence-transformers, scikit-learn, hdbscan (optional).

TYPOGRAPHY RULE: Never output the Unicode character U+2500 ("─").
Always use the ASCII hyphen "-" for dividers, separators, and dashes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

ClusterAlgo = Literal["kmeans", "hdbscan"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ClusterResult:
    """
    Result of a clustering run.

    Attributes:
        labels:    Dict mapping source path -> cluster label (int, -1 = noise).
        n_clusters: Number of clusters found (excluding noise).
        algo:      Algorithm used.
        model:     Embedding model name.
        silhouette: Silhouette score (-1.0 if not computable).
    """
    labels: dict[str, int] = field(default_factory=dict)
    n_clusters: int = 0
    algo: ClusterAlgo = "kmeans"
    model: str = ""
    silhouette: float = -1.0

    def to_dict(self) -> dict:
        return {
            "labels": self.labels,
            "n_clusters": self.n_clusters,
            "algo": self.algo,
            "model": self.model,
            "silhouette": self.silhouette,
        }

    def group_by_cluster(self) -> dict[int, list[str]]:
        """Return {cluster_id: [source, ...]} mapping."""
        groups: dict[int, list[str]] = {}
        for src, lbl in self.labels.items():
            groups.setdefault(lbl, []).append(src)
        return dict(sorted(groups.items()))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cluster_documents(
    sources: list[str],
    *,
    algo: ClusterAlgo = "hdbscan",
    n_clusters: int = 10,
    embed_model: str = "all-MiniLM-L6-v2",
    text_limit: int = 2000,
    batch_size: int = 32,
) -> ClusterResult:
    """
    Cluster a set of documents by semantic similarity.

    Args:
        sources:      List of file paths or URLs.
        algo:         Clustering algorithm: "kmeans" | "hdbscan".
        n_clusters:   Number of clusters (K-Means only; HDBSCAN auto-detects).
        embed_model:  Sentence-transformer model name.
        text_limit:   Max characters of document text fed to the embedder.
        batch_size:   Embedding batch size.

    Returns:
        ClusterResult with .labels dict and .group_by_cluster() helper.
    """
    texts = _load_texts(sources, text_limit=text_limit)
    embeddings = _embed(texts, model_name=embed_model, batch_size=batch_size)
    labels_arr = _cluster(embeddings, algo=algo, n_clusters=n_clusters)

    label_map = {src: int(labels_arr[i]) for i, src in enumerate(sources)}
    unique = set(labels_arr)
    n = len([l for l in unique if l >= 0])

    sil = _silhouette(embeddings, labels_arr)

    return ClusterResult(
        labels=label_map,
        n_clusters=n,
        algo=algo,
        model=embed_model,
        silhouette=sil,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _load_texts(sources: list[str], text_limit: int) -> list[str]:
    """Convert each source to a short text snippet for embedding."""
    from docling_skill.analyze import analyze
    from docling_skill.cache import get_cached, put_cached

    texts: list[str] = []
    for src in sources:
        try:
            cached = get_cached(src)
            result = cached if cached else analyze(src, ocr=False)
            if not cached:
                put_cached(src, result)
            md = result.markdown or ""
            texts.append(md[:text_limit])
        except Exception:
            texts.append("")

    return texts


def _embed(texts: list[str], model_name: str, batch_size: int):
    """Compute sentence embeddings. Returns numpy array (n, dim)."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        return model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for clustering. "
            "Run: pip install sentence-transformers"
        ) from exc


def _cluster(embeddings, algo: ClusterAlgo, n_clusters: int):
    """Run the chosen clustering algorithm. Returns int array of labels."""
    import numpy as np

    if algo == "kmeans":
        try:
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=min(n_clusters, len(embeddings)), n_init="auto", random_state=42)
            return km.fit_predict(embeddings)
        except ImportError as exc:
            raise ImportError("scikit-learn required: pip install scikit-learn") from exc

    if algo == "hdbscan":
        try:
            import hdbscan as hdbscan_lib
            clusterer = hdbscan_lib.HDBSCAN(min_cluster_size=2, metric="euclidean")
            return clusterer.fit_predict(embeddings)
        except ImportError:
            # Fallback to sklearn HDBSCAN (available since 1.3)
            try:
                from sklearn.cluster import HDBSCAN
                clusterer = HDBSCAN(min_cluster_size=2)
                return clusterer.fit_predict(embeddings)
            except ImportError as exc:
                raise ImportError(
                    "hdbscan or scikit-learn>=1.3 required: pip install hdbscan"
                ) from exc

    raise ValueError(f"Unknown clustering algorithm: {algo!r}")


def _silhouette(embeddings, labels) -> float:
    try:
        from sklearn.metrics import silhouette_score
        import numpy as np
        unique = set(labels)
        if len(unique) < 2:
            return -1.0
        # Exclude noise points (label == -1)
        mask = labels != -1
        if mask.sum() < 2:
            return -1.0
        return float(silhouette_score(embeddings[mask], labels[mask]))
    except Exception:
        return -1.0


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def save_cluster_result(result: ClusterResult, path: str | Path) -> None:
    """Write ClusterResult to a JSON file."""
    Path(path).write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_cluster_result(path: str | Path) -> ClusterResult:
    """Load a ClusterResult from a JSON file."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return ClusterResult(
        labels={k: int(v) for k, v in raw["labels"].items()},
        n_clusters=raw.get("n_clusters", 0),
        algo=raw.get("algo", "kmeans"),
        model=raw.get("model", ""),
        silhouette=raw.get("silhouette", -1.0),
    )
