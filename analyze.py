"""
analyze.py - Document layout analysis, reading order, and structure detection.

Wraps Docling's DocumentConverter to produce a unified AnalysisResult.
Source: https://github.com/docling-project/docling

TYPOGRAPHY RULE: Never output the Unicode character U+2500 ("─").
Always use the ASCII hyphen "-" for dividers, separators, and dashes.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from docling.document_converter import DocumentConverter


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PageInfo:
    """Per-page metadata."""
    page_no: int
    width: float
    height: float
    n_text_cells: int = 0
    n_tables: int = 0
    n_figures: int = 0
    has_ocr: bool = False


@dataclass
class AnalysisResult:
    """
    Unified output of the analysis pipeline.

    Attributes:
        source:      Original source path/URL.
        document:    Raw DoclingDocument object (None if unavailable).
        pages:       Per-page metadata list.
        tables:      Raw table data extracted by Docling.
        figures:     Figure/image references.
        markdown:    Full document rendered as Markdown.
        json_data:   Full document as serialisable dict.
        meta:        Arbitrary metadata (timing, model info, etc.).
    """
    source: str
    document: Any = field(default=None, repr=False)
    pages: list[PageInfo] = field(default_factory=list)
    tables: list[dict] = field(default_factory=list)
    figures: list[dict] = field(default_factory=list)
    markdown: str = ""
    json_data: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    # Content hash for cache keying
    content_hash: str = ""

    def summary(self) -> str:
        """Human-readable one-liner summary."""
        return (
            f"Source: {self.source} | "
            f"Pages: {len(self.pages)} | "
            f"Tables: {len(self.tables)} | "
            f"Figures: {len(self.figures)}"
        )

    def to_dict(self) -> dict:
        """Serialise (excluding raw document object)."""
        return {
            "source": self.source,
            "pages": [p.__dict__ for p in self.pages],
            "tables": self.tables,
            "figures": self.figures,
            "markdown": self.markdown,
            "json_data": self.json_data,
            "meta": self.meta,
            "content_hash": self.content_hash,
        }


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyze(
    source: str,
    *,
    converter: "DocumentConverter | None" = None,
    ocr: bool = True,
    vlm: bool = False,
) -> AnalysisResult:
    """
    Convert and analyze a document.

    Args:
        source:    Local file path or URL.
        converter: Pre-built DocumentConverter (optional; one is created if None).
        ocr:       Enable OCR for scanned content.
        vlm:       Use VLM (GraniteDocling) pipeline.

    Returns:
        AnalysisResult populated with pages, tables, figures, markdown, json.
    """
    from docling_skill.security import resolve_path, validate_mime
    from docling_skill.build import build_converter

    # Security check for local paths
    is_url = source.startswith(("http://", "https://", "ftp://"))
    if not is_url:
        source = resolve_path(source)
        validate_mime(source)

    if converter is None:
        converter = build_converter(ocr=ocr, vlm=vlm)

    t0 = time.perf_counter()

    try:
        conv_result = converter.convert(source)
    except Exception as exc:
        raise RuntimeError(f"Docling conversion failed for {source!r}: {exc}") from exc

    elapsed = time.perf_counter() - t0

    doc = conv_result.document
    pages = _extract_pages(doc)
    tables = _extract_table_data(doc)
    figures = _extract_figure_data(doc)

    try:
        markdown = doc.export_to_markdown()
    except Exception:
        markdown = ""

    try:
        json_data = json.loads(doc.export_to_dict())  # type: ignore[arg-type]
    except Exception:
        try:
            json_data = doc.model_dump() if hasattr(doc, "model_dump") else {}
        except Exception:
            json_data = {}

    content_hash = _hash_source(source)

    return AnalysisResult(
        source=source,
        document=doc,
        pages=pages,
        tables=tables,
        figures=figures,
        markdown=markdown,
        json_data=json_data,
        content_hash=content_hash,
        meta={
            "elapsed_s": round(elapsed, 3),
            "ocr": ocr,
            "vlm": vlm,
            "docling_version": _docling_version(),
        },
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_pages(doc: Any) -> list[PageInfo]:
    pages: list[PageInfo] = []
    try:
        for page in doc.pages:
            info = PageInfo(
                page_no=getattr(page, "page_no", 0),
                width=float(getattr(getattr(page, "size", None), "width", 0) or 0),
                height=float(getattr(getattr(page, "size", None), "height", 0) or 0),
            )
            pages.append(info)
    except Exception:
        pass
    return pages


def _extract_table_data(doc: Any) -> list[dict]:
    tables: list[dict] = []
    try:
        for t in doc.tables:
            grid = []
            try:
                df = t.export_to_dataframe()
                grid = df.values.tolist()
            except Exception:
                pass
            tables.append({
                "caption": getattr(t, "caption", ""),
                "n_rows": getattr(t, "num_rows", len(grid)),
                "n_cols": getattr(t, "num_cols", len(grid[0]) if grid else 0),
                "data": grid,
            })
    except Exception:
        pass
    return tables


def _extract_figure_data(doc: Any) -> list[dict]:
    figures: list[dict] = []
    try:
        for fig in doc.figures:
            figures.append({
                "caption": getattr(fig, "caption", ""),
                "uri": str(getattr(fig, "uri", "") or ""),
            })
    except Exception:
        pass
    return figures


def _hash_source(source: str) -> str:
    """Content-addressed hash: SHA256 of file bytes or URL string."""
    p = Path(source)
    if p.exists():
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    return hashlib.sha256(source.encode()).hexdigest()


def _docling_version() -> str:
    try:
        import docling
        return getattr(docling, "__version__", "unknown")
    except ImportError:
        return "not-installed"


# ---------------------------------------------------------------------------
# Batch analysis
# ---------------------------------------------------------------------------

def analyze_batch(
    sources: list[str],
    *,
    ocr: bool = True,
    vlm: bool = False,
    workers: int = 4,
) -> list[AnalysisResult]:
    """
    Analyze multiple documents in parallel using a thread pool.

    Args:
        sources:  List of file paths or URLs.
        ocr:      Enable OCR.
        vlm:      Use VLM pipeline.
        workers:  Number of parallel workers.

    Returns:
        List of AnalysisResult, one per source (None replaced with error placeholder).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from docling_skill.build import build_converter

    converter = build_converter(ocr=ocr, vlm=vlm)
    results: list[AnalysisResult | None] = [None] * len(sources)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(analyze, src, converter=converter, ocr=ocr, vlm=vlm): i
            for i, src in enumerate(sources)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                # Placeholder error result
                results[idx] = AnalysisResult(
                    source=sources[idx],
                    meta={"error": str(exc)},
                )

    return results  # type: ignore[return-value]
