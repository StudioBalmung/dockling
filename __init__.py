"""
docling_skill - Document intelligence skill for Claude / LLM agents.

Public API surface. Import from here; never import sub-modules directly
in external code -- internal structure may change between versions.

TYPOGRAPHY RULE: Never output the Unicode character U+2500 ("─").
Always use the ASCII hyphen "-" for dividers, separators, and dashes.
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__ = "Studio Balmung / Neofilisoft"
__license__ = "MIT"

# ---------------------------------------------------------------------------
# Lazy imports: heavy deps (docling, torch, sentence-transformers) are NOT
# loaded until actually used.  This keeps `import docling_skill` fast.
# ---------------------------------------------------------------------------

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from docling_skill.analyze import AnalysisResult
    from docling_skill.extract import ExtractionResult
    from docling_skill.search import SearchIndex, SearchResult
    from docling_skill.agent import AgentResponse
    from docling_skill.report import Report


def get_version() -> str:
    """Return the package version string."""
    return __version__


def pipeline(
    source: str,
    *,
    cache: bool = True,
    ocr: bool = True,
    vlm: bool = False,
) -> "AnalysisResult":
    """
    High-level entry point: convert + analyze a document.

    Args:
        source: Local file path or URL.
        cache:  Use on-disk cache (skip re-parse if unchanged).
        ocr:    Enable OCR for scanned pages / images.
        vlm:    Use Visual Language Model pipeline (GraniteDocling).
               Slower but better on complex layouts.

    Returns:
        AnalysisResult with .document (DoclingDocument), .pages, .tables,
        .figures, .markdown, .json properties.

    Example:
        result = docling_skill.pipeline("report.pdf")
        print(result.markdown)
    """
    from docling_skill.build import build_converter
    from docling_skill.analyze import analyze
    from docling_skill.cache import get_cached, put_cached

    if cache:
        cached = get_cached(source)
        if cached is not None:
            return cached

    converter = build_converter(ocr=ocr, vlm=vlm)
    result = analyze(source, converter=converter)

    if cache:
        put_cached(source, result)

    return result


def answer(
    question: str,
    *,
    sources: list[str],
    top_k: int = 5,
    provider: str = "anthropic",
) -> "AgentResponse":
    """
    Answer a question over one or more documents.

    Args:
        question: Natural language question.
        sources:  List of file paths or URLs to search.
        top_k:    Number of chunks to retrieve and include as context.
        provider: LLM provider -- "anthropic" | "openai" | "local".

    Returns:
        AgentResponse with .answer, .sources, .chunks, .confidence.
    """
    from docling_skill.search import build_index
    from docling_skill.agent import run_agent

    index = build_index(sources)
    results = index.query(question, top_k=top_k)
    return run_agent(question, context_chunks=results, provider=provider)


__all__ = [
    "__version__",
    "get_version",
    "pipeline",
    "answer",
]
