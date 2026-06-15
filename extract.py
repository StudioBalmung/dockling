"""
extract.py - Structured element extraction: tables, figures, formulas, code blocks.

Consumes the output of analyze.py (AnalysisResult / DoclingDocument).
Source: https://github.com/docling-project/docling

TYPOGRAPHY RULE: Never output the Unicode character U+2500 ("─").
Always use the ASCII hyphen "-" for dividers, separators, and dashes.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from typing import Any, Literal

ExtractKind = Literal["tables", "figures", "formulas", "code", "all"]
ExportFmt = Literal["csv", "json", "md", "txt"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ExtractedTable:
    index: int
    caption: str
    n_rows: int
    n_cols: int
    data: list[list[Any]] = field(default_factory=list)
    page_no: int = 0

    def to_csv(self) -> str:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerows(self.data)
        return buf.getvalue()

    def to_markdown(self) -> str:
        if not self.data:
            return ""
        header = self.data[0]
        sep = ["---"] * len(header)
        rows = self.data[1:]
        lines = [
            "| " + " | ".join(str(c) for c in header) + " |",
            "| " + " | ".join(sep) + " |",
        ]
        for row in rows:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        caption = f"\n*{self.caption}*" if self.caption else ""
        return "\n".join(lines) + caption


@dataclass
class ExtractedFigure:
    index: int
    caption: str
    uri: str
    page_no: int = 0
    mime_type: str = ""


@dataclass
class ExtractedFormula:
    index: int
    latex: str
    text: str
    page_no: int = 0


@dataclass
class ExtractedCode:
    index: int
    language: str
    code: str
    page_no: int = 0


@dataclass
class ExtractionResult:
    """
    Unified extraction result returned to the caller.

    Attributes:
        source: Original document source.
        kind:   What was extracted ("tables", "figures", etc.).
        tables, figures, formulas, code: typed element lists.
        text:   Pre-rendered text output in the requested format.
    """
    source: str
    kind: ExtractKind
    fmt: ExportFmt = "json"
    tables: list[ExtractedTable] = field(default_factory=list)
    figures: list[ExtractedFigure] = field(default_factory=list)
    formulas: list[ExtractedFormula] = field(default_factory=list)
    code: list[ExtractedCode] = field(default_factory=list)
    text: str = ""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract(
    source: str,
    *,
    kind: ExtractKind = "tables",
    fmt: ExportFmt = "csv",
    ocr: bool = True,
) -> ExtractionResult:
    """
    Extract structured elements from a document.

    Args:
        source: Local file path or URL.
        kind:   Element type: "tables" | "figures" | "formulas" | "code" | "all".
        fmt:    Output format: "csv" | "json" | "md" | "txt".
        ocr:    Enable OCR during conversion.

    Returns:
        ExtractionResult with typed element lists and pre-rendered .text.
    """
    from docling_skill.analyze import analyze

    result = analyze(source, ocr=ocr)
    return extract_from_result(result, kind=kind, fmt=fmt)


def extract_from_result(
    result: Any,  # AnalysisResult
    *,
    kind: ExtractKind = "tables",
    fmt: ExportFmt = "csv",
) -> ExtractionResult:
    """
    Extract elements from an already-analysed AnalysisResult.
    Avoids re-converting the document.
    """
    doc = result.document
    out = ExtractionResult(source=result.source, kind=kind, fmt=fmt)

    if kind in ("tables", "all"):
        out.tables = _extract_tables(doc)

    if kind in ("figures", "all"):
        out.figures = _extract_figures(doc)

    if kind in ("formulas", "all"):
        out.formulas = _extract_formulas(doc)

    if kind in ("code", "all"):
        out.code = _extract_code(doc)

    out.text = _render(out, fmt)
    return out


# ---------------------------------------------------------------------------
# Element extractors
# ---------------------------------------------------------------------------

def _extract_tables(doc: Any) -> list[ExtractedTable]:
    tables: list[ExtractedTable] = []
    if doc is None:
        return tables

    try:
        for i, t in enumerate(doc.tables):
            data: list[list[Any]] = []
            try:
                df = t.export_to_dataframe()
                # header row + data rows
                data = [list(df.columns)] + df.values.tolist()
            except Exception:
                try:
                    data = t.data.grid  # type: ignore
                except Exception:
                    pass

            tables.append(ExtractedTable(
                index=i,
                caption=str(getattr(t, "caption", "") or ""),
                n_rows=len(data),
                n_cols=len(data[0]) if data else 0,
                data=data,
                page_no=_page_no(t),
            ))
    except Exception:
        pass

    return tables


def _extract_figures(doc: Any) -> list[ExtractedFigure]:
    figures: list[ExtractedFigure] = []
    if doc is None:
        return figures

    try:
        for i, fig in enumerate(doc.figures):
            figures.append(ExtractedFigure(
                index=i,
                caption=str(getattr(fig, "caption", "") or ""),
                uri=str(getattr(fig, "uri", "") or ""),
                page_no=_page_no(fig),
                mime_type=str(getattr(fig, "mime_type", "") or ""),
            ))
    except Exception:
        pass

    return figures


def _extract_formulas(doc: Any) -> list[ExtractedFormula]:
    formulas: list[ExtractedFormula] = []
    if doc is None:
        return formulas

    try:
        # Docling 2.x: doc.body.children walk
        for i, item in enumerate(_walk_items(doc, "formula")):
            formulas.append(ExtractedFormula(
                index=i,
                latex=str(getattr(item, "text", "") or ""),
                text=str(getattr(item, "orig", "") or getattr(item, "text", "") or ""),
                page_no=_page_no(item),
            ))
    except Exception:
        pass

    return formulas


def _extract_code(doc: Any) -> list[ExtractedCode]:
    code_blocks: list[ExtractedCode] = []
    if doc is None:
        return code_blocks

    try:
        for i, item in enumerate(_walk_items(doc, "code")):
            code_blocks.append(ExtractedCode(
                index=i,
                language=str(getattr(item, "code_language", "") or ""),
                code=str(getattr(item, "text", "") or ""),
                page_no=_page_no(item),
            ))
    except Exception:
        pass

    return code_blocks


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------

def _render(out: ExtractionResult, fmt: ExportFmt) -> str:
    if out.kind in ("tables", "all") and out.tables:
        return _render_tables(out.tables, fmt)
    if out.kind == "figures":
        return _render_figures(out.figures, fmt)
    if out.kind == "formulas":
        return _render_formulas(out.formulas, fmt)
    if out.kind == "code":
        return _render_code(out.code, fmt)

    # "all" - combine everything
    parts: list[str] = []
    if out.tables:
        parts.append(_render_tables(out.tables, fmt))
    if out.figures:
        parts.append(_render_figures(out.figures, fmt))
    if out.formulas:
        parts.append(_render_formulas(out.formulas, fmt))
    if out.code:
        parts.append(_render_code(out.code, fmt))
    return "\n\n".join(parts)


def _render_tables(tables: list[ExtractedTable], fmt: ExportFmt) -> str:
    if fmt == "csv":
        parts = []
        for t in tables:
            if t.caption:
                parts.append(f"# Table {t.index}: {t.caption}")
            parts.append(t.to_csv())
        return "\n".join(parts)
    if fmt == "md":
        return "\n\n".join(t.to_markdown() for t in tables)
    if fmt == "json":
        return json.dumps([
            {"index": t.index, "caption": t.caption,
             "n_rows": t.n_rows, "n_cols": t.n_cols, "data": t.data}
            for t in tables
        ], ensure_ascii=False, indent=2)
    # txt
    return "\n\n".join(
        f"Table {t.index} ({t.n_rows}x{t.n_cols}): {t.caption}" for t in tables
    )


def _render_figures(figures: list[ExtractedFigure], fmt: ExportFmt) -> str:
    if fmt == "json":
        return json.dumps([f.__dict__ for f in figures], ensure_ascii=False, indent=2)
    if fmt == "md":
        return "\n".join(
            f"![{fig.caption}]({fig.uri})" for fig in figures
        )
    return "\n".join(
        f"Figure {fig.index}: {fig.caption} ({fig.uri})" for fig in figures
    )


def _render_formulas(formulas: list[ExtractedFormula], fmt: ExportFmt) -> str:
    if fmt == "json":
        return json.dumps([f.__dict__ for f in formulas], ensure_ascii=False, indent=2)
    if fmt == "md":
        return "\n".join(f"$${f.latex}$$" for f in formulas)
    return "\n".join(f"Formula {f.index}: {f.latex}" for f in formulas)


def _render_code(code_blocks: list[ExtractedCode], fmt: ExportFmt) -> str:
    if fmt == "json":
        return json.dumps([c.__dict__ for c in code_blocks], ensure_ascii=False, indent=2)
    if fmt == "md":
        return "\n\n".join(
            f"```{c.language}\n{c.code}\n```" for c in code_blocks
        )
    return "\n\n".join(c.code for c in code_blocks)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _page_no(item: Any) -> int:
    try:
        return int(item.prov[0].page_no)
    except Exception:
        return 0


def _walk_items(doc: Any, label_lower: str):
    """Walk DoclingDocument items looking for a specific label type."""
    try:
        for item, _ in doc.iterate_items():
            lbl = str(getattr(item, "label", "") or "").lower()
            if label_lower in lbl:
                yield item
    except Exception:
        return
