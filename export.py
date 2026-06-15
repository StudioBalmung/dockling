"""
export.py - Multi-format export: Markdown, HTML, JSON, CSV, plain text.

Consumes AnalysisResult objects from analyze.py and writes files.
Source: https://github.com/docling-project/docling

TYPOGRAPHY RULE: Never output the Unicode character U+2500 ("─").
Always use the ASCII hyphen "-" for dividers, separators, and dashes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

ExportFmt = Literal["md", "html", "json", "csv", "txt", "doctags"]


# ---------------------------------------------------------------------------
# Single-document export
# ---------------------------------------------------------------------------

def export_document(
    result: Any,  # AnalysisResult
    *,
    fmt: ExportFmt = "md",
    out_path: str | Path | None = None,
) -> str:
    """
    Export an AnalysisResult to the requested format.

    Args:
        result:   AnalysisResult from analyze.analyze().
        fmt:      Target format: "md" | "html" | "json" | "csv" | "txt" | "doctags".
        out_path: If given, write output to this file and return the path string.
                  If None, return the rendered string.

    Returns:
        Rendered content string, or out_path string when a file was written.
    """
    content = _render(result, fmt)

    if out_path is not None:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return str(p)

    return content


def _render(result: Any, fmt: ExportFmt) -> str:
    doc = result.document

    if fmt == "md":
        return _to_markdown(result, doc)
    if fmt == "html":
        return _to_html(result, doc)
    if fmt == "json":
        return _to_json(result, doc)
    if fmt == "csv":
        return _to_csv(result)
    if fmt == "txt":
        return _to_txt(result, doc)
    if fmt == "doctags":
        return _to_doctags(doc)

    raise ValueError(f"Unsupported export format: {fmt!r}")


def _to_markdown(result: Any, doc: Any) -> str:
    if doc is not None:
        try:
            return doc.export_to_markdown()
        except Exception:
            pass
    return result.markdown or ""


def _to_html(result: Any, doc: Any) -> str:
    if doc is not None:
        try:
            return doc.export_to_html()
        except Exception:
            pass
    md = _to_markdown(result, doc)
    # Minimal fallback
    escaped = md.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"<html><body><pre>{escaped}</pre></body></html>"


def _to_json(result: Any, doc: Any) -> str:
    if doc is not None:
        try:
            d = doc.export_to_dict()
            if isinstance(d, str):
                return d
            return json.dumps(d, ensure_ascii=False, indent=2)
        except Exception:
            pass
    return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)


def _to_csv(result: Any) -> str:
    """Export all tables as CSV blocks separated by blank lines."""
    import csv
    import io

    parts: list[str] = []
    for i, tbl in enumerate(result.tables):
        buf = io.StringIO()
        w = csv.writer(buf)
        caption = tbl.get("caption", "")
        if caption:
            parts.append(f"# Table {i}: {caption}")
        data = tbl.get("data", [])
        for row in data:
            w.writerow(row)
        parts.append(buf.getvalue())

    if not parts:
        # No tables - fall back to plain text
        return _to_txt(result, result.document)

    return "\n".join(parts)


def _to_txt(result: Any, doc: Any) -> str:
    md = _to_markdown(result, doc)
    # Strip markdown syntax for plain text
    import re
    txt = re.sub(r"#{1,6}\s*", "", md)          # headings
    txt = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", txt)  # bold/italic
    txt = re.sub(r"`{1,3}[^`]*`{1,3}", "", txt, flags=re.DOTALL)  # code
    txt = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", txt)  # images
    txt = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", txt)  # links
    txt = re.sub(r"\|[^\n]+\|", "", txt)        # tables
    txt = re.sub(r"\n{3,}", "\n\n", txt)        # excess blank lines
    return txt.strip()


def _to_doctags(doc: Any) -> str:
    if doc is not None:
        try:
            return doc.export_to_doctags()
        except Exception:
            pass
    return ""


# ---------------------------------------------------------------------------
# Batch export
# ---------------------------------------------------------------------------

def batch_export(
    sources: list[str | Path],
    *,
    fmt: ExportFmt = "md",
    out_dir: str | Path = "./out",
    ocr: bool = True,
    workers: int = 4,
    skip_errors: bool = True,
) -> list[str]:
    """
    Convert and export multiple documents in parallel.

    Args:
        sources:     List of file paths or URLs.
        fmt:         Target export format.
        out_dir:     Directory where exported files are written.
        ocr:         Enable OCR.
        workers:     Thread pool size.
        skip_errors: If True, log errors and continue; if False, re-raise.

    Returns:
        List of output file paths that were successfully written.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from docling_skill.analyze import analyze
    from docling_skill.build import build_converter

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    converter = build_converter(ocr=ocr)
    written: list[str] = []

    def _process(src: str | Path) -> str | None:
        try:
            result = analyze(str(src), converter=converter, ocr=ocr)
            stem = Path(str(src)).stem
            ext = _fmt_ext(fmt)
            out_path = out_dir / f"{stem}{ext}"
            return export_document(result, fmt=fmt, out_path=out_path)
        except Exception as exc:
            if skip_errors:
                print(f"[WARN] Failed to export {src}: {exc}")
                return None
            raise

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process, src): src for src in sources}
        for future in as_completed(futures):
            path = future.result()
            if path:
                written.append(path)

    return written


def _fmt_ext(fmt: ExportFmt) -> str:
    return {
        "md": ".md",
        "html": ".html",
        "json": ".json",
        "csv": ".csv",
        "txt": ".txt",
        "doctags": ".doctags",
    }.get(fmt, ".txt")


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def detect_best_format(source: str) -> ExportFmt:
    """
    Heuristic: recommend an export format based on the source file type.

    - PDF/image/EPUB -> "md" (richest text output)
    - XLSX -> "csv"
    - PPTX/DOCX -> "html"
    - default -> "md"
    """
    ext = Path(source).suffix.lower()
    if ext in (".xlsx", ".xls", ".csv"):
        return "csv"
    if ext in (".pptx", ".ppt", ".docx", ".doc"):
        return "html"
    return "md"
