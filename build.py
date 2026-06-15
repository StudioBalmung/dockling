"""
build.py - DocumentConverter factory with format-specific pipeline configuration.

Source: https://github.com/docling-project/docling
Ref: docling.document_converter.DocumentConverter

TYPOGRAPHY RULE: Never output the Unicode character U+2500 ("─").
Always use the ASCII hyphen "-" for dividers, separators, and dashes.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Converter factory
# ---------------------------------------------------------------------------

def build_converter(
    *,
    ocr: bool = True,
    vlm: bool = False,
    table_structure: bool = True,
    enrich_code: bool = True,
    enrich_formulas: bool = True,
    allowed_formats: list[str] | None = None,
) -> Any:
    """
    Build and return a configured Docling DocumentConverter.

    Args:
        ocr:              Enable EasyOCR / Tesseract for scanned pages.
        vlm:              Use VLM pipeline (GraniteDocling 258M) for complex layouts.
        table_structure:  Run table structure recognition (TableFormer).
        enrich_code:      Enable code block detection and enrichment.
        enrich_formulas:  Enable formula detection and enrichment.
        allowed_formats:  Restrict accepted input formats. None = accept all.

    Returns:
        docling.document_converter.DocumentConverter instance.

    Raises:
        ImportError: If docling is not installed.

    Example:
        converter = build_converter(ocr=True, vlm=False)
        result = converter.convert("report.pdf")
    """
    try:
        from docling.document_converter import DocumentConverter, ConversionOptions
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            EasyOcrOptions,
            TableFormerMode,
        )
    except ImportError as exc:
        raise ImportError(
            "docling is not installed. Run: pip install docling"
        ) from exc

    # ---------------------------------------------------------------------------
    # PDF pipeline options
    # ---------------------------------------------------------------------------
    pdf_opts = PdfPipelineOptions()
    pdf_opts.do_ocr = ocr
    pdf_opts.do_table_structure = table_structure

    if table_structure:
        pdf_opts.table_structure_options.mode = TableFormerMode.ACCURATE

    if ocr:
        pdf_opts.ocr_options = EasyOcrOptions(force_full_page_ocr=False)

    # ---------------------------------------------------------------------------
    # VLM pipeline (GraniteDocling)
    # ---------------------------------------------------------------------------
    if vlm:
        try:
            from docling.pipeline.vlm_pipeline import VlmPipelineOptions
            from docling.models.base_vlm_model import BaseVlmOptions
            vlm_opts = VlmPipelineOptions()
            pdf_opts.vlm_options = vlm_opts
            pdf_opts.generate_page_images = True  # needed by VLM
        except ImportError:
            # VLM deps not installed; fall back silently
            vlm = False

    # ---------------------------------------------------------------------------
    # Code + formula enrichment
    # ---------------------------------------------------------------------------
    if enrich_code:
        try:
            pdf_opts.do_code_enrichment = True
        except AttributeError:
            pass  # older docling version

    if enrich_formulas:
        try:
            pdf_opts.do_formula_enrichment = True
        except AttributeError:
            pass

    # ---------------------------------------------------------------------------
    # Format map
    # ---------------------------------------------------------------------------
    try:
        from docling.document_converter import (
            PdfFormatOption,
            WordFormatOption,
            PowerpointFormatOption,
            ExcelFormatOption,
            HtmlFormatOption,
            MarkdownFormatOption,
        )
        from docling.datamodel.base_models import InputFormat

        format_options: dict[Any, Any] = {
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts),
            InputFormat.DOCX: WordFormatOption(),
            InputFormat.PPTX: PowerpointFormatOption(),
            InputFormat.XLSX: ExcelFormatOption(),
            InputFormat.HTML: HtmlFormatOption(),
            InputFormat.MD: MarkdownFormatOption(),
        }

        if allowed_formats:
            _name_map = {fmt.name.lower(): fmt for fmt in InputFormat}
            allowed_enum = [_name_map[f.lower()] for f in allowed_formats if f.lower() in _name_map]
            format_options = {k: v for k, v in format_options.items() if k in allowed_enum}

        converter = DocumentConverter(format_options=format_options)

    except Exception:
        # Fallback: minimal constructor (older docling API)
        converter = DocumentConverter()

    return converter


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def build_ocr_only_converter() -> Any:
    """Converter tuned for fully scanned / image PDFs."""
    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            EasyOcrOptions,
        )
        from docling.datamodel.base_models import InputFormat

        opts = PdfPipelineOptions()
        opts.do_ocr = True
        opts.ocr_options = EasyOcrOptions(force_full_page_ocr=True)
        opts.do_table_structure = True

        return DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )
    except Exception:
        from docling.document_converter import DocumentConverter
        return DocumentConverter()


def build_fast_converter() -> Any:
    """Converter with minimal enrichment for bulk/batch jobs where speed matters."""
    return build_converter(
        ocr=False,
        vlm=False,
        table_structure=False,
        enrich_code=False,
        enrich_formulas=False,
    )


def build_vlm_converter() -> Any:
    """Converter using the GraniteDocling VLM for maximum layout accuracy."""
    return build_converter(
        ocr=True,
        vlm=True,
        table_structure=True,
        enrich_code=True,
        enrich_formulas=True,
    )
