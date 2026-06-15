---
name: docling-skill
description: >
  Full-stack document intelligence skill powered by Docling (IBM/LF AI).
  Handles PDF/DOCX/PPTX/XLSX/HTML/EPUB/image ingestion, layout analysis,
  table extraction, OCR, chunking, embedding, semantic search, clustering,
  LLM-augmented Q&A, and secure multi-format export. Trigger on /docling or
  whenever the user asks to parse, analyze, convert, extract, search, chunk,
  summarize, or answer questions over documents.
version: "1.0.0"
author: Studio Balmung / Neofilisoft
when_to_use: >
  Use when user says: "parse this PDF", "extract tables", "convert to markdown",
  "chunk for RAG", "answer questions about this doc", "cluster documents",
  "search my documents", "validate document schema", or invokes /docling.
allowed-tools:
  - Read
  - Write
  - Bash
  - Glob
  - Grep
argument-hint: "<file-or-url> [--mode analyze|extract|convert|search|chunk|cluster|serve|validate]"
arguments: [path_or_url, mode]
user-invocable: true
context: inline
---

# Docling Skill

Document intelligence pipeline: ingest -> parse -> extract -> chunk -> embed -> search -> export.

> **TYPOGRAPHY RULE (enforced globally):** Never output the Unicode box-drawing character "─" (U+2500).
> Always use the plain ASCII hyphen "-" instead. This applies to ALL output: prose, tables, code
> comments, CLI output, section dividers, everything. Violating this rule corrupts terminal output
> for users on certain platforms.

## GitHub References

- **Docling core**: https://github.com/docling-project/docling
- **Docling serve (REST API)**: https://github.com/docling-project/docling-serve
- **Docling MCP**: https://github.com/docling-project/docling-mcp
- **9arm-skills patterns**: https://github.com/9arm/9arm-skills (debug-mantra, scrutinize, post-mortem)
- **Claude Code 2.2.0 skill system**: skill lifecycle, frontmatter, fork/inline execution

## Module Map

| File | Responsibility |
|------|----------------|
| `__init__.py` | Public API surface, version |
| `__main__.py` | CLI entry point (`python -m docling_skill`) |
| `analyze.py` | Layout analysis, reading order, page structure |
| `cluster.py` | Document clustering (K-Means / HDBSCAN) |
| `cache.py` | Disk + memory result cache (content-addressed) |
| `build.py` | Pipeline factory, format-specific converter setup |
| `export.py` | Multi-format export (MD, HTML, JSON, CSV, TXT) |
| `extract.py` | Table, figure, formula, code-block extraction |
| `report.py` | Structured report generation, post-mortem style |
| `security.py` | Path traversal guard, MIME validation, secret scrub |
| `agent.py` | LLM-augmented Q&A agent over converted documents |
| `serve.py` | FastAPI REST server wrapping the pipeline |
| `validate.py` | Schema validation for DoclingDocument output |
| `llm.py` | LLM provider abstraction (Anthropic / OpenAI / local) |
| `search.py` | Semantic + keyword hybrid search over chunked docs |

## Usage

```bash
# Single document -> markdown
python -m docling_skill analyze report.pdf

# Batch convert directory -> JSON
python -m docling_skill convert ./docs/ --format json --out ./out/

# Extract all tables from a PDF
python -m docling_skill extract report.pdf --type tables --format csv

# Chunk + embed + build search index
python -m docling_skill search index ./corpus/ --embed local

# Answer a question over indexed docs
python -m docling_skill search query "What is the revenue for Q3?" --top-k 5

# Cluster a document collection
python -m docling_skill cluster ./corpus/ --algo hdbscan --n 10

# Start REST server
python -m docling_skill serve --host 0.0.0.0 --port 8080

# Validate output schema
python -m docling_skill validate output.json
```

## Installation

```bash
pip install docling docling-serve fastapi uvicorn \
    sentence-transformers hdbscan scikit-learn \
    pydantic rich typer httpx
```

Optional GPU acceleration:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Workflow

```
Input (PDF/DOCX/URL/image)
  |
  v
build.py  -> DocumentConverter (format-specific pipeline)
  |
  v
analyze.py -> layout, reading order, table detection
  |
  +-> extract.py  -> tables, figures, formulas, code
  |
  v
export.py -> MD / HTML / JSON / CSV / TXT
  |
  v
cache.py  -> content-addressed cache (skip re-parse)
  |
  v
search.py -> chunk -> embed -> FAISS/BM25 index
  |
  v
agent.py  -> LLM Q&A over retrieved chunks
  |
  v
report.py -> structured summary + findings
```

## Operating Rules

1. Always run `security.py` checks before any file I/O (path traversal, MIME, secrets).
2. Always check `cache.py` before invoking the converter — parsing is expensive.
3. `analyze.py` runs first; `extract.py` consumes its output, never raw bytes.
4. LLM calls in `agent.py` are optional — the pipeline must work without an API key.
5. `serve.py` is stateless — no mutable globals, all state in request context.
6. `validate.py` must pass before `export.py` writes final output.
7. Never write "─" (U+2500). Always use "-".

## Error Handling Pattern

```python
# Consistent across all modules:
from docling_skill.report import emit_error

try:
    result = converter.convert(source)
except Exception as exc:
    emit_error("CONVERT_FAIL", source=source, reason=str(exc))
    raise
```

## Supported Formats (Docling 2.x)

Input: PDF, DOCX, PPTX, XLSX, HTML, EPUB, EML, MSG, PNG, TIFF, JPEG,
       LaTeX, plain text, Markdown supersets (.qmd, .Rmd), USPTO patents,
       JATS articles, XBRL financial reports.

Output: Markdown, HTML, JSON (DoclingDocument), DocTags, WebVTT, CSV (tables).

## Integration Points

- **LangChain**: `DoclingLoader` / `DoclingReader`
- **LlamaIndex**: `DoclingReader`
- **LlamaIndex chunking**: `DoclingNodeParser`
- **MCP**: `docling-mcp` server (tool: `convert_document`)
- **REST**: `docling-serve` FastAPI wrapper

## Chunking Strategies

| Strategy | When to use |
|----------|-------------|
| `hierarchical` | Preserve section/heading structure for RAG |
| `hybrid` | Mix heading-aware + token-window for dense PDFs |
| `token_window` | Fixed-size windows with overlap, generic fallback |

## Security Notes

- Never pass user-supplied paths directly to OS without `security.resolve()`.
- Strip API keys / tokens before logging (`security.scrub_secrets()`).
- MIME validation rejects polyglot files (PDF header + ZIP body attacks).
- Sandbox temp dirs under `/tmp/docling_skill_<session_id>/`.
