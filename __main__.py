"""
CLI entry point for docling_skill.

Usage:
    python -m docling_skill <command> [args]

TYPOGRAPHY RULE: Never output the Unicode character U+2500 ("─").
Always use the ASCII hyphen "-" for dividers, separators, and dashes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="docling-skill",
    help="Document intelligence pipeline powered by Docling.",
    add_completion=False,
)
console = Console()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _err(msg: str) -> None:
    console.print(f"[bold red]ERROR:[/bold red] {msg}", file=sys.stderr)
    raise typer.Exit(1)


def _ok(msg: str) -> None:
    console.print(f"[bold green]OK[/bold green] {msg}")


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------

@app.command("analyze")
def cmd_analyze(
    source: str = typer.Argument(..., help="File path or URL."),
    ocr: bool = typer.Option(True, help="Enable OCR."),
    vlm: bool = typer.Option(False, help="Use VLM pipeline (GraniteDocling)."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Skip cache."),
    out: Path = typer.Option(None, "--out", "-o", help="Write markdown to file."),
) -> None:
    """Parse and analyze a document, print or save as Markdown."""
    from docling_skill import pipeline

    with console.status(f"Analyzing {source} ..."):
        result = pipeline(source, cache=not no_cache, ocr=ocr, vlm=vlm)

    if out:
        out.write_text(result.markdown, encoding="utf-8")
        _ok(f"Written to {out}")
    else:
        console.print(result.markdown)


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------

@app.command("extract")
def cmd_extract(
    source: str = typer.Argument(..., help="File path or URL."),
    type_: str = typer.Option("tables", "--type", "-t",
                               help="tables | figures | formulas | code"),
    format_: str = typer.Option("csv", "--format", "-f",
                                help="csv | json | md"),
    out: Path = typer.Option(None, "--out", "-o"),
) -> None:
    """Extract structured elements (tables, figures, formulas, code) from a document."""
    from docling_skill.extract import extract

    with console.status(f"Extracting {type_} from {source} ..."):
        result = extract(source, kind=type_, fmt=format_)

    if out:
        out.write_text(result.text, encoding="utf-8")
        _ok(f"Written to {out}")
    else:
        console.print(result.text)


# ---------------------------------------------------------------------------
# convert
# ---------------------------------------------------------------------------

@app.command("convert")
def cmd_convert(
    source: str = typer.Argument(..., help="File, directory, or URL."),
    format_: str = typer.Option("md", "--format", "-f",
                                help="md | html | json | csv | txt"),
    out: Path = typer.Option(Path("./out"), "--out", "-o",
                             help="Output directory."),
    recursive: bool = typer.Option(False, "--recursive", "-r"),
) -> None:
    """Batch-convert documents to a target format."""
    from docling_skill.export import batch_export

    out.mkdir(parents=True, exist_ok=True)
    paths = []

    src = Path(source)
    if src.is_dir():
        pat = "**/*" if recursive else "*"
        paths = [p for p in src.glob(pat) if p.is_file()]
    else:
        paths = [src]

    with console.status(f"Converting {len(paths)} file(s) -> {format_} ..."):
        exported = batch_export(paths, fmt=format_, out_dir=out)

    _ok(f"Exported {len(exported)} file(s) to {out}")


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

@app.command("index")
def cmd_index(
    directory: Path = typer.Argument(..., help="Directory to index."),
    embed: str = typer.Option("local", help="local | openai | anthropic"),
    index_path: Path = typer.Option(Path(".docling_index"), "--index"),
) -> None:
    """Build a semantic search index from a document corpus."""
    from docling_skill.search import build_index

    files = list(directory.rglob("*"))
    files = [f for f in files if f.is_file()]
    console.print(f"Indexing {len(files)} file(s) ...")
    idx = build_index([str(f) for f in files], embed_provider=embed)
    idx.save(str(index_path))
    _ok(f"Index saved to {index_path}")


@app.command("query")
def cmd_query(
    question: str = typer.Argument(..., help="Natural language question."),
    index_path: Path = typer.Option(Path(".docling_index"), "--index"),
    top_k: int = typer.Option(5, "--top-k", "-k"),
    provider: str = typer.Option("anthropic", "--provider", "-p"),
) -> None:
    """Query an index with a question and get an LLM-generated answer."""
    from docling_skill.search import SearchIndex
    from docling_skill.agent import run_agent

    idx = SearchIndex.load(str(index_path))
    chunks = idx.query(question, top_k=top_k)
    response = run_agent(question, context_chunks=chunks, provider=provider)

    console.rule("-" * 40)
    console.print(f"[bold]Answer:[/bold]\n{response.answer}")
    console.rule("-" * 40)
    console.print("[dim]Sources:[/dim]")
    for src in response.sources:
        console.print(f"  - {src}")


# ---------------------------------------------------------------------------
# cluster
# ---------------------------------------------------------------------------

@app.command("cluster")
def cmd_cluster(
    directory: Path = typer.Argument(...),
    algo: str = typer.Option("hdbscan", help="hdbscan | kmeans"),
    n: int = typer.Option(10, "--n", help="Number of clusters (K-Means only)."),
    out: Path = typer.Option(Path("clusters.json"), "--out"),
) -> None:
    """Cluster a document corpus by semantic similarity."""
    from docling_skill.cluster import cluster_documents
    import json

    files = [str(f) for f in directory.rglob("*") if Path(f).is_file()]
    with console.status(f"Clustering {len(files)} docs with {algo} ..."):
        labels = cluster_documents(files, algo=algo, n_clusters=n)

    out.write_text(json.dumps(labels, indent=2, ensure_ascii=False))
    _ok(f"Cluster map written to {out}")


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------

@app.command("serve")
def cmd_serve(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8080, "--port"),
    workers: int = typer.Option(1, "--workers"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """Start the Docling REST API server."""
    import uvicorn

    uvicorn.run(
        "docling_skill.serve:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        log_level="info",
    )


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

@app.command("validate")
def cmd_validate(
    path: Path = typer.Argument(..., help="DoclingDocument JSON file."),
) -> None:
    """Validate a DoclingDocument JSON against the schema."""
    from docling_skill.validate import validate_file

    errors = validate_file(str(path))
    if errors:
        for e in errors:
            console.print(f"[red]-[/red] {e}")
        _err(f"Validation failed with {len(errors)} error(s).")
    else:
        _ok(f"{path} is valid.")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

@app.command("report")
def cmd_report(
    source: str = typer.Argument(..., help="File path or URL."),
    out: Path = typer.Option(None, "--out"),
) -> None:
    """Generate a structured analysis report for a document."""
    from docling_skill.report import generate_report

    with console.status("Generating report ..."):
        rpt = generate_report(source)

    text = rpt.to_markdown()
    if out:
        out.write_text(text, encoding="utf-8")
        _ok(f"Report written to {out}")
    else:
        console.print(text)


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------

@app.command("version")
def cmd_version() -> None:
    """Print version."""
    from docling_skill import __version__
    console.print(f"docling-skill {__version__}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
