#!/usr/bin/env python3
"""
Index sample PDFs with Bedrock embeddings and run test queries.

Usage:
    python scripts/test_indexing.py           # skip already-indexed docs
    python scripts/test_indexing.py --force   # re-index everything (picks up image descriptions)

Prerequisites:
    - AWS credentials in .env (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION)
    - Bedrock access enabled for:
        amazon.titan-embed-text-v2:0               (embeddings)
        us.anthropic.claude-3-5-sonnet-20241022-v2:0  (image descriptions)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.embeddings.bedrock_embeddings import BedrockEmbeddings
from src.ingestion.indexing_pipeline import IndexingPipeline
from src.retrieval.retriever import Retriever
from src.vectordb.chroma_client import ChromaVectorDB

console = Console()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-index even if already indexed")
    args = parser.parse_args()

    # --- Initialise shared components (embeddings client is reused for both
    #     indexing and retrieval to avoid creating multiple boto3 sessions) ---
    console.print("[bold]Initialising components...[/]")
    embeddings = BedrockEmbeddings()
    vectordb = ChromaVectorDB()
    pipeline = IndexingPipeline(embeddings=embeddings, vectordb=vectordb)  # describe_images=True by default
    retriever = Retriever(embeddings=embeddings, vectordb=vectordb)

    # --- Index sample PDFs ---
    sample_dir = Path("tests/sample_pdfs")
    pdfs = sorted(sample_dir.glob("*.pdf"))
    console.print(f"\n[bold]Indexing {len(pdfs)} PDFs (force={args.force})...[/]\n")

    results = pipeline.index_batch(pdfs, force_reindex=args.force)

    t = Table(title="Indexing Results", show_lines=True)
    t.add_column("File", max_width=45)
    t.add_column("Status")
    t.add_column("Chunks", justify="right")
    t.add_column("Text", justify="right")
    t.add_column("Tables", justify="right")
    t.add_column("Images", justify="right")
    t.add_column("Time (s)", justify="right")

    for r in results:
        if r.skipped:
            status, color = "skipped", "yellow"
        elif r.error:
            status, color = f"error: {r.error}", "red"
        else:
            status, color = "indexed", "green"

        t.add_row(
            r.filename[:45],
            f"[{color}]{status}[/]",
            str(r.total_chunks),
            str(r.text_chunks),
            str(r.table_chunks),
            str(r.image_chunks),
            f"{r.processing_time_seconds:.1f}" if not r.skipped else "—",
        )

    console.print(t)

    stats = vectordb.get_stats()
    console.print(f"\n[bold]ChromaDB:[/] {stats['total_chunks']} total chunks stored\n")

    # --- Test queries ---
    test_queries = [
        ("General", "What are the requirements for seat belt webbing?"),
        ("Specific", "What is the minimum elongation requirement?"),
        ("Table", "Show me tables about load or force requirements"),
        ("Location", "What happens at maximum webbing extension?"),
    ]

    console.print("[bold]Running test queries...[/]\n")
    for label, query in test_queries:
        console.print(Panel(f"[bold cyan]{query}[/]", title=f"[{label}] Query"))
        chunks = retriever.retrieve(query, n_results=3)

        if not chunks:
            console.print("  [red]No results[/]\n")
            continue

        for i, chunk in enumerate(chunks, 1):
            console.print(
                f"  [dim]#{i}[/] score=[bold]{chunk.score:.3f}[/] | "
                f"[italic]{chunk.format_citation()}[/]"
            )
            preview = chunk.content[:250].replace("\n", " ")
            console.print(f"  {preview}\n")


if __name__ == "__main__":
    main()
