#!/usr/bin/env python3
"""Test and compare chunking strategies."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.ingestion.pdf_parser import PDFParser
from src.ingestion.chunking.hierarchical import HierarchicalChunker
from src.ingestion.chunking.semantic import SemanticChunker

console = Console()


def compare_chunkers(pdf_path: str | Path, chunk_size: int = 1000):
    """Compare hierarchical and semantic chunking on a PDF."""
    pdf_path = Path(pdf_path)
    console.print(f"\n[bold blue]Analyzing:[/] {pdf_path.name}\n")

    # Parse document
    parser = PDFParser(ocr_enabled=False)
    with console.status("[bold green]Parsing PDF..."):
        doc = parser.parse(pdf_path)

    console.print(f"Document: {doc.metadata.total_pages} pages, {len(doc.content_items)} items\n")

    # Initialize chunkers
    hierarchical = HierarchicalChunker(
        chunk_size=chunk_size,
        chunk_overlap=200,
        min_chunk_size=100,
    )
    semantic = SemanticChunker(
        chunk_size=chunk_size,
        chunk_overlap=200,
        min_chunk_size=100,
        similarity_threshold=0.5,
    )

    # Chunk with hierarchical
    console.print("[bold]Hierarchical Chunking...[/]")
    hier_result = hierarchical.chunk(doc)

    # Chunk with semantic
    console.print("[bold]Semantic Chunking...[/]")
    with console.status("[bold green]Computing embeddings..."):
        sem_result = semantic.chunk(doc)

    # Compare results
    console.print(Panel("[bold]Comparison Results[/]"))

    comparison = Table(title="Chunking Strategy Comparison")
    comparison.add_column("Metric", style="cyan")
    comparison.add_column("Hierarchical", justify="right")
    comparison.add_column("Semantic", justify="right")

    comparison.add_row("Total Chunks", str(hier_result.total_chunks), str(sem_result.total_chunks))
    comparison.add_row("Avg Chunk Size", f"{hier_result.avg_chunk_size:.0f}", f"{sem_result.avg_chunk_size:.0f}")
    comparison.add_row("Min Chunk Size", str(hier_result.min_chunk_size), str(sem_result.min_chunk_size))
    comparison.add_row("Max Chunk Size", str(hier_result.max_chunk_size), str(sem_result.max_chunk_size))
    comparison.add_row("Text Chunks", str(hier_result.text_chunks), str(sem_result.text_chunks))
    comparison.add_row("Table Chunks", str(hier_result.table_chunks), str(sem_result.table_chunks))
    comparison.add_row("Image Chunks", str(hier_result.image_chunks), str(sem_result.image_chunks))

    console.print(comparison)

    # Show sample chunks from each
    console.print("\n[bold]Sample Hierarchical Chunks:[/]")
    for i, chunk in enumerate(hier_result.chunks[:3]):
        console.print(Panel(
            f"[dim]Pages: {chunk.metadata.page_numbers} | Type: {chunk.metadata.content_type.value}[/]\n"
            f"[dim]Section: {chunk.metadata.get_context_prefix() or 'N/A'}[/]\n\n"
            f"{chunk.content[:300]}{'...' if len(chunk.content) > 300 else ''}",
            title=f"Chunk {i+1} ({chunk.metadata.char_count} chars)"
        ))

    console.print("\n[bold]Sample Semantic Chunks:[/]")
    for i, chunk in enumerate(sem_result.chunks[:3]):
        console.print(Panel(
            f"[dim]Pages: {chunk.metadata.page_numbers} | Type: {chunk.metadata.content_type.value}[/]\n"
            f"[dim]Section: {chunk.metadata.get_context_prefix() or 'N/A'}[/]\n\n"
            f"{chunk.content[:300]}{'...' if len(chunk.content) > 300 else ''}",
            title=f"Chunk {i+1} ({chunk.metadata.char_count} chars)"
        ))

    return hier_result, sem_result


def main():
    # Get sample PDFs
    sample_dir = Path(__file__).parent.parent / "tests" / "sample_pdfs"
    pdf_files = list(sample_dir.glob("*.pdf"))

    if not pdf_files:
        console.print("[red]No PDF files found in tests/sample_pdfs/[/]")
        return

    console.print(f"[bold]Found {len(pdf_files)} PDF files[/]")

    # Test with the first PDF
    pdf_path = pdf_files[0]
    hier_result, sem_result = compare_chunkers(pdf_path)

    # Summary
    console.print("\n[bold green]Chunking test completed![/]")
    console.print(f"\nHierarchical: {hier_result.total_chunks} chunks")
    console.print(f"Semantic: {sem_result.total_chunks} chunks")


if __name__ == "__main__":
    main()
