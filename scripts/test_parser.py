#!/usr/bin/env python3
"""Test script for PDF parser."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

from src.ingestion.pdf_parser import PDFParser

console = Console()


def test_parser(pdf_path: str | Path):
    """Test the PDF parser on a single file."""
    pdf_path = Path(pdf_path)
    console.print(f"\n[bold blue]Testing PDF Parser on:[/] {pdf_path.name}\n")

    # Initialize parser
    parser = PDFParser(
        extract_tables=True,
        extract_images=True,
        ocr_enabled=False,  # Disable OCR for faster testing
    )

    # Parse document
    with console.status("[bold green]Parsing PDF..."):
        doc = parser.parse(pdf_path)

    # Display metadata
    console.print(Panel("[bold]Document Metadata[/]"))
    meta_table = Table(show_header=False, box=None)
    meta_table.add_column("Field", style="cyan")
    meta_table.add_column("Value", style="white")
    meta_table.add_row("Document ID", str(doc.metadata.document_id))
    meta_table.add_row("Filename", doc.metadata.filename)
    meta_table.add_row("Title", doc.metadata.title or "N/A")
    meta_table.add_row("Total Pages", str(doc.metadata.total_pages))
    meta_table.add_row("File Size", f"{doc.metadata.file_size_bytes / 1024:.1f} KB")
    meta_table.add_row("File Hash", doc.metadata.file_hash[:16] + "...")
    meta_table.add_row("Processing Time", f"{doc.metadata.processing_time_seconds:.2f}s")
    console.print(meta_table)

    # Display sections
    console.print(Panel(f"[bold]Sections Found: {len(doc.sections)}[/]"))
    if doc.sections:
        for i, section in enumerate(doc.sections[:15]):  # Show first 15
            indent = "  " * (section["level"] - 1)
            heading_preview = section['heading'][:60] + "..." if len(section['heading']) > 60 else section['heading']
            console.print(f"{indent}[dim]L{section['level']}[/] {heading_preview} [dim](p.{section['page_number']})[/]")
        if len(doc.sections) > 15:
            console.print(f"  ... and {len(doc.sections) - 15} more sections")

    # Display content items summary
    console.print(Panel(f"[bold]Content Items: {len(doc.content_items)}[/]"))
    item_types = {}
    for item in doc.content_items:
        item_type = item["type"]
        item_types[item_type] = item_types.get(item_type, 0) + 1
    for item_type, count in sorted(item_types.items()):
        console.print(f"  {item_type}: {count}")

    # Display tables
    console.print(Panel(f"[bold]Tables Found: {len(doc.tables)}[/]"))
    if doc.tables:
        for i, table in enumerate(doc.tables[:3]):  # Show first 3
            console.print(f"\n[cyan]Table {i+1}[/] (Pages: {table['page_numbers']})")
            if table["caption"]:
                console.print(f"Caption: {table['caption']}")
            # Show preview of markdown
            md_preview = table["markdown"][:500] + "..." if len(table["markdown"]) > 500 else table["markdown"]
            console.print(md_preview)
        if len(doc.tables) > 3:
            console.print(f"\n... and {len(doc.tables) - 3} more tables")

    # Display images
    console.print(Panel(f"[bold]Images Found: {len(doc.images)}[/]"))
    if doc.images:
        for i, image in enumerate(doc.images[:5]):
            console.print(f"  Image {i+1}: Pages {image['page_numbers']}, Caption: {image['caption'] or 'N/A'}")
        if len(doc.images) > 5:
            console.print(f"  ... and {len(doc.images) - 5} more images")

    # Display markdown content preview
    console.print(Panel("[bold]Markdown Content Preview (first 2000 chars)[/]"))
    content_preview = doc.markdown_content[:2000]
    console.print(content_preview)
    console.print(f"\n[dim]... Total length: {len(doc.markdown_content)} characters[/]")

    return doc


def main():
    # Get sample PDFs
    sample_dir = Path(__file__).parent.parent / "tests" / "sample_pdfs"
    # pdf_files = list(sample_dir.glob("*.pdf"))
    pdf_files = [sample_dir / "CFR-2023-title49-vol6-sec571-209.pdf"]

    if not pdf_files:
        console.print("[red]No PDF files found in tests/sample_pdfs/[/]")
        return

    console.print(f"[bold]Found {len(pdf_files)} PDF files:[/]")
    for i, pdf in enumerate(pdf_files, 1):
        console.print(f"  {i}. {pdf.name}")

    # Test with the first PDF (or you can specify which one)
    console.print("\n[bold]Testing with first PDF file...[/]")
    doc = test_parser(pdf_files[0])

    console.print("\n[bold green]Parser test completed successfully![/]")


if __name__ == "__main__":
    main()
