#!/usr/bin/env python3
"""
Phase 3 test: run end-to-end RAG queries against indexed documents.

Prerequisites:
  - Phase 2 completed: PDFs indexed in ChromaDB (run scripts/test_indexing.py first)
  - .env populated with AWS credentials and Bedrock access enabled for:
      amazon.titan-embed-text-v2:0   (embeddings)
      anthropic.claude-3-5-sonnet-20241022-v2:0  (LLM)

Usage:
    source py312_pdf_rag/bin/activate
    python scripts/test_rag.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from src.embeddings.bedrock_embeddings import BedrockEmbeddings
from src.generation.rag_pipeline import RAGPipeline
from src.vectordb.chroma_client import ChromaVectorDB

console = Console()

TEST_QUERIES = [
    {
        "label": "Requirement lookup",
        "question": "What is the minimum breaking strength requirement for seat belt webbing?",
    },
    {
        "label": "Technical spec",
        "question": "What is the elongation requirement, and how is it measured?",
    },
    {
        "label": "Table query",
        "question": "What load or force values are specified in any tables?",
        "content_type": "table",
    },
    {
        "label": "Out-of-context (should say so)",
        "question": "What are the emission standards for diesel engines?",
    },
]


def main():
    console.print(Rule("[bold]Phase 3 — RAG Query Test[/]"))

    # --- Check ChromaDB has data ---
    vectordb = ChromaVectorDB()
    stats = vectordb.get_stats()
    if stats["total_chunks"] == 0:
        console.print(
            "[red]ChromaDB is empty. Run scripts/test_indexing.py first to index documents.[/]"
        )
        sys.exit(1)
    console.print(f"[dim]ChromaDB: {stats['total_chunks']} chunks indexed[/]\n")

    # --- Build pipeline (shared embeddings client) ---
    embeddings = BedrockEmbeddings()
    pipeline = RAGPipeline(embeddings=embeddings, vectordb=vectordb)

    # --- Run queries ---
    total_cost = 0.0

    for q in TEST_QUERIES:
        console.print(Rule(f"[cyan]{q['label']}[/]"))
        console.print(f"[bold]Q:[/] {q['question']}\n")

        response = pipeline.query(
            question=q["question"],
            n_results=5,
            content_type=q.get("content_type"),
        )

        # Answer
        console.print(Panel(response.answer, title="Answer", border_style="green"))

        # Sources
        console.print("[bold]Sources:[/]")
        console.print(response.format_sources())

        # Usage
        cost = response.estimated_cost_usd()
        total_cost += cost
        console.print(
            f"\n[dim]Tokens: {response.input_tokens} in / {response.output_tokens} out "
            f"| Est. cost: ${cost:.5f}[/]\n"
        )

    console.print(Rule())
    console.print(f"[bold]Total estimated cost:[/] ${total_cost:.4f}")


if __name__ == "__main__":
    main()
