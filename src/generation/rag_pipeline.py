"""End-to-end RAG query pipeline: question → retrieve → generate → respond."""

import logging
from dataclasses import dataclass, field

from src.embeddings.bedrock_embeddings import BedrockEmbeddings
from src.generation.response_generator import GenerationResult, ResponseGenerator
from src.retrieval.retriever import RetrievedChunk, Retriever
from src.vectordb.chroma_client import ChromaVectorDB

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """
    The complete output of a RAG query.

    Contains the generated answer, the source chunks it was grounded in,
    and token usage for cost tracking.
    """

    question: str
    answer: str
    sources: list[RetrievedChunk]
    model_id: str
    input_tokens: int
    output_tokens: int

    def format_sources(self) -> str:
        """Format a numbered source list matching the inline [N] citations."""
        if not self.sources:
            return "No sources."
        lines = []
        for i, chunk in enumerate(self.sources, 1):
            lines.append(f"[{i}] {chunk.format_citation()}")
        return "\n".join(lines)

    def estimated_cost_usd(self) -> float:
        """
        Rough cost estimate for the LLM call (Claude 3.5 Sonnet pricing).

        Prices as of 2024: $3 / 1M input tokens, $15 / 1M output tokens.
        This is an approximation — check AWS pricing for current rates.
        """
        input_cost = (self.input_tokens / 1_000_000) * 3.0
        output_cost = (self.output_tokens / 1_000_000) * 15.0
        return input_cost + output_cost


class RAGPipeline:
    """
    Orchestrates the full RAG query flow:

    1. Embed the question (Bedrock Titan Embed v2)
    2. Retrieve top-k relevant chunks (ChromaDB)
    3. Assemble numbered context block
    4. Generate a grounded, cited answer (Claude via Bedrock)
    5. Return RAGResponse with answer + sources + usage

    Components are injected so they can be shared (e.g. a single
    BedrockEmbeddings instance across retrieval and generation).
    """

    def __init__(
        self,
        retriever: Retriever | None = None,
        generator: ResponseGenerator | None = None,
        # Shared components (used when retriever/generator are not pre-built)
        embeddings: BedrockEmbeddings | None = None,
        vectordb: ChromaVectorDB | None = None,
    ):
        # Build shared components if not provided
        _embeddings = embeddings or BedrockEmbeddings()
        _vectordb = vectordb or ChromaVectorDB()

        self.retriever = retriever or Retriever(embeddings=_embeddings, vectordb=_vectordb)
        self.generator = generator or ResponseGenerator()

    def query(
        self,
        question: str,
        n_results: int = 5,
        document_id: str | None = None,
        document_ids: list[str] | None = None,
        content_type: str | None = None,
        min_score: float = 0.0,
        max_tokens: int = 1024,
    ) -> RAGResponse:
        """
        Run a full RAG query.

        Args:
            question: Natural-language question from the user.
            n_results: Number of chunks to retrieve.
            document_id: Restrict retrieval to a specific document.
            document_ids: Restrict retrieval to a list of documents.
            content_type: Filter retrieval by type ("text", "table", etc.).
            min_score: Minimum cosine similarity for retrieved chunks.
            max_tokens: Maximum tokens in the generated answer.

        Returns:
            RAGResponse with the answer, sources, and token usage.
        """
        # Step 1 & 2: retrieve relevant chunks
        logger.info(f"Retrieving {n_results} chunks for: {question!r}")
        chunks = self.retriever.retrieve(
            query=question,
            n_results=n_results,
            document_id=document_id,
            document_ids=document_ids,
            content_type=content_type,
            min_score=min_score,
        )

        if not chunks:
            logger.warning("No chunks retrieved — returning empty-context answer")
            return RAGResponse(
                question=question,
                answer="I could not find any relevant information in the indexed documents.",
                sources=[],
                model_id=self.generator.model_id,
                input_tokens=0,
                output_tokens=0,
            )

        logger.info(
            f"Retrieved {len(chunks)} chunks "
            f"(scores: {chunks[0].score:.3f} – {chunks[-1].score:.3f})"
        )

        # Step 3 & 4: generate grounded answer
        result: GenerationResult = self.generator.generate(
            question=question,
            context_chunks=chunks,
            max_tokens=max_tokens,
        )

        logger.info(
            f"Generated answer — {result.input_tokens} in / "
            f"{result.output_tokens} out tokens"
        )

        return RAGResponse(
            question=question,
            answer=result.answer,
            sources=chunks,
            model_id=result.model_id,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )

    def query_stream(
        self,
        question: str,
        n_results: int = 5,
        document_id: str | None = None,
        document_ids: list[str] | None = None,
        content_type: str | None = None,
        min_score: float = 0.0,
        max_tokens: int = 1024,
    ):
        """
        Streaming variant — retrieves chunks synchronously, then streams
        the answer token by token.

        Yields:
            str deltas while generating, then a final RAGResponse object
            (as the last yielded item) so the caller can access sources.

        Usage:
            chunks_seen = None
            for item in pipeline.query_stream(question):
                if isinstance(item, RAGResponse):
                    response = item   # final object with sources
                else:
                    print(item, end="", flush=True)   # stream token
        """
        chunks = self.retriever.retrieve(
            query=question,
            n_results=n_results,
            document_id=document_id,
            document_ids=document_ids,
            content_type=content_type,
            min_score=min_score,
        )

        if not chunks:
            yield RAGResponse(
                question=question,
                answer="I could not find any relevant information in the indexed documents.",
                sources=[],
                model_id=self.generator.model_id,
                input_tokens=0,
                output_tokens=0,
            )
            return

        full_answer = []
        for delta in self.generator.generate_stream(
            question=question,
            context_chunks=chunks,
            max_tokens=max_tokens,
        ):
            full_answer.append(delta)
            yield delta

        # Yield the final structured response so caller can access sources
        yield RAGResponse(
            question=question,
            answer="".join(full_answer),
            sources=chunks,
            model_id=self.generator.model_id,
            input_tokens=0,   # not available from streaming API
            output_tokens=0,
        )
