"""LLM response generation using Claude via AWS Bedrock."""

import json
import logging
from dataclasses import dataclass
from typing import Any, Generator

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# System prompt for document Q&A
_DEFAULT_SYSTEM_PROMPT = """\
You are a precise assistant that answers questions based on provided document excerpts.

Answer the user's question using ONLY the information in the numbered context passages provided.
Follow these rules strictly:
- Cite sources inline using [1], [2], etc. matching the passage numbers
- Preserve all numbers, measurements, and data exactly as written
- If the context does not contain enough information to answer fully, say so clearly and state what is missing
- If passages contain conflicting information, note the discrepancy explicitly
- Do not add information from your own knowledge beyond what the context provides
"""


@dataclass
class GenerationResult:
    """Result from a single LLM call."""

    answer: str
    model_id: str
    input_tokens: int
    output_tokens: int


class ResponseGenerator:
    """
    Generates answers using Claude via AWS Bedrock.

    Builds a numbered context block from retrieved chunks, then calls
    Claude with a system prompt that enforces grounded, cited answers.
    """

    def __init__(
        self,
        model_id: str = "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        region: str = "us-east-1",
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
    ):
        self.model_id = model_id
        self.system_prompt = system_prompt

        session_kwargs: dict[str, Any] = {"region_name": region}
        if aws_access_key_id and aws_secret_access_key:
            session_kwargs["aws_access_key_id"] = aws_access_key_id
            session_kwargs["aws_secret_access_key"] = aws_secret_access_key

        self.client = boto3.client("bedrock-runtime", **session_kwargs)

    # ------------------------------------------------------------------
    # Context assembly
    # ------------------------------------------------------------------

    def build_context_block(self, chunks: list) -> str:
        """
        Format retrieved chunks into a numbered context block for the prompt.

        Each passage includes its source metadata so Claude can cite correctly.
        """
        if not chunks:
            return "No context passages available."

        parts = []
        for i, chunk in enumerate(chunks, 1):
            pages = ", ".join(f"p.{p}" for p in chunk.page_numbers) if chunk.page_numbers else "?"
            section = " > ".join(chunk.section_hierarchy) if chunk.section_hierarchy else chunk.heading or "—"

            header = f"[{i}] File: {chunk.filename} | Section: {section} | Pages: {pages}"
            parts.append(f"{header}\n{chunk.content}")

        return "\n\n".join(parts)

    def build_user_message(self, question: str, context_block: str) -> str:
        """Assemble the full user message with context + question."""
        return (
            f"Context passages:\n\n{context_block}\n\n"
            f"---\n\nQuestion: {question}"
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        question: str,
        context_chunks: list,
        max_tokens: int = 1024,
    ) -> GenerationResult:
        """
        Generate an answer grounded in the provided context chunks.

        Args:
            question: The user's question.
            context_chunks: List of RetrievedChunk objects from the retriever.
            max_tokens: Maximum tokens in the response.

        Returns:
            GenerationResult with answer text and token usage.
        """
        context_block = self.build_context_block(context_chunks)
        user_message = self.build_user_message(question, context_block)

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": self.system_prompt,
            "messages": [{"role": "user", "content": user_message}],
        }

        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
        except ClientError as e:
            code = e.response["Error"]["Code"]
            msg = str(e)
            if code == "AccessDeniedException":
                raise RuntimeError(
                    f"Bedrock access denied for model '{self.model_id}'. "
                    "Ensure model access is enabled in the Bedrock console under Model Access."
                ) from e
            if code == "ValidationException" and "inference profile" in msg:
                raise RuntimeError(
                    f"Model '{self.model_id}' requires a cross-region inference profile ID. "
                    "Use the format 'us.anthropic.claude-...' instead of 'anthropic.claude-...'"
                ) from e
            if code == "ResourceNotFoundException" and "use case" in msg:
                raise RuntimeError(
                    f"Anthropic use case details not submitted for this AWS account. "
                    "Go to Bedrock console → Model access → Claude → fill out the use case form. "
                    "Then wait ~15 minutes before retrying."
                ) from e
            raise

        result = json.loads(response["body"].read())
        return GenerationResult(
            answer=result["content"][0]["text"],
            model_id=self.model_id,
            input_tokens=result["usage"]["input_tokens"],
            output_tokens=result["usage"]["output_tokens"],
        )

    def generate_stream(
        self,
        question: str,
        context_chunks: list,
        max_tokens: int = 1024,
    ) -> Generator[str, None, None]:
        """
        Stream the answer token by token.

        Yields text deltas as they arrive from the model.
        Useful for Streamlit's st.write_stream() in Phase 4.
        """
        context_block = self.build_context_block(context_chunks)
        user_message = self.build_user_message(question, context_block)

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": self.system_prompt,
            "messages": [{"role": "user", "content": user_message}],
        }

        response = self.client.invoke_model_with_response_stream(
            modelId=self.model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )

        for event in response["body"]:
            chunk = json.loads(event["chunk"]["bytes"])
            if chunk.get("type") == "content_block_delta":
                delta = chunk.get("delta", {})
                if delta.get("type") == "text_delta":
                    yield delta.get("text", "")
