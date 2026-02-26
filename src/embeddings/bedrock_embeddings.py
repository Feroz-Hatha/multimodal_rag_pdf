"""AWS Bedrock embeddings using Amazon Titan Embed Text v2."""

import json
import logging
import time
from typing import Any

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# Titan Embed Text v2 limits
_MAX_INPUT_TOKENS = 8192
_MAX_INPUT_CHARS = _MAX_INPUT_TOKENS * 4  # ~32K chars (rough estimate)


class BedrockEmbeddings:
    """
    Generate embeddings using Amazon Titan Embed Text v2 via AWS Bedrock.

    Titan Embed Text v2 specs:
    - Max input: 8192 tokens
    - Output dimensions: 256, 512, or 1024 (1024 used for best quality)
    - Normalized embeddings (cosine similarity ready)
    """

    def __init__(
        self,
        model_id: str = "amazon.titan-embed-text-v2:0",
        region: str = "us-east-1",
        dimensions: int = 1024,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
    ):
        self.model_id = model_id
        self.dimensions = dimensions

        session_kwargs: dict[str, Any] = {"region_name": region}
        if aws_access_key_id and aws_secret_access_key:
            session_kwargs["aws_access_key_id"] = aws_access_key_id
            session_kwargs["aws_secret_access_key"] = aws_secret_access_key

        self.client = boto3.client("bedrock-runtime", **session_kwargs)

    def _truncate(self, text: str) -> str:
        if len(text) > _MAX_INPUT_CHARS:
            logger.warning(f"Text truncated from {len(text)} to {_MAX_INPUT_CHARS} chars")
            return text[:_MAX_INPUT_CHARS]
        return text

    def embed(self, text: str, max_retries: int = 3) -> list[float]:
        """
        Generate an embedding for a single text.

        Retries with exponential back-off on throttling errors.
        """
        text = self._truncate(text.strip())
        if not text:
            return [0.0] * self.dimensions

        body = json.dumps({
            "inputText": text,
            "dimensions": self.dimensions,
            "normalize": True,
        })

        for attempt in range(max_retries):
            try:
                response = self.client.invoke_model(
                    modelId=self.model_id,
                    body=body,
                    contentType="application/json",
                    accept="application/json",
                )
                return json.loads(response["body"].read())["embedding"]
            except ClientError as e:
                code = e.response["Error"]["Code"]
                if code == "ThrottlingException" and attempt < max_retries - 1:
                    wait = 2 ** attempt  # 1s, 2s, 4s
                    logger.warning(f"Throttled, retrying in {wait}s (attempt {attempt + 1})")
                    time.sleep(wait)
                else:
                    raise

        raise RuntimeError(f"Embedding failed after {max_retries} attempts")

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 20,
        delay_between_batches: float = 0.1,
        show_progress: bool = True,
    ) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.

        Bedrock has no native batch API, so this calls embed() per text.
        A small pause between batches avoids rate-limit errors.
        """
        embeddings = []

        try:
            from tqdm import tqdm
            iterator = tqdm(range(len(texts)), desc="Embedding", unit="chunk") if show_progress else range(len(texts))
        except ImportError:
            iterator = range(len(texts))

        for i in iterator:
            embeddings.append(self.embed(texts[i]))
            if (i + 1) % batch_size == 0 and i + 1 < len(texts):
                time.sleep(delay_between_batches)

        return embeddings
