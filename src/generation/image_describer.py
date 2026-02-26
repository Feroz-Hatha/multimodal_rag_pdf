"""Image description using Claude Vision via AWS Bedrock."""

import json
import logging
from typing import Any

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

_DESCRIPTION_PROMPT = """\
This image is from a document.

Describe it concisely but completely. Include:
- What type of visual it is (diagram, schematic, chart, graph, photograph, figure, table screenshot, etc.)
- What it depicts or shows
- Any visible text, labels, numbers, measurements, or data
- Its significance if apparent from context

Keep the description to 3-5 sentences. Be factual and precise."""


class ImageDescriber:
    """
    Generates text descriptions for images using Claude Vision via Bedrock.

    Each image content item in a ParsedDocument carries a base64-encoded PNG
    in its `image_data` field (populated by PDFParser when
    `generate_picture_images=True`). This class sends those images to Claude
    and returns natural-language descriptions suitable for embedding and retrieval.
    """

    def __init__(
        self,
        model_id: str = "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        region: str = "us-east-1",
        max_tokens: int = 400,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
    ):
        self.model_id = model_id
        self.max_tokens = max_tokens

        session_kwargs: dict[str, Any] = {"region_name": region}
        if aws_access_key_id and aws_secret_access_key:
            session_kwargs["aws_access_key_id"] = aws_access_key_id
            session_kwargs["aws_secret_access_key"] = aws_secret_access_key

        self.client = boto3.client("bedrock-runtime", **session_kwargs)

    def describe(self, image_b64: str, section_context: str = "") -> str:
        """
        Generate a description for a single base64-encoded PNG image.

        Args:
            image_b64: Base64-encoded PNG string.
            section_context: Section hierarchy string for additional context
                             (e.g. "Section 3 > 3.2 Test Procedures").

        Returns:
            Natural-language description of the image.
        """
        prompt = _DESCRIPTION_PROMPT
        if section_context:
            prompt = f"Section context: {section_context}\n\n{prompt}"

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }

        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
            result = json.loads(response["body"].read())
            return result["content"][0]["text"].strip()
        except ClientError as e:
            logger.warning(f"Image description failed: {e}")
            return ""

    def describe_document_images(
        self,
        content_items: list[dict],
        show_progress: bool = True,
    ) -> int:
        """
        Describe all image items in a document's content_items list in-place.

        For each item with type=="image" and a non-empty "image_data" field:
        - Calls Claude Vision to generate a description
        - Sets item["text"] to the description
        - Removes item["image_data"] to free memory

        Args:
            content_items: The list from ParsedDocument.content_items.
            show_progress: Show a tqdm progress bar.

        Returns:
            Number of images successfully described.
        """
        image_items = [
            item for item in content_items
            if item.get("type") == "image" and item.get("image_data")
        ]

        if not image_items:
            return 0

        try:
            from tqdm import tqdm
            iterator = tqdm(image_items, desc="Describing images", unit="img") if show_progress else image_items
        except ImportError:
            iterator = image_items

        described = 0
        for item in iterator:
            section = " > ".join(item.get("section_hierarchy", []))
            description = self.describe(
                image_b64=item["image_data"],
                section_context=section,
            )
            item["text"] = description
            item.pop("image_data")  # free memory immediately

            if description:
                described += 1
                logger.debug(f"Described image on page {item.get('page_numbers')}: {description[:80]}...")
            else:
                logger.warning(f"Empty description for image on page {item.get('page_numbers')}")

        return described
