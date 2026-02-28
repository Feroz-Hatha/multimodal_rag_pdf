"""PDF parsing using Docling."""

import hashlib
import logging
import time
from pathlib import Path
from typing import Any
from uuid import UUID

import base64
import io
import re

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import (
    DoclingDocument,
    TableItem,
    PictureItem,
    SectionHeaderItem,
    TextItem,
    ListItem,
)

from src.ingestion.metadata import DocumentMetadata, ParsedDocument

logger = logging.getLogger(__name__)


class PDFParser:
    """Parse PDF documents using Docling."""

    def __init__(
        self,
        extract_tables: bool = True,
        extract_images: bool = True,
        ocr_enabled: bool = True,
    ):
        """
        Initialize the PDF parser.

        Args:
            extract_tables: Whether to extract tables
            extract_images: Whether to extract images
            ocr_enabled: Whether to enable OCR for scanned content
        """
        self.extract_tables = extract_tables
        self.extract_images = extract_images
        self.ocr_enabled = ocr_enabled

        # Standard converter — OCR runs only when Docling detects it's needed
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = ocr_enabled
        pipeline_options.do_table_structure = extract_tables
        pipeline_options.generate_picture_images = extract_images
        pipeline_options.images_scale = 2.0
        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )

        # Fallback converter — forces OCR on every page (for image-based PDFs)
        fallback_options = PdfPipelineOptions()
        fallback_options.do_ocr = True
        fallback_options.ocr_options = EasyOcrOptions(force_full_page_ocr=True)
        fallback_options.do_table_structure = extract_tables
        fallback_options.generate_picture_images = extract_images
        fallback_options.images_scale = 2.0
        self.fallback_converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=fallback_options)}
        )

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _clean_text(self, text: str) -> str:
        """Clean individual text items of artifacts."""
        if not text:
            return ""

        # Remove <unknown> tags (both raw and HTML-escaped versions)
        text = re.sub(r'<unknown>\s*', '', text)
        text = re.sub(r'\s*</unknown>', '', text)
        text = re.sub(r'&lt;unknown&gt;\s*', '', text)
        text = re.sub(r'\s*&lt;/unknown&gt;', '', text)

        return text.strip()

    def _clean_markdown(self, content: str) -> str:
        """Clean markdown content of artifacts and noise."""
        # Remove <unknown> tags (both raw and HTML-escaped versions)
        content = re.sub(r'<unknown>\s*', '', content)
        content = re.sub(r'\s*</unknown>', '', content)
        content = re.sub(r'&lt;unknown&gt;\s*', '', content)
        content = re.sub(r'\s*&lt;/unknown&gt;', '', content)

        # Remove empty image placeholders
        content = re.sub(r'<!-- image -->\s*', '', content)

        # Remove excessive newlines (more than 2 consecutive)
        content = re.sub(r'\n{4,}', '\n\n\n', content)

        # Remove leading/trailing whitespace
        content = content.strip()

        return content

    def _extract_tables(self, doc: DoclingDocument) -> list[dict[str, Any]]:
        """Extract tables from the document."""
        tables = []

        for item, level in doc.iterate_items():
            if isinstance(item, TableItem):
                table_data = {
                    "table_id": str(item.self_ref) if hasattr(item, 'self_ref') else str(len(tables)),
                    "caption": None,
                    "markdown": "",
                    "page_numbers": [],
                }

                # Get table as markdown (pass doc to avoid deprecation warning)
                if hasattr(item, 'export_to_markdown'):
                    try:
                        table_data["markdown"] = item.export_to_markdown(doc=doc)
                    except TypeError:
                        # Fallback for older versions
                        table_data["markdown"] = item.export_to_markdown()
                elif hasattr(item, 'text'):
                    table_data["markdown"] = item.text

                # Extract page numbers from provenance
                if hasattr(item, 'prov') and item.prov:
                    for prov in item.prov:
                        if hasattr(prov, 'page_no'):
                            if prov.page_no not in table_data["page_numbers"]:
                                table_data["page_numbers"].append(prov.page_no)

                # Try to get caption
                if hasattr(item, 'caption') and item.caption:
                    table_data["caption"] = item.caption

                tables.append(table_data)

        return tables

    def _extract_images(self, doc: DoclingDocument) -> list[dict[str, Any]]:
        """Extract image metadata from the document."""
        images = []

        for item, level in doc.iterate_items():
            if isinstance(item, PictureItem):
                image_data = {
                    "image_id": item.self_ref if hasattr(item, 'self_ref') else str(len(images)),
                    "caption": None,
                    "page_numbers": [],
                    "description": None,  # Will be filled by vision model later
                }

                # Extract page numbers from provenance
                if hasattr(item, 'prov') and item.prov:
                    for prov in item.prov:
                        if hasattr(prov, 'page_no'):
                            if prov.page_no not in image_data["page_numbers"]:
                                image_data["page_numbers"].append(prov.page_no)

                # Try to get caption
                if hasattr(item, 'caption') and item.caption:
                    image_data["caption"] = item.caption

                images.append(image_data)

        return images

    def _merge_page_break_continuations(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Merge text fragments split by PDF page breaks.

        When a paragraph is physically split across pages, Docling may produce two
        separate text items. This detects those splits by checking:
          - The items are on consecutive pages
          - The first item ends without terminal punctuation (.?!:;)
          - The second item starts with a lowercase letter

        Both conditions together strongly indicate a mid-sentence page break.
        """
        if len(items) <= 1:
            return items

        merged = []
        i = 0
        while i < len(items):
            item = items[i]

            # Only attempt merges for text-type items with a following item
            if item["type"] not in ("text", "list_item") or i + 1 >= len(items):
                merged.append(item)
                i += 1
                continue

            next_item = items[i + 1]

            if next_item["type"] not in ("text", "list_item"):
                merged.append(item)
                i += 1
                continue

            current_text = item["text"].strip()
            next_text = next_item["text"].strip()

            if not current_text or not next_text:
                merged.append(item)
                i += 1
                continue

            current_pages = item.get("page_numbers", [])
            next_pages = next_item.get("page_numbers", [])

            # Must have page info and be on consecutive pages
            if (not current_pages or not next_pages
                    or max(current_pages) + 1 != min(next_pages)):
                merged.append(item)
                i += 1
                continue

            # Continuation heuristic: no terminal punctuation + next starts lowercase
            ends_incomplete = current_text[-1] not in ".?!:;"
            starts_lowercase = next_text[0].islower()

            if ends_incomplete and starts_lowercase:
                merged_item = dict(item)
                merged_item["text"] = current_text + " " + next_text
                merged_item["page_numbers"] = sorted(set(current_pages + next_pages))
                merged.append(merged_item)
                i += 2  # consume both items
            else:
                merged.append(item)
                i += 1

        return merged

    def _extract_content_items(self, doc: DoclingDocument) -> list[dict[str, Any]]:
        """
        Extract all content items with their positions and metadata.

        This provides a structured view of the document content that preserves
        order, page numbers, and section hierarchy for chunking.
        """
        items = []
        current_section_hierarchy = []

        for item, level in doc.iterate_items():
            item_type = type(item).__name__
            label = str(getattr(item, 'label', '')).lower()

            # Track section hierarchy
            if isinstance(item, SectionHeaderItem) or 'section_header' in label:
                heading_text = item.text if hasattr(item, 'text') else str(item)
                heading_text = heading_text.strip()

                # Adjust hierarchy based on level
                while len(current_section_hierarchy) >= level:
                    current_section_hierarchy.pop()
                current_section_hierarchy.append(heading_text)

            # Get page numbers
            page_numbers = []
            if hasattr(item, 'prov') and item.prov:
                for prov in item.prov:
                    if hasattr(prov, 'page_no') and prov.page_no not in page_numbers:
                        page_numbers.append(prov.page_no)

            # Get text content and clean it
            text = item.text if hasattr(item, 'text') else ""
            text = self._clean_text(text)

            # Determine content type
            if isinstance(item, TableItem):
                content_type = "table"
                # Get markdown representation of table
                try:
                    text = item.export_to_markdown(doc=doc)
                except TypeError:
                    text = item.export_to_markdown() if hasattr(item, 'export_to_markdown') else text
            elif isinstance(item, PictureItem):
                content_type = "image"
                # Extract image bytes for later vision description
                # Stored as base64 PNG; consumed by ImageDescriber in the indexing pipeline
                image_data = None
                try:
                    pil_img = item.get_image(doc)
                    if pil_img and pil_img.width >= 100 and pil_img.height >= 100:
                        buf = io.BytesIO()
                        pil_img.save(buf, format="PNG")
                        image_data = base64.b64encode(buf.getvalue()).decode("utf-8")
                except Exception as e:
                    logger.debug(f"Could not extract image bytes: {e}")
            elif isinstance(item, SectionHeaderItem) or 'section_header' in label:
                content_type = "heading"
            elif isinstance(item, ListItem):
                content_type = "list_item"
            else:
                content_type = "text"

            item_dict: dict = {
                "type": content_type,
                "text": text,
                "level": level,
                "page_numbers": page_numbers,
                "section_hierarchy": list(current_section_hierarchy),
                "item_ref": str(item.self_ref) if hasattr(item, 'self_ref') else None,
            }
            if content_type == "image" and image_data:
                item_dict["image_data"] = image_data
            items.append(item_dict)

        items = self._merge_page_break_continuations(items)
        return items

    def _extract_sections(self, doc: DoclingDocument) -> list[dict[str, Any]]:
        """Extract document section structure."""
        sections = []
        current_hierarchy = []

        for item, level in doc.iterate_items():
            # Check if this is a section header
            is_section_header = isinstance(item, SectionHeaderItem)

            # Also check label for section_header
            if not is_section_header and hasattr(item, 'label'):
                label_str = str(item.label).lower()
                is_section_header = 'section_header' in label_str or 'heading' in label_str

            if is_section_header:
                # This is a heading
                heading_text = item.text if hasattr(item, 'text') else str(item)

                # Clean the heading text
                heading_text = heading_text.strip()

                # Adjust hierarchy based on level
                while len(current_hierarchy) >= level:
                    current_hierarchy.pop()
                current_hierarchy.append(heading_text)

                page_no = None
                if hasattr(item, 'prov') and item.prov:
                    for prov in item.prov:
                        if hasattr(prov, 'page_no'):
                            page_no = prov.page_no
                            break

                sections.append({
                    "heading": heading_text,
                    "level": level,
                    "hierarchy": list(current_hierarchy),
                    "page_number": page_no,
                })

        return sections

    def parse(self, file_path: str | Path) -> ParsedDocument:
        """
        Parse a PDF file.

        Args:
            file_path: Path to the PDF file

        Returns:
            ParsedDocument containing parsed content and metadata
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        if not file_path.suffix.lower() == ".pdf":
            raise ValueError(f"File is not a PDF: {file_path}")

        logger.info(f"Parsing PDF: {file_path.name}")
        start_time = time.time()

        # Compute file hash
        file_hash = self._compute_file_hash(file_path)
        file_size = file_path.stat().st_size

        # Convert document — fall back to force-OCR if nothing is extracted
        result = self.converter.convert(file_path)
        doc: DoclingDocument = result.document

        if not list(doc.iterate_items()):
            logger.info(f"No content extracted with standard converter — retrying with full-page OCR for '{file_path.name}'")
            result = self.fallback_converter.convert(file_path)
            doc = result.document

        # Get total pages
        total_pages = 0
        if hasattr(doc, 'pages') and doc.pages:
            total_pages = len(doc.pages)

        # Export to markdown and clean
        markdown_content = doc.export_to_markdown()
        markdown_content = self._clean_markdown(markdown_content)

        # Extract structured elements
        tables = self._extract_tables(doc) if self.extract_tables else []
        images = self._extract_images(doc) if self.extract_images else []
        sections = self._extract_sections(doc)
        content_items = self._extract_content_items(doc)

        # Extract document metadata
        title = None
        author = None
        creation_date = None

        # Try to get title from document
        if hasattr(doc, 'name') and doc.name:
            title = doc.name

        processing_time = time.time() - start_time
        logger.info(f"Parsed {file_path.name} in {processing_time:.2f}s")

        # Create document metadata
        metadata = DocumentMetadata(
            filename=file_path.name,
            title=title or file_path.stem,
            total_pages=total_pages,
            file_hash=file_hash,
            file_size_bytes=file_size,
            author=author,
            creation_date=creation_date,
            processing_time_seconds=processing_time,
        )

        return ParsedDocument(
            metadata=metadata,
            markdown_content=markdown_content,
            tables=tables,
            images=images,
            sections=sections,
            content_items=content_items,
        )

    def parse_batch(self, file_paths: list[str | Path]) -> list[ParsedDocument]:
        """
        Parse multiple PDF files.

        Args:
            file_paths: List of paths to PDF files

        Returns:
            List of ParsedDocument instances
        """
        results = []
        for path in file_paths:
            try:
                result = self.parse(path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to parse {path}: {e}")
                raise

        return results
