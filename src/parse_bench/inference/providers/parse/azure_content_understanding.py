"""Provider for Azure AI Content Understanding (PARSE).

Azure Content Understanding is the successor to (and distinct from) Azure
Document Intelligence. It is a multimodal generative AI service that extracts
structured insights from unstructured content — documents, images, audio, and
video — using Azure's language, vision, and speech models. This provider uses its
document analysis capability (OCR, layout, and tables).

Uses the official ``azure-ai-contentunderstanding`` SDK (mirroring the
``azure_document_intelligence`` provider). Raw document bytes are submitted via
``begin_analyze_binary``; the returned ``AnalysisResult`` is persisted with
``as_dict()`` and, on ``normalize``, reconstructed into the SDK's typed
``AnalysisResult`` model so markdown/pages/layout are read via typed attribute
access rather than raw-dict digging.

Auth/config (env var or ``base_config`` key):
  * ``AZURE_CONTENT_UNDERSTANDING_KEY`` / ``api_key`` (``AzureKeyCredential``)
  * ``AZURE_CONTENT_UNDERSTANDING_ENDPOINT`` / ``endpoint``
  * ``AZURE_CONTENT_UNDERSTANDING_ANALYZER_ID`` / ``analyzer_id``
  * ``api_version`` (default ``2025-11-01``)
"""

from __future__ import annotations

import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from azure.ai.contentunderstanding import ContentUnderstandingClient
from azure.ai.contentunderstanding.models import (
    AnalysisContent,
    AnalysisResult,
    DocumentContent,
)
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import AzureError

from parse_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderTransientError,
)
from parse_bench.inference.providers.registry import register_provider
from parse_bench.schemas.parse_output import (
    LayoutItemIR,
    LayoutSegmentIR,
    PageIR,
    ParseLayoutPageIR,
    ParseOutput,
)
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from parse_bench.schemas.product import ProductType

logger = logging.getLogger(__name__)

# Default GA API version for Azure Content Understanding.
_DEFAULT_API_VERSION = "2025-11-01"

# Content Understanding paragraph role -> Canonical17 label string.
# Mirrors the Azure DI mapping so layout cross-eval is consistent across the two
# Azure providers.
CU_LABEL_MAP: dict[str, str] = {
    "title": "Title",
    "sectionHeading": "Section-header",
    "pageHeader": "Page-header",
    "pageFooter": "Page-footer",
    "footnote": "Footnote",
}

# Default label for paragraphs without a recognized role.
_DEFAULT_PARAGRAPH_LABEL = "Text"

# Virtual page dimension for coordinate conversion; CU polygons are normalized to
# [0,1] via the page's own inch dimensions, so this value cancels out (kept for
# parity with the Azure DI provider).
_VIRTUAL_PAGE_DIM = 1000.0

_PAGE_FURNITURE_ROLES = frozenset({"pageHeader", "pageFooter", "pageNumber"})

_MULTILINE_BOLD_RE = re.compile(
    r"(?:\A|(?<=[\r\n]))[ \t]*\*\*(?P<body>(?:(?!\*\*).)+?)\*\*[ \t]*(?=[\r\n]|\Z)",
    re.DOTALL,
)


@register_provider("azure_content_understanding")
class AzureContentUnderstandingProvider(Provider):
    """Provider for Azure AI Content Understanding PARSE (official SDK)."""

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        """Initialize the provider.

        :param provider_name: Name of the provider
        :param base_config: Optional configuration with:
            - ``api_key``: Content Understanding API key
              (defaults to ``AZURE_CONTENT_UNDERSTANDING_KEY`` env var)
            - ``endpoint``: Content Understanding endpoint URL
              (defaults to ``AZURE_CONTENT_UNDERSTANDING_ENDPOINT`` env var)
            - ``api_version``: REST API version (default: ``2025-11-01``)
            - ``analyzer_id``: Analyzer to use (defaults to
              ``AZURE_CONTENT_UNDERSTANDING_ANALYZER_ID``)
        """
        super().__init__(provider_name, base_config)

        # Get API key and endpoint
        self._api_key = self.base_config.get("api_key") or os.getenv("AZURE_CONTENT_UNDERSTANDING_KEY")
        self._endpoint = self.base_config.get("endpoint") or os.getenv("AZURE_CONTENT_UNDERSTANDING_ENDPOINT")

        if not self._api_key:
            raise ProviderConfigError(
                "Azure Content Understanding API key is required. "
                "Set AZURE_CONTENT_UNDERSTANDING_KEY environment variable "
                "or pass api_key in base_config."
            )
        if not self._endpoint:
            raise ProviderConfigError(
                "Azure Content Understanding endpoint is required. "
                "Set AZURE_CONTENT_UNDERSTANDING_ENDPOINT environment variable "
                "or pass endpoint in base_config."
            )

        self._api_version = self.base_config.get("api_version", _DEFAULT_API_VERSION)
        self._analyzer_id = self.base_config.get("analyzer_id") or os.getenv("AZURE_CONTENT_UNDERSTANDING_ANALYZER_ID")
        if not self._analyzer_id:
            raise ProviderConfigError(
                "Azure Content Understanding analyzer ID is required. "
                "Set AZURE_CONTENT_UNDERSTANDING_ANALYZER_ID environment variable "
                "or pass analyzer_id in base_config."
            )

        # Initialize the SDK client (mirrors the azure_document_intelligence provider).
        self._client = ContentUnderstandingClient(
            endpoint=self._endpoint,
            credential=AzureKeyCredential(self._api_key),
            api_version=self._api_version,
        )

    def _analyze(self, pdf_path: Path) -> dict[str, Any]:
        """Read the document, analyze it via the SDK, and return wire-shape JSON.

        Owns the full service call (read bytes -> analyze -> attach ``_config``),
        mirroring the ``azure_document_intelligence`` provider's ``_parse_pdf``.
        """
        if not pdf_path.exists():
            raise ProviderPermanentError(f"PDF file not found: {pdf_path}")

        pdf_bytes = pdf_path.read_bytes()
        try:
            poller = self._client.begin_analyze_binary(
                self._analyzer_id,
                pdf_bytes,
                content_type="application/octet-stream",
            )
            result = poller.result()
        except AzureError as e:
            # Classify as transient (retryable) or permanent. HttpResponseError (a
            # subclass) carries a structured ``status_code`` for HTTP failures;
            # plain AzureError (e.g. a dropped connection) has none, so the word
            # keywords catch code-less transport errors.
            message = f"Content Understanding analyze failed: {e}"
            status_code = getattr(e, "status_code", None)
            transient_status_codes = (408, 429, 500, 502, 503, 504)
            transient_keywords = (
                "timeout",
                "timed out",
                "network",
                "connection",
                "temporarily",
                "throttl",
                "rate limit",
            )
            if status_code in transient_status_codes or any(k in message.lower() for k in transient_keywords):
                raise ProviderTransientError(message) from e
            raise ProviderPermanentError(message) from e

        # Persist the camelCase wire JSON; ``normalize`` reconstructs the typed
        # ``AnalysisResult`` model from it.
        raw_output = result.as_dict()
        raw_output["_config"] = {
            "analyzer_id": self._analyzer_id,
            "api_version": self._api_version,
        }
        return raw_output

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"AzureContentUnderstandingProvider only supports PARSE product type, got {request.product_type}"
            )

        started_at = datetime.now()
        try:
            raw_output = self._analyze(Path(request.source_file_path))
        except (ProviderPermanentError, ProviderTransientError, ProviderConfigError):
            raise
        except Exception as e:  # noqa: BLE001 - non-SDK failure (e.g. file read)
            raise ProviderPermanentError(f"Unexpected error during inference: {e}") from e

        completed_at = datetime.now()
        latency_ms = int((completed_at - started_at).total_seconds() * 1000)
        return RawInferenceResult(
            request=request,
            pipeline=pipeline,
            pipeline_name=pipeline.pipeline_name,
            product_type=request.product_type,
            raw_output=raw_output,
            started_at=started_at,
            completed_at=completed_at,
            latency_in_ms=latency_ms,
        )

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        if raw_result.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"AzureContentUnderstandingProvider only supports PARSE product type, got {raw_result.product_type}"
            )

        # Reconstruct the SDK's typed model from the persisted wire JSON so we read
        # markdown/pages/layout via typed attribute access instead of dict digging.
        result = AnalysisResult(raw_result.raw_output)
        contents = result.contents or []

        # Reuse each content's normalized markdown for the document and per-page IR.
        content_markdown = [render_content_markdown(c) for c in contents]

        # Full-document markdown = concatenation of per-content markdown blocks.
        markdown = "\n\n".join(md for md in content_markdown if md).strip()

        # Coarse per-page IR (markdown is document-level; page_index from content range).
        pages: list[PageIR] = []
        for c, md in zip(contents, content_markdown, strict=True):
            start = int(getattr(c, "start_page_number", None) or 1)
            pages.append(PageIR(page_index=max(start - 1, 0), markdown=md))

        layout_pages = _build_layout_pages(contents)

        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=pages,
            layout_pages=layout_pages,
            markdown=markdown,
        )

        return InferenceResult(
            request=raw_result.request,
            pipeline_name=raw_result.pipeline_name,
            product_type=raw_result.product_type,
            raw_output=raw_result.raw_output,
            output=output,
            started_at=raw_result.started_at,
            completed_at=raw_result.completed_at,
            latency_in_ms=raw_result.latency_in_ms,
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _to_normalized_bbox(
    source: str | None,
    page_dims: dict[int, tuple[float, float]],
) -> tuple[int, float, float, float, float]:
    if not source:
        raise ValueError("CU element has no source region")

    regions = source.split(";")
    if len(regions) > 1:
        logger.warning("CU source has %d regions; using the first: %s", len(regions), source)

    encoded = regions[0]
    if not encoded.startswith("D(") or not encoded.endswith(")"):
        raise ValueError(f"Malformed CU source region: {source}")
    try:
        values = [float(value) for value in encoded[2:-1].split(",")]
    except ValueError as error:
        raise ValueError(f"Malformed CU source region: {source}") from error
    if len(values) < 9:
        raise ValueError(f"Malformed CU source region: {source}")

    page_number = int(values[0])
    xs = values[1:9:2]
    ys = values[2:9:2]
    x_min, y_min, x_max, y_max = min(xs), min(ys), max(xs), max(ys)
    page_width, page_height = page_dims.get(page_number, (1.0, 1.0))
    return (
        page_number,
        x_min / page_width if page_width > 0 else 0.0,
        y_min / page_height if page_height > 0 else 0.0,
        (x_max - x_min) / page_width if page_width > 0 else 0.0,
        (y_max - y_min) / page_height if page_height > 0 else 0.0,
    )


def _parse_element_reference(reference: str) -> tuple[str, int]:
    """Parse ``/<element-type>/<index>`` into its typed components."""
    if not reference.startswith("/"):
        raise ValueError(f"Invalid CU element reference: {reference}")
    parts = reference.removeprefix("/").split("/")
    if len(parts) != 2:
        raise ValueError(f"Invalid CU element reference: {reference}")
    try:
        index = int(parts[1])
    except ValueError:
        raise ValueError(f"Invalid CU element reference: {reference}") from None
    return parts[0], index


@dataclass(frozen=True)
class CULayoutItem:
    element_type: str
    element_index: int
    page_number: int
    item: LayoutItemIR


def _build_paragraph(
    index: int,
    content: DocumentContent,
    page_dims: dict[int, tuple[float, float]],
) -> CULayoutItem:
    paragraph = list(content.paragraphs or [])[index]
    page_number, x, y, width, height = _to_normalized_bbox(paragraph.source, page_dims)
    label = CU_LABEL_MAP.get(paragraph.role, _DEFAULT_PARAGRAPH_LABEL)
    if paragraph.role == "pageNumber":
        label = "Page-header" if y + height / 2 < 0.5 else "Page-footer"
    segment = LayoutSegmentIR(
        x=x,
        y=y,
        w=width,
        h=height,
        confidence=1.0,
        label=label,
    )
    return CULayoutItem(
        element_type="paragraphs",
        element_index=index,
        page_number=page_number,
        item=LayoutItemIR(
            type="text",
            value=paragraph.content or "",
            bbox=segment,
            layout_segments=[segment],
        ),
    )


def _build_table(
    index: int,
    content: DocumentContent,
    page_dims: dict[int, tuple[float, float]],
) -> list[CULayoutItem]:
    table = list(content.tables or [])[index]
    page_number, x, y, width, height = _to_normalized_bbox(table.source, page_dims)
    segment = LayoutSegmentIR(
        x=x,
        y=y,
        w=width,
        h=height,
        confidence=1.0,
        label="Table",
    )
    content_items = [
        item
        for cell in table.cells or []
        for reference in (cell.elements or [])
        for item in _build_layout_item(reference, content, page_dims)
    ]
    caption_items = [
        item
        for reference in (table.caption.elements or [] if table.caption else [])
        for item in _build_layout_item(reference, content, page_dims)
    ]
    footnote_items = [
        item
        for footnote in table.footnotes or []
        for reference in footnote.elements or []
        for item in _build_layout_item(reference, content, page_dims)
    ]

    table_item = CULayoutItem(
        element_type="tables",
        element_index=index,
        page_number=page_number,
        item=LayoutItemIR(
            type="table",
            value=" ".join(cell.content for cell in (table.cells or []) if cell.content),
            bbox=segment,
            layout_segments=[segment],
        ),
    )

    caption_before = not content_items or (
        caption_items and caption_items[0].element_index < content_items[0].element_index
    )
    footnote_before = not content_items or (
        footnote_items and footnote_items[0].element_index < content_items[0].element_index
    )

    items = [table_item]
    items.extend(caption_items if caption_before else [])
    items.extend(footnote_items if footnote_before else [])
    items.extend(content_items)
    items.extend(caption_items if not caption_before else [])
    items.extend(footnote_items if not footnote_before else [])
    return items


def _build_figure(
    index: int,
    content: DocumentContent,
    page_dims: dict[int, tuple[float, float]],
) -> list[CULayoutItem]:
    figure = list(content.figures or [])[index]
    paragraphs = list(content.paragraphs or [])
    texts = [
        paragraphs[paragraph_index].content
        for reference in dict.fromkeys(figure.elements or [])
        for element_type, paragraph_index in [_parse_element_reference(reference)]
        if element_type == "paragraphs" and paragraphs[paragraph_index].content
    ]
    page_number, x, y, width, height = _to_normalized_bbox(figure.source, page_dims)
    segment = LayoutSegmentIR(
        x=x,
        y=y,
        w=width,
        h=height,
        confidence=1.0,
        label="Picture",
    )
    content_items = [
        item for reference in figure.elements or [] for item in _build_layout_item(reference, content, page_dims)
    ]
    caption_items = [
        item
        for reference in (figure.caption.elements or [] if figure.caption else [])
        for item in _build_layout_item(reference, content, page_dims)
    ]
    footnote_items = [
        item
        for footnote in figure.footnotes or []
        for reference in footnote.elements or []
        for item in _build_layout_item(reference, content, page_dims)
    ]

    caption_before = not content_items or (
        caption_items and caption_items[0].element_index < content_items[0].element_index
    )
    footnote_before = not content_items or (
        footnote_items and footnote_items[0].element_index < content_items[0].element_index
    )

    figure_item = CULayoutItem(
        element_type="figures",
        element_index=index,
        page_number=page_number,
        item=LayoutItemIR(
            type="image",
            value=" ".join(texts) if texts else getattr(figure.caption, "content", None) or "",
            bbox=segment,
            layout_segments=[segment],
        ),
    )
    items = [figure_item]
    items.extend(caption_items if caption_before else [])
    items.extend(footnote_items if footnote_before else [])
    items.extend(content_items)
    items.extend(caption_items if not caption_before else [])
    items.extend(footnote_items if not footnote_before else [])
    return items


def _build_section(
    index: int,
    content: DocumentContent,
    page_dims: dict[int, tuple[float, float]],
) -> list[CULayoutItem]:
    section = list(content.sections or [])[index]
    items: list[CULayoutItem] = []
    for reference in section.elements or []:
        items.extend(_build_layout_item(reference, content, page_dims))
    return items


def _build_layout_item(
    reference: str,
    content: DocumentContent,
    page_dims: dict[int, tuple[float, float]],
) -> list[CULayoutItem]:
    element_type, index = _parse_element_reference(reference)
    collection = getattr(content, element_type, None)
    if collection is None or index < 0 or index >= len(collection):
        raise ValueError(f"CU element reference out of range: {reference}")
    if element_type == "paragraphs":
        return [_build_paragraph(index, content, page_dims)]
    if element_type == "tables":
        return _build_table(index, content, page_dims)
    if element_type == "figures":
        return _build_figure(index, content, page_dims)
    return _build_section(index, content, page_dims)


def _insert_page_furniture(
    items: list[CULayoutItem],
    content: DocumentContent,
    page_dims: dict[int, tuple[float, float]],
) -> None:
    referenced_paragraphs = {item.element_index for item in items if item.element_type == "paragraphs"}
    for index, paragraph in enumerate(content.paragraphs or []):
        if paragraph.role not in _PAGE_FURNITURE_ROLES or index in referenced_paragraphs:
            continue
        position = 0
        while position < len(items):
            item = items[position]
            if item.element_type == "paragraphs" and item.element_index >= index:
                break
            position += 1
        items.insert(position, _build_paragraph(index, content, page_dims))


def _build_layout_pages(contents: list[AnalysisContent]) -> list[ParseLayoutPageIR]:
    """Build layout_pages from CU paragraphs/tables/figures for layout cross-eval.

    Reads the SDK's typed ``DocumentContent`` model directly. Groups elements by
    page using the source-polygon page index and converts CU polygon coordinates
    (page units) into normalized [0,1] ``LayoutSegmentIR`` entries.
    """
    # Only document contents carry pages/paragraphs/tables/figures.
    document_contents = [c for c in contents if isinstance(c, DocumentContent)]

    page_items: dict[int, list[LayoutItemIR]] = defaultdict(list)
    page_headers: dict[int, list[str]] = defaultdict(list)
    page_footers: dict[int, list[str]] = defaultdict(list)
    page_numbers: dict[int, list[str]] = defaultdict(list)

    for content in document_contents:
        page_dims = {
            int(page.page_number or 1): (
                float(page.width or 1.0),
                float(page.height or 1.0),
            )
            for page in content.pages or []
        }

        if not content.sections:
            logger.warning("CU document content has no sections; using paragraph order")
            layout_items = [
                _build_paragraph(index, content, page_dims) for index, _ in enumerate(content.paragraphs or [])
            ]
        else:
            # Build the top level section (index 0), which contains all elements in reading order.
            layout_items = _build_layout_item("/sections/0", content, page_dims)

            # Insert remaining page furniture paragraphs (headers/footers/page numbers)
            _insert_page_furniture(layout_items, content, page_dims)

        for item in layout_items:
            page_items[item.page_number].append(item.item)
            if item.element_type != "paragraphs":
                continue
            paragraph = list(content.paragraphs or [])[item.element_index]
            if item.item.bbox.label == "Page-header" and item.item.value:
                page_headers[item.page_number].append(item.item.value)
            elif item.item.bbox.label == "Page-footer" and item.item.value:
                page_footers[item.page_number].append(item.item.value)
            if paragraph.role == "pageNumber" and item.item.value:
                page_numbers[item.page_number].append(item.item.value)

    return [
        ParseLayoutPageIR(
            page_number=page_num,
            width=_VIRTUAL_PAGE_DIM,
            height=_VIRTUAL_PAGE_DIM,
            page_header_markdown="\n".join(page_headers[page_num]),
            page_footer_markdown="\n".join(page_footers[page_num]),
            printed_page_number="\n".join(page_numbers[page_num]),
            items=page_items[page_num],
        )
        for page_num in sorted(page_items)
    ]


def _apply_markdown_edits(md: str, edits: list[tuple[int, int, str]]) -> str:
    """Apply edits right-to-left, keeping the first of any overlapping spans."""
    last_start = len(md)

    # A larger end offset comes first. For equal ends, the smaller start offset
    # puts the parent span before a nested span.
    ordered = sorted(edits, key=lambda edit: (edit[1], -edit[0]), reverse=True)
    for start, stop, replacement in ordered:
        if stop > last_start:
            logger.warning(
                "skipping_overlapping_cu_markdown_edit: span=[%d, %d)",
                start,
                stop,
            )
            continue

        md = md[:start] + replacement + md[stop:]
        last_start = start

    return md


def render_content_markdown(content: AnalysisContent) -> str:
    """Return content Markdown with links and page furniture normalized.

    Figures are preserved in the original Markdown for chart evaluation.
    Page header, footer, and number comments are unwrapped in their Markdown
    spans, preserving their original text and formatting instead of substituting
    structured paragraph content. Hyperlinks
    with structured spans are replaced with their visible content; spanless
    links are preserved because they cannot be located authoritatively.
    Non-furniture paragraphs use a local regex to fold standalone multiline
    bold blocks onto one line by replacing their internal line boundaries with
    spaces. Raw responses and layout are unchanged.
    """
    md = content.markdown or ""
    if not md:
        return md

    edits: list[tuple[int, int, str]] = []
    for paragraph in getattr(content, "paragraphs", None) or []:
        role = paragraph.role
        if paragraph.span is None:
            raise ValueError(f"CU {role} paragraph has no span")
        start = int(paragraph.span.offset)
        stop = start + int(paragraph.span.length)
        paragraph_markdown = md[start:stop]
        if role in _PAGE_FURNITURE_ROLES:
            replacement = re.sub(
                r"<!--\s*Page(?:Header|Footer|Number)\s*:\s*(.*?)\s*-->",
                r"\1",
                paragraph_markdown,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if replacement:
                replacement = f" {replacement} "
            edits.append((start, stop, replacement))
        else:
            if re.search(r"[`\\<>]|~~~", paragraph_markdown):
                continue
            for match in _MULTILINE_BOLD_RE.finditer(paragraph_markdown):
                body = match.group("body")
                if any(not line.strip() for line in re.split(r"\r\n|\r|\n", body)):
                    continue
                offset = start + match.start("body")
                for boundary in re.finditer(r"[ \t]*(?:\r\n|\r|\n)[ \t]*", body):
                    edits.append((offset + boundary.start(), offset + boundary.end(), " "))

    for hyperlink in getattr(content, "hyperlinks", None) or []:
        if hyperlink.span is None:
            continue
        start = int(hyperlink.span.offset)
        stop = start + int(hyperlink.span.length)
        edits.append((start, stop, hyperlink.content or ""))

    return _apply_markdown_edits(md, edits)
