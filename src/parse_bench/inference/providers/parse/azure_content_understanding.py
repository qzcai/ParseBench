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
  * ``api_version`` (default ``2025-11-01``), ``analyzer_id`` (default
    ``prebuilt-layout``)
"""

import logging
import os
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
    "pageNumber": "Page-footer",
}

# Default label for paragraphs without a recognized role.
_DEFAULT_PARAGRAPH_LABEL = "Text"

# Virtual page dimension for coordinate conversion; CU polygons are normalized to
# [0,1] via the page's own inch dimensions, so this value cancels out (kept for
# parity with the Azure DI provider).
_VIRTUAL_PAGE_DIM = 1000.0



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
            - ``analyzer_id``: Analyzer to use (default: ``prebuilt-layout``)
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
        self._analyzer_id = self.base_config.get("analyzer_id", "prebuilt-layout")

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
            if status_code in transient_status_codes or any(
                k in message.lower() for k in transient_keywords
            ):
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

        # Full-document markdown = concatenation of per-content markdown blocks.
        markdown = "\n\n".join(c.markdown for c in contents if c.markdown).strip()

        # Coarse per-page IR (markdown is document-level; page_index from content range).
        pages: list[PageIR] = []
        for c in contents:
            start = int(getattr(c, "start_page_number", None) or 1)
            pages.append(PageIR(page_index=max(start - 1, 0), markdown=c.markdown or ""))

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


def _polygon_to_normalized_bbox(
    polygon: list[float],
    page_width: float,
    page_height: float,
) -> tuple[float, float, float, float]:
    """Convert an 8-float corner polygon (page units) to normalized [0,1] xywh."""
    xs = [polygon[i] for i in range(0, len(polygon), 2)]
    ys = [polygon[i] for i in range(1, len(polygon), 2)]
    x_min, y_min, x_max, y_max = min(xs), min(ys), max(xs), max(ys)
    nx = x_min / page_width if page_width > 0 else 0.0
    ny = y_min / page_height if page_height > 0 else 0.0
    nw = (x_max - x_min) / page_width if page_width > 0 else 0.0
    nh = (y_max - y_min) / page_height if page_height > 0 else 0.0
    return (nx, ny, nw, nh)


def _build_layout_pages(contents: list[AnalysisContent]) -> list[ParseLayoutPageIR]:
    """Build layout_pages from CU paragraphs/tables/figures for layout cross-eval.

    Reads the SDK's typed ``DocumentContent`` model directly. Groups elements by
    page using the source-polygon page index and converts CU polygon coordinates
    (page units) into normalized [0,1] ``LayoutSegmentIR`` entries.
    """
    from collections import defaultdict

    # Only document contents carry pages/paragraphs/tables/figures.
    document_contents = [c for c in contents if isinstance(c, DocumentContent)]

    # page dimensions (inches) keyed by page number.
    page_dims: dict[int, tuple[float, float]] = {}
    for content in document_contents:
        for page in content.pages or []:
            pnum = int(page.page_number or 1)
            width = float(page.width or 1.0)
            height = float(page.height or 1.0)
            page_dims[pnum] = (width, height)

    # (label, nx, ny, nw, nh, content, confidence) grouped by page.
    pages_items: dict[int, list[tuple[str, float, float, float, float, str, float]]] = defaultdict(list)

    def _add(label: str, source: str | None, text: str | None) -> None:
        # Every layout element must carry a position; an empty source is a bug.
        if not source:
            raise ValueError(f"CU element '{label}' has no source polygon")
        # Text is optional (e.g. a figure with no caption is still a valid box).
        text = text or ""
        # CU encodes position as ``D(page, x1,y1, ..., x4,y4)``. An element spanning
        # multiple non-contiguous regions is a ``;``-separated list of D(...) blocks;
        # we only need one bbox, so warn and take the first.
        regions = source.split(";")
        if len(regions) > 1:
            logger.warning("CU source has %d regions; using the first: %s", len(regions), source)
        # Strip the ``D(`` prefix and ``)`` suffix, then require a page + 4 corners.
        nums = [float(x) for x in regions[0][2:-1].split(",")]
        assert len(nums) >= 9, f"Malformed CU source polygon: {source}"

        page_num, polygon = int(nums[0]), nums[1:9]
        pw, ph = page_dims.get(page_num, (1.0, 1.0))
        nx, ny, nw, nh = _polygon_to_normalized_bbox(polygon, pw, ph)
        pages_items[page_num].append((label, nx, ny, nw, nh, text, 1.0))

    for content in document_contents:
        # Paragraphs -> text / heading / header / footer elements.
        for para in content.paragraphs or []:
            role = para.role
            label = CU_LABEL_MAP.get(role, _DEFAULT_PARAGRAPH_LABEL) if role else _DEFAULT_PARAGRAPH_LABEL
            _add(label, para.source, para.content)

        # Tables -> Table elements (CU table objects carry their own source).
        for table in content.tables or []:
            text = " ".join(cell.content for cell in (table.cells or []) if cell.content)
            _add("Table", table.source, text)

        # Figures (charts/images) -> Picture elements.
        for fig in content.figures or []:
            caption = fig.caption.content if fig.caption else None
            _add("Picture", fig.source, caption)

    layout_pages: list[ParseLayoutPageIR] = []
    for page_num in sorted(pages_items.keys()):
        items: list[LayoutItemIR] = []
        for label, nx, ny, nw, nh, text, confidence in pages_items[page_num]:
            seg = LayoutSegmentIR(x=nx, y=ny, w=nw, h=nh, confidence=confidence, label=label)
            norm = label.strip().lower()
            if norm == "table":
                item_type = "table"
            elif norm == "picture":
                item_type = "image"
            else:
                item_type = "text"
            items.append(LayoutItemIR(type=item_type, value=text, bbox=seg, layout_segments=[seg]))
        layout_pages.append(
            ParseLayoutPageIR(
                page_number=page_num,
                width=_VIRTUAL_PAGE_DIM,
                height=_VIRTUAL_PAGE_DIM,
                items=items,
            )
        )

    return layout_pages
