"""Tests for Azure Content Understanding Markdown normalization."""

import json
from datetime import datetime

import pytest
from azure.ai.contentunderstanding.models import DocumentContent

from parse_bench.inference.providers.parse.azure_content_understanding import (
    AzureContentUnderstandingProvider,
    render_content_markdown,
)
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest, RawInferenceResult

_CHART_JSON = json.dumps({"type": "bar", "data": {"labels": ["2024"], "datasets": [{"label": "Sales", "data": [7]}]}})
_FIGURE_MARKDOWN = (
    "<figure>\n"
    "<figcaption>Quarterly sales</figcaption>\n\n"
    "![Quarterly sales](chart.png)\n\n"
    f"```chart\n{_CHART_JSON}\n```\n"
    "</figure>"
)


@pytest.fixture
def document_content() -> dict:
    return {
        "kind": "document",
        "startPageNumber": 3,
        "endPageNumber": 3,
        "markdown": _FIGURE_MARKDOWN,
        "pages": [{"pageNumber": 3, "width": 10, "height": 10}],
        "figures": [
            {
                "kind": "chart",
                "content": _CHART_JSON,
                "span": {"offset": 0, "length": len(_FIGURE_MARKDOWN)},
                "caption": {"content": "Quarterly sales"},
                "source": "D(3,1,1,9,1,9,9,1,9)",
            }
        ],
        "sections": [{"elements": ["/figures/0"]}],
    }


def test_normalize_preserves_raw_chart_markdown_and_figure_layout(document_content: dict) -> None:
    pipeline = PipelineSpec(
        pipeline_name="azure_cu_test", provider_name="azure_content_understanding", product_type="parse"
    )
    now = datetime.now()
    raw_result = RawInferenceResult(
        request=InferenceRequest(example_id="chart/test", source_file_path="chart.pdf", product_type="parse"),
        pipeline=pipeline,
        pipeline_name=pipeline.pipeline_name,
        product_type="parse",
        raw_output={"contents": [document_content]},
        started_at=now,
        completed_at=now,
        latency_in_ms=0,
    )
    provider = object.__new__(AzureContentUnderstandingProvider)

    result = provider.normalize(raw_result)

    assert result.output.markdown == _FIGURE_MARKDOWN
    assert result.output.pages[0].markdown == _FIGURE_MARKDOWN
    assert result.output.pages[0].page_index == 2
    figure = result.output.layout_pages[0].items[0]
    assert figure.type == "image"
    assert figure.value == "Quarterly sales"
    assert figure.bbox is not None
    assert figure.bbox.label == "Picture"
    assert result.raw_output == raw_result.raw_output


def test_render_content_markdown_keeps_other_normalization_around_chart(document_content: dict) -> None:
    header = "<!-- PageHeader: Annual report -->"
    link = "[Report](https://example.com)"
    prefix = f"{header}\n{link}\n"
    bold = "**Quarterly\nsales**"
    document_content["markdown"] = f"{prefix}{_FIGURE_MARKDOWN}\n{bold}"
    document_content["figures"][0]["span"]["offset"] = len(prefix)
    document_content["paragraphs"] = [
        {"role": "pageHeader", "content": "Annual report", "span": {"offset": 0, "length": len(header)}},
        {
            "content": "Quarterly\nsales",
            "span": {"offset": len(prefix) + len(_FIGURE_MARKDOWN) + 1, "length": len(bold)},
        },
    ]
    document_content["hyperlinks"] = [{"content": "Report", "span": {"offset": len(header) + 1, "length": len(link)}}]

    markdown = render_content_markdown(DocumentContent(document_content))

    assert markdown == f" Annual report \nReport\n{_FIGURE_MARKDOWN}\n**Quarterly sales**"
