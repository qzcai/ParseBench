from datetime import UTC, datetime

import pytest

from parse_bench.evaluation.layout_adapters import create_layout_adapter_for_result
from parse_bench.evaluation.layout_adapters.adapters import MistralOCRLayoutAdapter
from parse_bench.schemas.layout_detection_output import LayoutDetectionModel
from parse_bench.schemas.parse_output import (
    LayoutItemIR,
    LayoutSegmentIR,
    ParseLayoutPageIR,
    ParseOutput,
)
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
from parse_bench.schemas.product import ProductType


@pytest.mark.parametrize(("confidence", "expected_score"), [(0.9, 0.9), (0.0, 0.0), (None, 1.0)])
@pytest.mark.parametrize("rotation", [None, 30.0])
def test_mistral_layout_adapter_preserves_string_labels(
    confidence: float | None, expected_score: float, rotation: float | None
) -> None:
    timestamp = datetime.now(UTC)
    pipeline_name = "mistral_ocr_4"
    result = InferenceResult(
        request=InferenceRequest(
            example_id="layout-case",
            source_file_path="source.pdf",
            product_type=ProductType.PARSE,
        ),
        pipeline_name=pipeline_name,
        product_type=ProductType.PARSE,
        raw_output={},
        output=ParseOutput(
            example_id="layout-case",
            pipeline_name=pipeline_name,
            markdown="table",
            layout_pages=[
                ParseLayoutPageIR(
                    page_number=1,
                    width=100,
                    height=200,
                    items=[
                        LayoutItemIR(
                            type="table",
                            value="table",
                            layout_segments=[
                                LayoutSegmentIR(
                                    x=0.1,
                                    y=0.2,
                                    w=0.3,
                                    h=0.4,
                                    confidence=confidence,
                                    r=rotation,
                                    label="Table",
                                )
                            ],
                        )
                    ],
                )
            ],
        ),
        started_at=timestamp,
        completed_at=timestamp,
        latency_in_ms=0,
    )

    adapter = create_layout_adapter_for_result(result)
    assert isinstance(adapter, MistralOCRLayoutAdapter)

    layout = adapter.to_layout_output(result)

    assert layout.model == LayoutDetectionModel.MISTRAL_OCR_LAYOUT
    assert layout.image_width == 100
    assert layout.image_height == 200
    assert len(layout.predictions) == 1
    assert layout.predictions[0].label == "Table"
    assert layout.predictions[0].bbox == pytest.approx([10.0, 40.0, 40.0, 120.0])
    assert layout.predictions[0].score == expected_score
    assert layout.predictions[0].r == rotation
