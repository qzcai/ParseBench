"""Regression coverage for LLMParser replay, transport and layout fidelity."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import httpx
import pytest

from parse_bench.evaluation.layout_adapters.adapters import LLMParserLayoutAdapter
from parse_bench.evaluation.layout_label_mappers.base import MappingContext
from parse_bench.evaluation.layout_label_mappers.mappers import LLMParserLabelMapper
from parse_bench.inference.providers.base import ProviderConfigError, ProviderPermanentError
from parse_bench.inference.providers.parse import llmparser
from parse_bench.inference.providers.parse.llmparser import LLMParserProvider, normalize_llmparser_result
from parse_bench.schemas.layout_detection_output import LayoutDetectionModel
from parse_bench.schemas.layout_ontology import CanonicalLabel
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest
from parse_bench.schemas.product import ProductType

ORIGIN = "https://llmparser.example"
OPERATION = ORIGIN + llmparser.OPERATION_PREFIX + "test-operation?api-version=2024-11-30"
PIPELINE = PipelineSpec(
    pipeline_name="llmparser_terra_low",
    provider_name="llmparser",
    product_type=ProductType.PARSE,
    config={"model": "gpt-6.1-terra", "reasoning_effort": "low"},
)


@pytest.fixture(autouse=True)
def isolated_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in ("LLMPARSER_ENDPOINT", "LLMPARSER_API_KEY", "LLMPARSER_PRECOMPUTED_DIR", "LLMPARSER_OPERATION_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLMPARSER_OPERATION_DIR", str(tmp_path / "operations"))


@pytest.fixture
def request_file(tmp_path: Path) -> InferenceRequest:
    source = tmp_path / "input.pdf"
    source.write_bytes(b"%PDF-1.4\nsynthetic fixture")
    return InferenceRequest(example_id="example", source_file_path=str(source), product_type=ProductType.PARSE)


@pytest.fixture
def payload() -> dict:
    content = "# Heading\n\nBody"
    return {
        "apiVersion": llmparser.API_VERSION,
        "modelId": llmparser.MODEL_ID,
        "contentFormat": "markdown",
        "stringIndexType": "unicodeCodePoint",
        "content": content,
        "pages": [{"pageNumber": 1, "spans": [{"offset": 0, "length": len(content)}]}],
        "_llmparser": {
            "coordinateSystem": "normalized",
            "bboxFormat": "xywh",
            "profile": {"deployment": "gpt-6.1-terra", "reasoning_effort": "low"},
            "regions": [
                {"pageNumber": 1, "label": label.value, "text": f"  {label.value}\n", "bbox": [0.1, 0.2, 0.3, 0.4]}
                for label in CanonicalLabel
            ],
        },
    }


def save_response(tmp_path: Path, request: InferenceRequest, payload: dict) -> Path:
    root = tmp_path / "responses"
    root.mkdir()
    digest = hashlib.sha256(Path(request.source_file_path).read_bytes()).hexdigest()
    (root / "manifest.json").write_text(
        json.dumps({"schemaVersion": 1, "entries": [{"inputPath": "document.pdf", "sha256": digest}]}),
        encoding="utf-8",
    )
    (root / "document.pdf.json").write_text(json.dumps(payload), encoding="utf-8")
    return root


def test_exact_replay_preserves_markdown_regions_and_all_labels(
    tmp_path: Path, request_file: InferenceRequest, payload: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = save_response(tmp_path, request_file, payload)
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: pytest.fail("offline mode attempted network access"))
    provider = LLMParserProvider("llmparser", {**PIPELINE.config, "precomputed_dir": str(root)})
    raw = provider.run_inference(PIPELINE, request_file)
    normalized = provider.normalize(raw)
    assert normalized.output.markdown == payload["content"]
    assert normalized.output.pages[0].markdown == payload["content"]
    assert raw.raw_output["_llmparser"] == payload["_llmparser"]
    assert raw.raw_output["_config"]["response_source"] == "precomputed"
    layout = LLMParserLayoutAdapter().to_layout_output(normalized)
    assert layout.model == LayoutDetectionModel.LLMPARSER_LAYOUT
    assert layout.image_width == layout.image_height == 1
    assert [prediction.label for prediction in layout.predictions] == [label.value for label in CanonicalLabel]
    assert layout.predictions[0].bbox == pytest.approx([0.1, 0.2, 0.4, 0.6])
    assert [prediction.provider_metadata["order_index"] for prediction in layout.predictions] == list(range(17))
    assert normalized.output.layout_pages[0].items[0].value == payload["_llmparser"]["regions"][0]["text"]
    mapper = LLMParserLabelMapper()
    context = MappingContext(
        provider_name="llmparser",
        pipeline_name=PIPELINE.pipeline_name,
        model=layout.model,
        raw_output=raw.raw_output,
        layout_output=layout,
    )
    assert [mapper.to_canonical(prediction.label, prediction, context) for prediction in layout.predictions] == list(
        CanonicalLabel
    )
    assert not LLMParserLayoutAdapter().to_layout_output(normalized, page_filter=2).predictions


def test_replay_hash_miss_never_falls_back_to_live(
    tmp_path: Path, request_file: InferenceRequest, payload: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = save_response(tmp_path, request_file, payload)
    monkeypatch.setenv("LLMPARSER_ENDPOINT", ORIGIN)
    monkeypatch.setenv("LLMPARSER_API_KEY", "test-only")
    provider = LLMParserProvider("llmparser", {"precomputed_dir": str(root)})
    Path(request_file.source_file_path).write_bytes(b"different input bytes")
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: pytest.fail("hash miss attempted live inference"))
    with pytest.raises(ProviderPermanentError, match="exact input bytes"):
        provider.run_inference(PIPELINE, request_file)


@pytest.mark.parametrize("kind", ["hash", "path", "duplicate"])
def test_invalid_manifest_is_rejected(kind: str, tmp_path: Path, request_file: InferenceRequest, payload: dict) -> None:
    root = save_response(tmp_path, request_file, payload)
    manifest = json.loads((root / "manifest.json").read_text())
    entry = manifest["entries"][0]
    if kind == "hash":
        entry["sha256"] = "invalid"
    elif kind == "path":
        entry["inputPath"] = "../outside.pdf"
    else:
        manifest["entries"].append({**entry, "inputPath": "another.pdf"})
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ProviderConfigError):
        LLMParserProvider("llmparser", {"precomputed_dir": str(root)})


def test_replay_rejects_mismatched_pipeline_provenance(
    tmp_path: Path, request_file: InferenceRequest, payload: dict
) -> None:
    payload["_llmparser"]["profile"]["reasoning_effort"] = "high"
    root = save_response(tmp_path, request_file, payload)
    provider = LLMParserProvider("llmparser", {"precomputed_dir": str(root)})
    with pytest.raises(ProviderPermanentError, match="provenance"):
        provider.run_inference(PIPELINE, request_file)


def test_empty_replay_setting_does_not_enable_live(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMPARSER_PRECOMPUTED_DIR", "")
    monkeypatch.setenv("LLMPARSER_ENDPOINT", ORIGIN)
    monkeypatch.setenv("LLMPARSER_API_KEY", "test-only")
    with pytest.raises(ProviderConfigError, match="nonempty directory"):
        LLMParserProvider("llmparser", PIPELINE.config)


@pytest.mark.parametrize(
    "change",
    [
        {"bbox": [0, 0, 0, 0.5]},
        {"bbox": [0.8, 0.2, 0.3, 0.4]},
        {"bbox": [0.1, 0.2, float("nan"), 0.4]},
        {"bbox": [False, 0.2, 0.3, 0.4]},
        {"label": "not-a-canonical-label"},
        {"pageNumber": 2},
        {"pageNumber": True},
        {"text": None},
        {"attributes": {"invalid": 42}},
    ],
)
def test_invalid_region_is_an_explicit_provider_error(change: dict, payload: dict) -> None:
    payload["_llmparser"]["regions"][0].update(change)
    with pytest.raises(ProviderPermanentError):
        normalize_llmparser_result(payload, "example", PIPELINE.pipeline_name)


def test_missing_regions_are_not_a_successful_empty_layout(payload: dict) -> None:
    payload["_llmparser"]["regions"] = []
    with pytest.raises(ProviderPermanentError, match="lacks predicted layout"):
        normalize_llmparser_result(payload, "example", PIPELINE.pipeline_name)


def test_legacy_receipt_regions_and_utf16_page_spans(payload: dict) -> None:
    region = copy.deepcopy(payload["_llmparser"]["regions"][0])
    region.pop("pageNumber")
    payload["_llmparser"].pop("regions")
    payload["_llmparser"]["receipts"] = [{"page": page, "elements": [region]} for page in (1, 2)]
    payload["content"] = "A\U0001f600B\nC"
    payload["stringIndexType"] = "utf16CodeUnit"
    payload["pages"] = [
        {"pageNumber": 1, "spans": [{"offset": 0, "length": 4}]},
        {"pageNumber": 2, "spans": [{"offset": 4, "length": 2}]},
    ]
    output = normalize_llmparser_result(payload, "example", PIPELINE.pipeline_name)
    assert [page.markdown for page in output.pages] == ["A\U0001f600B", "\nC"]
    assert [page.page_number for page in output.layout_pages] == [1, 2]
    payload["pages"][0]["spans"][0]["length"] = 2
    with pytest.raises(ProviderPermanentError, match="splits a UTF-16"):
        normalize_llmparser_result(payload, "example", PIPELINE.pipeline_name)


@pytest.mark.parametrize(
    "span", [{"offset": -1, "length": 1}, {"offset": 0, "length": 100}, {"offset": True, "length": 1}]
)
def test_invalid_page_span_is_rejected(span: dict, payload: dict) -> None:
    payload["pages"][0]["spans"] = [span]
    with pytest.raises(ProviderPermanentError, match="span"):
        normalize_llmparser_result(payload, "example", PIPELINE.pipeline_name)


def mock_live(monkeypatch: pytest.MonkeyPatch, handler) -> LLMParserProvider:
    monkeypatch.setenv("LLMPARSER_ENDPOINT", ORIGIN)
    monkeypatch.setenv("LLMPARSER_API_KEY", "test-only")
    client_class = httpx.Client
    monkeypatch.setattr(
        llmparser.httpx, "Client", lambda **kwargs: client_class(transport=httpx.MockTransport(handler), **kwargs)
    )
    return LLMParserProvider("llmparser", PIPELINE.config)


def test_live_submit_and_poll_preserve_raw_metadata(
    request_file: InferenceRequest, payload: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["Ocp-Apim-Subscription-Key"] == "test-only"
        if request.method == "POST":
            assert request.url.params["stringIndexType"] == "unicodeCodePoint"
            assert request.url.params["llmparserModel"] == "gpt-6.1-terra"
            assert request.content == Path(request_file.source_file_path).read_bytes()
            return httpx.Response(202, headers={"Operation-Location": OPERATION})
        return httpx.Response(200, json={"status": "succeeded", "analyzeResult": payload})

    provider = mock_live(monkeypatch, handler)
    raw = provider.run_inference(PIPELINE, request_file)
    assert [request.method for request in requests] == ["POST", "GET"]
    assert raw.raw_output["_llmparser"] == payload["_llmparser"]
    assert raw.raw_output["_operation"]["status"] == "succeeded"
    assert raw.raw_output["_config"]["response_source"] == "live"
    assert "test-only" not in json.dumps(raw.raw_output)
    assert provider.normalize(raw).output.layout_pages


def test_poll_failure_resumes_without_another_post(
    request_file: InferenceRequest, payload: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    methods = []
    fail_poll = True

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        if request.method == "POST":
            return httpx.Response(202, headers={"Operation-Location": OPERATION})
        if fail_poll:
            raise httpx.ReadTimeout("synthetic polling timeout")
        return httpx.Response(200, json={"status": "succeeded", "analyzeResult": payload})

    provider = mock_live(monkeypatch, handler)
    with pytest.raises(ProviderPermanentError) as error:
        provider.run_inference(PIPELINE, request_file)
    assert error.value.debug_payload["operationLocation"] == OPERATION
    fail_poll = False
    provider = LLMParserProvider("llmparser", PIPELINE.config)
    provider.run_inference(PIPELINE, request_file)
    assert methods == ["POST", "GET", "GET"]


def test_uncertain_submission_is_not_replayed(request_file: InferenceRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    methods = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        raise httpx.ReadTimeout("synthetic ambiguous submission")

    provider = mock_live(monkeypatch, handler)
    with pytest.raises(ProviderPermanentError, match="uncertain"):
        provider.run_inference(PIPELINE, request_file)
    provider = LLMParserProvider("llmparser", PIPELINE.config)
    with pytest.raises(ProviderPermanentError, match="uncertain"):
        provider.run_inference(PIPELINE, request_file)
    assert methods == ["POST"]


def test_polling_deadline_preserves_existing_operation(
    request_file: InferenceRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    methods = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        if request.method == "POST":
            return httpx.Response(202, headers={"Operation-Location": OPERATION})
        return httpx.Response(200, json={"status": "running"})

    provider = mock_live(monkeypatch, handler)
    ticks = iter([0, 0, 601])
    monkeypatch.setattr(llmparser.time, "monotonic", lambda: next(ticks, 601))
    with pytest.raises(ProviderPermanentError, match="polling timed out") as error:
        provider.run_inference(PIPELINE, request_file)
    assert error.value.debug_payload["operationLocation"] == OPERATION
    assert methods == ["POST", "GET"]


@pytest.mark.parametrize("kind", ["cross-origin", "submission-redirect", "poll-redirect", "failed"])
def test_unsafe_or_failed_operations_never_redirect_or_resubmit(
    kind: str, request_file: InferenceRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "POST":
            if kind == "submission-redirect":
                return httpx.Response(307, headers={"Location": "https://other.example/steal"})
            operation = "https://other.example/steal" if kind == "cross-origin" else OPERATION
            return httpx.Response(202, headers={"Operation-Location": operation})
        if kind == "poll-redirect":
            return httpx.Response(307, headers={"Location": "https://other.example/steal"})
        return httpx.Response(200, json={"status": "failed", "error": {"code": "InferenceFailed"}})

    provider = mock_live(monkeypatch, handler)
    for _ in range(2):
        with pytest.raises(ProviderPermanentError):
            provider.run_inference(PIPELINE, request_file)
    assert sum(request.method == "POST" for request in requests) == 1
    assert all(request.url.host == "llmparser.example" for request in requests)
