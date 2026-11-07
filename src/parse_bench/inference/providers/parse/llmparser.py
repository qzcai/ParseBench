"""LLMParser DI-compatible transport and Canonical17 region normalization."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from parse_bench.inference.providers.base import Provider, ProviderConfigError, ProviderPermanentError
from parse_bench.inference.providers.registry import register_provider
from parse_bench.schemas.layout_ontology import CanonicalLabel
from parse_bench.schemas.parse_output import LayoutItemIR, LayoutSegmentIR, PageIR, ParseLayoutPageIR, ParseOutput
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult, RawInferenceResult
from parse_bench.schemas.product import ProductType

API_VERSION = "2024-11-30"
MODEL_ID = "prebuilt-layout"
OPERATION_PREFIX = f"/documentintelligence/documentModels/{MODEL_ID}/analyzeResults/"
MAX_DOCUMENT_BYTES = 25 * 1024 * 1024


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ProviderConfigError(f"LLMParser {name} must be a finite positive number.")
    return float(value)


def _endpoint(value: Any) -> str:
    if not isinstance(value, str):
        raise ProviderConfigError("LLMPARSER_ENDPOINT must be an HTTPS origin.")
    try:
        parsed = urlsplit(value)
        if parsed.port == 0:
            raise ProviderConfigError("LLMPARSER_ENDPOINT requires a positive port.")
    except ValueError as error:
        raise ProviderConfigError("LLMPARSER_ENDPOINT has an invalid hostname or port.") from error
    if (
        not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
        or not (
            parsed.scheme == "https" or parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
        )
    ):
        raise ProviderConfigError("LLMPARSER_ENDPOINT must be an HTTPS origin (HTTP loopback is allowed).")
    return value.rstrip("/")


def _confined_path(root: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute() or "\\" in relative:
        raise ProviderConfigError("LLMParser manifest inputPath must be a relative POSIX path.")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ProviderConfigError("LLMParser precomputed response path escapes its manifest directory.")
    return path


def _load_precomputed(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise ProviderConfigError("LLMPARSER_PRECOMPUTED_DIR must be an existing directory.")
    responses: dict[str, Path] = {}
    try:
        for manifest_path in sorted(root.rglob("manifest.json")):
            if not manifest_path.resolve().is_relative_to(root):
                raise ProviderConfigError("LLMParser manifest escapes its precomputed directory.")
            manifest = json.loads(manifest_path.read_bytes())
            if manifest.get("schemaVersion") != 1:
                raise ProviderConfigError("LLMParser precomputed manifest requires schemaVersion=1.")
            for entry in manifest["entries"]:
                digest = entry["sha256"]
                if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                    raise ProviderConfigError("LLMParser manifest sha256 must be a lowercase SHA-256 digest.")
                relative = entry["inputPath"]
                if not isinstance(relative, str):
                    raise ProviderConfigError("LLMParser manifest inputPath must be a string.")
                response = _confined_path(manifest_path.parent.resolve(), relative + ".json")
                if digest in responses and responses[digest] != response:
                    raise ProviderConfigError("LLMParser precomputed manifest has an ambiguous duplicate input hash.")
                responses[digest] = response
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise ProviderConfigError("Cannot load the LLMParser precomputed manifest.") from error
    if not responses:
        raise ProviderConfigError("LLMParser precomputed directory contains no manifest entries.")
    return responses


def _result(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ProviderPermanentError("LLMParser response must be a JSON object.")
    if "status" in payload:
        if payload["status"] != "succeeded":
            raise ProviderPermanentError(
                "LLMParser operation did not succeed; no new submission will be attempted.",
                debug_payload=payload,
            )
        payload = payload.get("analyzeResult")
    if not isinstance(payload, dict) or not isinstance(payload.get("_llmparser"), dict):
        raise ProviderPermanentError("LLMParser response is missing its region metadata.")
    return payload


def _page_markdown(content: str, spans: Any, index_type: str) -> str:
    if not isinstance(spans, list) or not spans:
        raise ProviderPermanentError("LLMParser page is missing its content spans.")
    encoded = content.encode("utf-16-le") if index_type == "utf16CodeUnit" else None
    length = len(encoded) // 2 if encoded is not None else len(content)
    fragments = []
    for span in spans:
        if not isinstance(span, dict):
            raise ProviderPermanentError("LLMParser page span must be an object.")
        offset, size = span.get("offset"), span.get("length")
        if type(offset) is not int or type(size) is not int or offset < 0 or size < 0 or offset + size > length:
            raise ProviderPermanentError("LLMParser page span is outside the document content.")
        try:
            fragment = (
                encoded[offset * 2 : (offset + size) * 2].decode("utf-16-le")
                if encoded is not None
                else content[offset : offset + size]
            )
        except UnicodeDecodeError as error:
            raise ProviderPermanentError("LLMParser page span splits a UTF-16 character.") from error
        fragments.append(fragment)
    return "".join(fragments)


def normalize_llmparser_result(result: dict[str, Any], example_id: str, pipeline_name: str) -> ParseOutput:
    """Preserve document Markdown and predicted Canonical17 regions without inventing geometry."""
    if result.get("modelId") != MODEL_ID or result.get("apiVersion") != API_VERSION:
        raise ProviderPermanentError("LLMParser result has an unsupported DI model or API version.")
    content = result.get("content")
    if result.get("contentFormat") != "markdown" or not isinstance(content, str) or not content.strip():
        raise ProviderPermanentError("LLMParser result requires nonempty Markdown.")
    index_type = result.get("stringIndexType")
    if not isinstance(index_type, str) or index_type not in {"unicodeCodePoint", "utf16CodeUnit"}:
        raise ProviderPermanentError("LLMParser result has an unsupported stringIndexType.")
    raw_pages = result.get("pages")
    if not isinstance(raw_pages, list) or not raw_pages or any(not isinstance(page, dict) for page in raw_pages):
        raise ProviderPermanentError("LLMParser result is missing document pages.")
    numbers = [page.get("pageNumber") for page in raw_pages]
    if any(type(number) is not int for number in numbers) or numbers != list(range(1, len(numbers) + 1)):
        raise ProviderPermanentError("LLMParser pages must have ordered, contiguous page numbers.")
    pages = [
        PageIR(
            page_index=page["pageNumber"] - 1,
            markdown=_page_markdown(content, page.get("spans"), index_type),
        )
        for page in raw_pages
    ]
    metadata = result.get("_llmparser")
    if not isinstance(metadata, dict):
        raise ProviderPermanentError("LLMParser result is missing region metadata.")
    if metadata.get("coordinateSystem", "normalized") != "normalized" or metadata.get("bboxFormat", "xywh") != "xywh":
        raise ProviderPermanentError("LLMParser regions must use normalized xywh coordinates.")
    regions = metadata.get("regions")
    if regions is None:
        receipts = metadata.get("receipts")
        if not isinstance(receipts, list):
            raise ProviderPermanentError("LLMParser result is missing predicted regions.")
        regions = []
        for receipt in receipts:
            if (
                not isinstance(receipt, dict)
                or not isinstance(receipt.get("elements"), list)
                or any(not isinstance(element, dict) for element in receipt["elements"])
            ):
                raise ProviderPermanentError("LLMParser receipt is missing predicted elements.")
            regions.extend({"pageNumber": receipt.get("page"), **element} for element in receipt["elements"])
    if not isinstance(regions, list):
        raise ProviderPermanentError("LLMParser regions must be an array.")
    items: dict[int, list[LayoutItemIR]] = {number: [] for number in numbers}
    for region in regions:
        if not isinstance(region, dict) or type(region.get("pageNumber")) is not int:
            raise ProviderPermanentError("LLMParser region requires an integer pageNumber.")
        number = region["pageNumber"]
        if number not in items:
            raise ProviderPermanentError("LLMParser region references a missing page.")
        label, text, box = region.get("label"), region.get("text"), region.get("bbox")
        if not isinstance(label, str) or not isinstance(text, str):
            raise ProviderPermanentError("LLMParser region requires a label and text.")
        try:
            label = CanonicalLabel(label).value
        except ValueError as error:
            raise ProviderPermanentError("LLMParser region has an unknown Canonical17 label.") from error
        if (
            not isinstance(box, (list, tuple))
            or len(box) != 4
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                for value in box
            )
        ):
            raise ProviderPermanentError("LLMParser region requires four finite bbox coordinates.")
        x, y, width, height = box
        if min(x, y) < 0 or min(width, height) <= 0 or x + width > 1.001 or y + height > 1.001:
            raise ProviderPermanentError("LLMParser region bbox must be normalized positive-area xywh.")
        attributes = region.get("attributes", {})
        if not isinstance(attributes, dict) or any(
            not isinstance(key, str) or not isinstance(value, str) for key, value in attributes.items()
        ):
            raise ProviderPermanentError("LLMParser region attributes must be a string-valued object.")
        segment = LayoutSegmentIR(x=x, y=y, w=width, h=height, label=label)
        items[number].append(
            LayoutItemIR(
                type=label, md=text, value=text, bbox=segment, layout_segments=[segment], attributes=attributes
            )
        )
    if any(not page_items for page_items in items.values()):
        raise ProviderPermanentError("LLMParser output lacks predicted layout regions on one or more pages.")
    return ParseOutput(
        example_id=example_id,
        pipeline_name=pipeline_name,
        markdown=content,
        pages=pages,
        layout_pages=[
            ParseLayoutPageIR(page_number=number, width=1.0, height=1.0, items=items[number]) for number in numbers
        ],
    )


@register_provider("llmparser")
class LLMParserProvider(Provider):
    """Call the DI-compatible LLMParser API or strictly replay exact-byte saved results."""

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)
        allowed = {
            "model",
            "reasoning_effort",
            "endpoint",
            "precomputed_dir",
            "operation_dir",
            "poll_timeout",
            "poll_interval",
        }
        if set(self.base_config) - allowed:
            raise ProviderConfigError("LLMParser configuration contains unknown keys.")
        self._model = self.base_config.get("model", "gpt-6.1-terra")
        self._reasoning = self.base_config.get("reasoning_effort", "low")
        if (
            not isinstance(self._model, str)
            or not self._model
            or not isinstance(self._reasoning, str)
            or self._reasoning not in {"low", "medium", "high"}
        ):
            raise ProviderConfigError("LLMParser requires a model name and a supported reasoning_effort.")
        self._poll_timeout = _positive_number(self.base_config.get("poll_timeout", 600), "poll_timeout")
        self._poll_interval = _positive_number(self.base_config.get("poll_interval", 1), "poll_interval")
        self._submission_lock = threading.Lock()
        precomputed = self.base_config.get("precomputed_dir", os.getenv("LLMPARSER_PRECOMPUTED_DIR"))
        if precomputed is not None and (not isinstance(precomputed, str) or not precomputed.strip()):
            raise ProviderConfigError("LLMPARSER_PRECOMPUTED_DIR must be a nonempty directory path.")
        self._responses = (
            _load_precomputed(Path(precomputed).expanduser().resolve()) if precomputed is not None else None
        )
        self._origin: str | None = None
        self._key: str | None = None
        self._operation_dir: Path | None = None
        if self._responses is None:
            self._origin = _endpoint(self.base_config.get("endpoint") or os.getenv("LLMPARSER_ENDPOINT"))
            self._key = os.getenv("LLMPARSER_API_KEY", "").strip()
            if not self._key:
                raise ProviderConfigError("LLMPARSER_API_KEY is required for live LLMParser inference.")
            operation_dir = self.base_config.get("operation_dir", os.getenv("LLMPARSER_OPERATION_DIR"))
            if not isinstance(operation_dir, str) or not operation_dir.strip():
                raise ProviderConfigError("LLMPARSER_OPERATION_DIR is required for safe live inference resumption.")
            self._operation_dir = Path(operation_dir).expanduser().resolve()
            try:
                self._operation_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            except OSError as error:
                raise ProviderConfigError("Cannot create the LLMParser operation checkpoint directory.") from error

    def _operation_url(self, value: str | None) -> str:
        if not value or self._origin is None:
            raise ProviderPermanentError("LLMParser accepted the request without a polling URL.")
        try:
            expected, actual = urlsplit(self._origin), urlsplit(value)
            if actual.port == 0:
                raise ValueError("Invalid operation port")
        except ValueError as error:
            raise ProviderPermanentError("LLMParser returned an invalid polling URL.") from error
        if (
            (actual.scheme, actual.netloc) != (expected.scheme, expected.netloc)
            or actual.username
            or actual.password
            or actual.fragment
            or not actual.path.startswith(OPERATION_PREFIX)
        ):
            raise ProviderPermanentError("LLMParser returned an unsafe polling URL; credentials were not forwarded.")
        return value

    def _submitted_operation(self, client: httpx.Client, document: bytes, digest: str) -> tuple[str, Path]:
        assert self._origin is not None and self._key is not None and self._operation_dir is not None
        identity = {
            "endpoint": self._origin,
            "apiVersion": API_VERSION,
            "model": self._model,
            "reasoningEffort": self._reasoning,
            "inputSha256": digest,
        }
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        checkpoint = self._operation_dir / f"{fingerprint}.operation.json"
        debug = {"operationCheckpoint": str(checkpoint), "inputSha256": digest, "automaticSubmissionRetries": 0}
        with self._submission_lock:
            try:
                descriptor = os.open(checkpoint, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                try:
                    state = json.loads(checkpoint.read_bytes())
                except (OSError, ValueError) as error:
                    raise ProviderPermanentError(
                        "Cannot read the LLMParser operation checkpoint.", debug_payload=debug
                    ) from error
                if not isinstance(state, dict) or state.get("schemaVersion") != 1 or state.get("request") != identity:
                    raise ProviderPermanentError(
                        "LLMParser operation checkpoint does not match this request.", debug_payload=debug
                    ) from None
                if not isinstance(state.get("operationLocation"), str):
                    raise ProviderPermanentError(
                        "Previous LLMParser submission is uncertain; reconcile its checkpoint before resubmitting.",
                        debug_payload=debug,
                    ) from None
                return self._operation_url(state["operationLocation"]), checkpoint
            except OSError as error:
                raise ProviderPermanentError(
                    "Cannot claim the LLMParser submission checkpoint.", debug_payload=debug
                ) from error
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    json.dump({"schemaVersion": 1, "request": identity, "status": "submitting"}, handle)
                    handle.flush()
                    os.fsync(handle.fileno())
            except OSError as error:
                raise ProviderPermanentError(
                    "Cannot persist the LLMParser submission claim.", debug_payload=debug
                ) from error
            try:
                response = client.post(
                    f"{self._origin}/documentintelligence/documentModels/{MODEL_ID}:analyze",
                    params={
                        "api-version": API_VERSION,
                        "outputContentFormat": "markdown",
                        "stringIndexType": "unicodeCodePoint",
                        "llmparserModel": self._model,
                        "llmparserReasoningEffort": self._reasoning,
                    },
                    headers={"Ocp-Apim-Subscription-Key": self._key, "Content-Type": "application/pdf"},
                    content=document,
                )
            except httpx.HTTPError as error:
                raise ProviderPermanentError(
                    "LLMParser submission failed with an uncertain outcome; no POST retry was attempted.",
                    debug_payload=debug,
                ) from error
            if response.status_code != 202:
                raise ProviderPermanentError(
                    f"LLMParser submission returned HTTP {response.status_code}.", debug_payload=debug
                )
            operation_url = self._operation_url(response.headers.get("Operation-Location"))
            temporary: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=self._operation_dir, prefix=".operation-", delete=False
                ) as handle:
                    temporary = Path(handle.name)
                    json.dump(
                        {
                            "schemaVersion": 1,
                            "request": identity,
                            "status": "submitted",
                            "operationLocation": operation_url,
                        },
                        handle,
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, checkpoint)
            except OSError as error:
                raise ProviderPermanentError(
                    "Cannot persist the accepted LLMParser operation; do not submit again.",
                    debug_payload={**debug, "operationLocation": operation_url},
                ) from error
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            return operation_url, checkpoint

    def _analyze(self, document: bytes, digest: str) -> dict[str, Any]:
        assert self._origin is not None and self._key is not None
        headers = {"Ocp-Apim-Subscription-Key": self._key}
        with httpx.Client(timeout=30, follow_redirects=False) as client:
            operation_url, checkpoint = self._submitted_operation(client, document, digest)
            debug = {
                "operationLocation": operation_url,
                "operationCheckpoint": str(checkpoint),
                "inputSha256": digest,
                "automaticSubmissionRetries": 0,
            }
            deadline = time.monotonic() + self._poll_timeout
            while time.monotonic() < deadline:
                try:
                    response = client.get(operation_url, headers=headers)
                except httpx.HTTPError as error:
                    raise ProviderPermanentError(
                        "LLMParser polling failed; resume the existing operation, do not submit again.",
                        debug_payload=debug,
                    ) from error
                if response.status_code != 200:
                    raise ProviderPermanentError(
                        f"LLMParser polling returned HTTP {response.status_code}.", debug_payload=debug
                    )
                try:
                    payload = response.json()
                except ValueError as error:
                    raise ProviderPermanentError(
                        "LLMParser polling returned invalid JSON.", debug_payload=debug
                    ) from error
                if not isinstance(payload, dict):
                    raise ProviderPermanentError("LLMParser operation must be an object.", debug_payload=debug)
                status = payload.get("status")
                if not isinstance(status, str):
                    raise ProviderPermanentError("LLMParser operation is missing its status.", debug_payload=debug)
                if status in {"running", "notStarted"}:
                    remaining = deadline - time.monotonic()
                    if remaining > 0:
                        time.sleep(min(self._poll_interval, remaining))
                    continue
                try:
                    result = _result(payload)
                except ProviderPermanentError as error:
                    error.debug_payload = {**debug, "operation": payload}
                    raise
                return {
                    **result,
                    "_operation": {
                        **{key: value for key, value in payload.items() if key != "analyzeResult"},
                        "operationLocation": operation_url,
                    },
                }
            raise ProviderPermanentError(
                "LLMParser polling timed out; resume the existing operation, do not submit again.", debug_payload=debug
            )

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError("LLMParser only supports PARSE.")
        if request.config_override:
            raise ProviderConfigError("LLMParser requires a complete pipeline, not per-case configuration overrides.")
        started = datetime.now()
        try:
            document = Path(request.source_file_path).read_bytes()
        except OSError as error:
            raise ProviderPermanentError("Cannot read the LLMParser input document.") from error
        if not 0 < len(document) <= MAX_DOCUMENT_BYTES:
            raise ProviderPermanentError("LLMParser document must be nonempty and at most 25 MiB.")
        digest = hashlib.sha256(document).hexdigest()
        if self._responses is not None:
            response_path = self._responses.get(digest)
            if response_path is None:
                raise ProviderPermanentError("No precomputed LLMParser result matches the exact input bytes.")
            try:
                result = _result(json.loads(response_path.read_bytes()))
            except (OSError, ValueError) as error:
                raise ProviderPermanentError("Cannot read the precomputed LLMParser response.") from error
            source = "precomputed"
        else:
            result = self._analyze(document, digest)
            source = "live"
        profile = result["_llmparser"].get("profile", {})
        if (
            not isinstance(profile, dict)
            or profile.get("deployment") != self._model
            or profile.get("reasoning_effort") != self._reasoning
        ):
            raise ProviderPermanentError("LLMParser response model/reasoning provenance does not match the pipeline.")
        raw = {
            **result,
            "_config": {
                "provider": "llmparser",
                "model": self._model,
                "reasoning_effort": self._reasoning,
                "api_version": API_VERSION,
                "response_source": source,
                "input_sha256": digest,
            },
        }
        completed = datetime.now()
        return RawInferenceResult(
            request=request,
            pipeline=pipeline,
            pipeline_name=pipeline.pipeline_name,
            product_type=ProductType.PARSE,
            raw_output=raw,
            started_at=started,
            completed_at=completed,
            latency_in_ms=int((completed - started).total_seconds() * 1000),
        )

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        if raw_result.product_type != ProductType.PARSE:
            raise ProviderPermanentError("LLMParser only supports PARSE.")
        output = normalize_llmparser_result(
            raw_result.raw_output, raw_result.request.example_id, raw_result.pipeline_name
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
