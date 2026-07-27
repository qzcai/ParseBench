from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from pypdf import PdfWriter

from parse_bench.inference.providers.base import ProviderConfigError
from parse_bench.inference.providers.parse.mistral_ocr import (
    MistralOCRProvider,
    _MISTRAL_OCR_URL,
)
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest
from parse_bench.schemas.product import ProductType


class MistralOCRTransportTests(unittest.TestCase):
    def _request_for(self, provider_name: str, endpoint: str | None):
        response = Mock(status_code=200, headers={})
        response.json.return_value = {"pages": [], "usage_info": {}}
        environment = {
            "MISTRAL_API_KEY": "test-key",
            "MISTRAL_OCR_PROVIDER": provider_name,
        }
        if endpoint is not None:
            environment["MISTRAL_API_ENDPOINT"] = endpoint
        with tempfile.TemporaryDirectory() as directory:
            document = Path(directory) / "document.pdf"
            writer = PdfWriter()
            writer.add_blank_page(width=72, height=72)
            with document.open("wb") as stream:
                writer.write(stream)
            pipeline = PipelineSpec(
                pipeline_name="mistral_ocr_4",
                provider_name="mistral_ocr",
                product_type=ProductType.PARSE,
                config={"model": "mistral-ocr-4-0", "include_blocks": True},
            )
            request = InferenceRequest(
                example_id="transport",
                source_file_path=str(document),
                product_type=ProductType.PARSE,
            )
            with (
                patch.dict("os.environ", environment, clear=True),
                patch(
                    "parse_bench.inference.providers.parse.mistral_ocr.requests.post",
                    return_value=response,
                ) as post,
            ):
                MistralOCRProvider("mistral_ocr", pipeline.config).run_inference(
                    pipeline,
                    request,
                )
        return post.call_args

    def test_foundry_changes_only_the_request_url(self) -> None:
        direct = self._request_for("mistral", None)
        foundry = self._request_for(
            "azure",
            "https://example.services.ai.azure.com",
        )

        self.assertEqual(direct.args, (_MISTRAL_OCR_URL,))
        self.assertEqual(
            foundry.args,
            ("https://example.services.ai.azure.com/providers/mistral/azure/ocr",),
        )
        self.assertEqual(direct.kwargs, foundry.kwargs)
        self.assertTrue(foundry.kwargs["json"]["include_blocks"])

    def test_azure_requires_endpoint(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "MISTRAL_API_KEY": "test-key",
                "MISTRAL_OCR_PROVIDER": "azure",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ProviderConfigError, "MISTRAL_API_ENDPOINT"):
                MistralOCRProvider("mistral_ocr")

    def test_rejects_unknown_provider(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "MISTRAL_API_KEY": "test-key",
                "MISTRAL_OCR_PROVIDER": "other",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ProviderConfigError, "mistral.*azure"):
                MistralOCRProvider("mistral_ocr")


if __name__ == "__main__":
    unittest.main()
