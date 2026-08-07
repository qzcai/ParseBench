"""Detailed HTML report: the embedded viewer script."""

from __future__ import annotations

import json
import re
from pathlib import Path

from parse_bench.analysis.detailed_report import _build_data_blob, generate_detailed_html_report
from parse_bench.schemas.evaluation import EvaluationResult, EvaluationSummary, MetricValue


def _summary() -> EvaluationSummary:
    return EvaluationSummary(
        total_examples=1,
        successful=1,
        failed=0,
        skipped=0,
        aggregate_metrics={"avg_rule_pass_rate": 0.5, "min_rule_pass_rate": 0.5, "max_rule_pass_rate": 0.5},
        per_example_results=[
            EvaluationResult(
                test_id="group/doc with spaces",
                example_id="group/doc with spaces",
                pipeline_name="p",
                product_type="parse",
                success=True,
                metrics=[
                    MetricValue(
                        metric_name="rule_pass_rate",
                        value=0.5,
                        metadata={
                            "passed": 1,
                            "total": 2,
                            "rule_results": [
                                {"type": "text_presence", "id": "r1", "passed": True, "message": "ok"},
                                {"type": "text_presence", "id": "r2", "passed": False, "message": "missing"},
                            ],
                        },
                    )
                ],
            )
        ],
    )

def _render(tmp_path: Path, **kwargs: object) -> str:
    report_dir = tmp_path / "report"
    report_dir.mkdir(exist_ok=True)
    path = generate_detailed_html_report(_summary(), report_dir, test_cases_dir=tmp_path / "cases", **kwargs)  # type: ignore[arg-type]
    return path.read_text(encoding="utf-8")


def test_pdf_url_builder_encodes_root_relative_bases(tmp_path: Path) -> None:
    """Only ``http(s)://`` bases used to be percent-encoded; a root-relative
    ``/files/...`` base with spaces or unicode in the test id produced a broken
    PDF URL. Both forms now go through ``buildPdfDocumentUrl``."""
    html = _render(tmp_path, pdf_base_url="/files/data")

    assert "function buildPdfDocumentUrl(baseUrl, testId)" in html
    builder = html[html.index("function buildPdfDocumentUrl") :]
    builder = builder[: builder.index("window.loadPdf")]
    assert re.search(r"/\^https\?:\\/\\//i\.test\(baseUrl\) \|\| baseUrl\.charAt\(0\) === '/'", builder)
    assert "encodeURIComponent" in builder
    # loadPdf delegates instead of carrying its own copy of the logic.
    load_pdf = html[html.index("window.loadPdf = function") :]
    load_pdf = load_pdf[: load_pdf.index("function doLoadPdf")]
    assert "buildPdfDocumentUrl(baseUrl, testId)" in load_pdf
    assert "encodeURIComponent" not in load_pdf


def test_rule_results_panel_reports_failed_count_and_auto_expands(tmp_path: Path) -> None:
    """The panel used to render collapsed with a bare rule count; readers open
    an example to see the failures, so it now says ``N failed / M rules`` and
    starts expanded whenever anything failed."""
    html = _render(tmp_path)

    assert "failedRules" in html
    assert "' failed</span> / ' + totalRules + ' rules'" in html
    assert "var ruleOpenClass = failedRules > 0 ? ' open' : '';" in html
    assert "'<div class=\"detail-collapsible-body' + ruleOpenClass + '\">'" in html
    # Both the boolean and the tri-state shapes count as a failure.
    assert "metricRules[fri].status === 'fail' || metricRules[fri].passed === false" in html


def test_expected_output_prefers_sidecar_then_falls_back_to_jsonl(tmp_path):
    (tmp_path / "sample.test.json").write_text(
        json.dumps({"expected_markdown": "sidecar expected"}),
        encoding="utf-8",
    )
    records = [
        {
            "pdf": "docs/table/sample.pdf",
            "expected_markdown": "jsonl should not replace sidecar",
        },
        {
            "pdf": "docs/table/fallback.pdf",
            "expected_markdown": "jsonl expected",
        },
        {
            "pdf": "docs/layout/checks.pdf",
            "type": "layout",
            "rule": json.dumps({"canonical_class": "Title", "ro_index": 0}),
        },
        {
            "pdf": "docs/layout/checks.pdf",
            "type": "layout",
            "rule": json.dumps({"canonical_class": "Text", "ro_index": 1}),
        },
    ]
    (tmp_path / "table.jsonl").write_text(
        "".join(f"{json.dumps(record)}\n" for record in records),
        encoding="utf-8",
    )
    summary = EvaluationSummary(
        total_examples=3,
        successful=3,
        failed=0,
        skipped=0,
        per_example_results=[
            EvaluationResult(
                test_id="table/sample",
                example_id="table/sample",
                pipeline_name="test",
                product_type="parse",
                success=True,
            ),
            EvaluationResult(
                test_id="table/fallback",
                example_id="table/fallback",
                pipeline_name="test",
                product_type="parse",
                success=True,
            ),
            EvaluationResult(
                test_id="layout/checks",
                example_id="layout/checks",
                pipeline_name="test",
                product_type="parse",
                success=True,
            ),
        ],
    )

    examples = _build_data_blob(summary, test_cases_dir=tmp_path)["examples"]

    assert examples[0]["expectedOutput"] == "sidecar expected"
    assert examples[1]["expectedOutput"] == "jsonl expected"
    assert examples[2]["expectedOutput"] == (
        "Evaluation checks (2 checks)\n\n"
        "```json\n"
        "[\n"
        "  {\n"
        '    "type": "layout",\n'
        '    "rule": {\n'
        '      "canonical_class": "Title",\n'
        '      "ro_index": 0\n'
        "    }\n"
        "  },\n"
        "  {\n"
        '    "type": "layout",\n'
        '    "rule": {\n'
        '      "canonical_class": "Text",\n'
        '      "ro_index": 1\n'
        "    }\n"
        "  }\n"
        "]\n"
        "```"
    )


def test_expected_output_uses_checks_from_requested_group(tmp_path):
    for group, rule_type in [
        ("text_content", "missing_specific_word"),
        ("text_formatting", "is_bold"),
    ]:
        (tmp_path / f"{group}.jsonl").write_text(
            json.dumps(
                {
                    "pdf": "docs/text/shared.pdf",
                    "type": rule_type,
                    "rule": json.dumps({"text": "expected"}),
                }
            ),
            encoding="utf-8",
        )

    summary = EvaluationSummary(
        total_examples=1,
        successful=1,
        failed=0,
        skipped=0,
        per_example_results=[
            EvaluationResult(
                test_id="text/shared",
                example_id="text/shared",
                pipeline_name="test",
                product_type="parse",
                success=True,
            )
        ],
    )

    examples = _build_data_blob(summary, test_cases_dir=tmp_path, group="text_formatting")["examples"]

    assert '"type": "is_bold"' in examples[0]["expectedOutput"]
    assert "missing_specific_word" not in examples[0]["expectedOutput"]
