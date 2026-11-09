import json
from pathlib import Path

import pytest

from trustdesk.evals.metrics import cohen_kappa, hit_at_k, reciprocal_rank, summarize
from trustdesk.llm import inline_schema
from trustdesk.telemetry import JsonlSpanExporter


def test_hit_at_k_and_mrr() -> None:
    retrieved = ["a", "b", "c", "d"]
    assert hit_at_k(retrieved, {"c"}, 3) == 1.0
    assert hit_at_k(retrieved, {"d"}, 3) == 0.0
    assert hit_at_k(retrieved, set(), 3) == 0.0
    assert reciprocal_rank(retrieved, {"b", "d"}) == 0.5
    assert reciprocal_rank(retrieved, {"z"}) == 0.0


def row(label: str, abstained: bool, correctness: str | None, faith: float | None = 1.0) -> dict:
    r = {
        "label": label,
        "retrieved_urls": ["u1", "u2", "u3"],
        "expected_urls": ["u2"] if label != "unanswerable" else [],
        "abstained": abstained,
        "citations_verified": 1,
        "citations_total": 2,
        "seconds": 10.0,
        "error": None,
    }
    if correctness:
        r["judge"] = {"correctness": correctness, "faithfulness": faith}
    return r


def test_summarize_headline_metrics() -> None:
    s = summarize(
        [
            row("answerable", False, "correct"),
            row("partial", False, "partially_correct", 0.5),
            row("unanswerable", True, "correct", None),
            row("unanswerable", False, "incorrect", 0.0),
        ]
    )
    assert s["questions"] == 4
    assert s["retrieval_hit_at_3"] == 1.0 and s["retrieval_mrr"] == 0.5
    assert s["citation_accuracy"] == 0.5
    assert s["correct_abstention_rate"] == 0.5
    assert s["false_abstention_rate"] == 0.0
    assert s["correctness"] == 0.625  # (1 + 0.5 + 1 + 0) / 4
    assert s["faithfulness"] == 0.5  # mean of 1.0, 0.5, 0.0; None skipped
    assert s["correctness_by_label"]["unanswerable"] == 0.5


def test_cohen_kappa() -> None:
    assert cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == 0.0
    with pytest.raises(ValueError):
        cohen_kappa([], [])


def test_inline_schema_resolves_refs() -> None:
    schema = {
        "type": "object",
        "properties": {"c": {"type": "array", "items": {"$ref": "#/$defs/Cit"}}},
        "$defs": {"Cit": {"type": "object", "properties": {"q": {"type": "string"}}}},
    }
    out = inline_schema(schema)
    assert "$defs" not in out
    assert out["properties"]["c"]["items"]["properties"]["q"]["type"] == "string"


def test_jsonl_exporter_writes_spans(tmp_path: Path) -> None:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    file = tmp_path / "t.jsonl"
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(JsonlSpanExporter(file)))
    tr = provider.get_tracer("test")
    with tr.start_as_current_span("parent") as p:
        p.set_attribute("openinference.span.kind", "CHAIN")
        with tr.start_as_current_span("child"):
            pass
    spans = [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines()]
    assert [s["name"] for s in spans] == ["child", "parent"]
    assert spans[0]["parent_id"] == spans[1]["span_id"]
    assert spans[1]["attributes"]["openinference.span.kind"] == "CHAIN"
