"""OpenTelemetry tracing with OpenInference attribute names (O1-O3).

Every answer is one trace: a CHAIN span with child RETRIEVER and LLM spans. Spans always go to
a local JSONL file (reports/traces/, git-ignored). If OTEL_EXPORTER_OTLP_ENDPOINT is set
(Phoenix Cloud, Langfuse, any OTLP/HTTP collector) they are exported there as well.

Local Phoenix is not used on the development laptop because Windows Smart App Control blocks
pandas, which Phoenix needs (see docs/adr/0003).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import Span, Status, StatusCode

from trustdesk.config import load_env, path, settings

# OpenInference semantic conventions (subset).
KIND = "openinference.span.kind"
INPUT = "input.value"
OUTPUT = "output.value"
MODEL = "llm.model_name"
TOK_PROMPT = "llm.token_count.prompt"
TOK_COMPLETION = "llm.token_count.completion"


class JsonlSpanExporter(SpanExporter):
    """Append finished spans to a JSONL file, one span per line."""

    def __init__(self, file: Path) -> None:
        self.file = file
        file.parent.mkdir(parents=True, exist_ok=True)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        with self.file.open("a", encoding="utf-8") as f:
            for s in spans:
                ctx = s.get_span_context()
                parent = s.parent.span_id if s.parent else None
                f.write(
                    json.dumps(
                        {
                            "trace_id": f"{ctx.trace_id:032x}" if ctx else None,
                            "span_id": f"{ctx.span_id:016x}" if ctx else None,
                            "parent_id": f"{parent:016x}" if parent else None,
                            "name": s.name,
                            "start": s.start_time,
                            "end": s.end_time,
                            "duration_ms": round(
                                ((s.end_time or 0) - (s.start_time or 0)) / 1e6, 1
                            ),
                            "status": s.status.status_code.name,
                            "attributes": dict(s.attributes or {}),
                        },
                        ensure_ascii=False,
                        default=str,
                    )
                    + "\n"
                )
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None


@lru_cache(maxsize=1)
def tracer() -> trace.Tracer:
    load_env()
    cfg = settings().tracing
    provider = TracerProvider(resource=Resource.create({"service.name": cfg.service_name}))
    file = path(cfg.jsonl_dir) / f"traces-{datetime.now():%Y%m%d}.jsonl"
    provider.add_span_processor(SimpleSpanProcessor(JsonlSpanExporter(file)))
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        # Endpoint and headers (API key) come from the standard OTEL_* environment variables.
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    return trace.get_tracer("trustdesk")


@contextmanager
def span(name: str, kind: str, **attrs: Any) -> Iterator[Span]:
    """Start a span with an OpenInference kind; records exceptions as errors."""
    with tracer().start_as_current_span(name) as s:
        s.set_attribute(KIND, kind)
        for k, v in attrs.items():
            if v is not None:
                s.set_attribute(k, v if isinstance(v, str | int | float | bool) else json.dumps(v))
        try:
            yield s
        except Exception as exc:
            s.set_status(Status(StatusCode.ERROR, str(exc)[:200]))
            s.record_exception(exc)
            raise


def current_trace_id() -> str:
    ctx = trace.get_current_span().get_span_context()
    return f"{ctx.trace_id:032x}" if ctx and ctx.trace_id else ""


def flush() -> None:
    provider = trace.get_tracer_provider()
    if isinstance(provider, TracerProvider):
        provider.force_flush()
