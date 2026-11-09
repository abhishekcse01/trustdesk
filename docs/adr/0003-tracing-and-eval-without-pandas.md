# ADR 0003: Vendor-neutral tracing and in-house eval metrics

Date: 2026-11-09 · Status: accepted

## Context

Phase 2 planned Arize Phoenix for tracing and RAGAS / DeepEval for metrics. On the development
laptop, Windows Smart App Control blocks pandas' compiled parser, and all three depend on pandas.
Phoenix also needs grpc, which is blocked too (ADR 0002). OpenTelemetry's SDK and OTLP/HTTP
exporter load fine.

## Decision

- **Tracing:** instrument with the OpenTelemetry SDK using OpenInference attribute names
  (`openinference.span.kind`, `llm.token_count.*`, `retrieval.documents.*`). Each answer is one
  trace: CHAIN → RETRIEVER + LLM spans. Spans always go to `reports/traces/*.jsonl`; setting the
  standard `OTEL_EXPORTER_OTLP_ENDPOINT` / `OTEL_EXPORTER_OTLP_HEADERS` variables also exports them
  to Phoenix Cloud, Langfuse or any OTLP/HTTP collector with no code change.
- **Metrics:** implement them in `trustdesk.evals`: retrieval hit@3 and MRR against reference
  pages, verbatim citation accuracy, abstention rates, and an LLM judge for correctness (against
  a reference) and faithfulness (RAGAS-style claim splitting, checked against retrieved context).
- **Judge reliability (R3):** the judge runs on an API model, parse failures count as missing,
  and `trustdesk agreement` reports judge-vs-human agreement and Cohen's kappa on 50 hand labels.

## Consequences

- No local trace UI on this laptop; the JSONL is readable, and a hosted backend is one env var away.
- Metric code is small and covered by unit tests, which makes it easy to explain in interviews.
- RAGAS can still be added later as a CI-only job on Linux for comparison.
