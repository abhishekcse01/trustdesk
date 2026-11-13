# ADR 0002: Working with Windows Smart App Control

Date: 2026-11-09 · Status: accepted

## Context

Smart App Control is on for the development laptop. It blocked:

- the `.exe` launchers uv creates in `.venv\Scripts` (pytest, mypy, trustdesk);
- grpcio's `cygrpc` DLL (tried 1.62.3, 1.66.2, 1.68.1 and the latest), which ChromaDB imports
  only for its gRPC OpenTelemetry exporter;
- mypy's mypyc-compiled DLL.

Everything else (pydantic-core, numpy, onnxruntime, tokenizers, chromadb's Rust bindings) loads.
Turning Smart App Control off cannot be undone without reinstalling Windows, so it stays on.

## Decision

- Run tools as modules: `uv run python -m pytest`, `uv run python -m mypy`,
  `uv run python -m trustdesk ...` (added `trustdesk/__main__.py`).
- `trustdesk._compat.ensure_chromadb_importable()` registers a stand-in for ChromaDB's gRPC span
  exporter when grpc cannot load. TrustDesk never uses that exporter; Phoenix tracing (phase 2)
  uses OTLP over HTTP.
- `[tool.uv] no-binary-package = ["mypy"]` installs mypy as pure Python.

## Consequences

- Slightly slower mypy. CI on Ubuntu is unaffected and runs the normal commands.
- If a future dependency ships a blocked DLL, check it with the import probe before adopting it.

## Update 2026-11-13: uuid-utils (phase 4)

LangGraph pulls in langchain-core and langsmith, which import `uuid_utils` for `uuid7()`. Its
Rust DLL is blocked (tried 0.10.0, 0.11.0, 0.14.0 and 0.19.0). Both libraries call only
`uuid_utils.compat.uuid7(timestamp=None, nanos=None)`, so `trustdesk._compat` registers a
pure-Python RFC 9562 uuid7 when the real module fails to load. All shims now run from
`trustdesk/__init__.py`, before anything imports chromadb or langgraph; `tests/conftest.py`
imports trustdesk first, and pytest runs with `-p no:langsmith_plugin` because that plugin
loads before conftest. Pandas is also blocked (ADR 0003).
