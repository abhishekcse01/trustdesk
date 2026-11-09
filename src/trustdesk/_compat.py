"""Workarounds for Windows Smart App Control (roadblock R16).

Smart App Control blocks some unsigned native wheels. On the development laptop it blocks
grpcio's `cygrpc` DLL, which ChromaDB imports only to export OpenTelemetry spans over gRPC.
TrustDesk never enables that exporter (Phoenix tracing uses OTLP over HTTP), so when grpc
cannot load we register a stand-in exporter module and ChromaDB works normally.
"""

from __future__ import annotations

import sys
import types
from typing import Any

_EXPORTER = "opentelemetry.exporter.otlp.proto.grpc.trace_exporter"


class _DisabledOTLPSpanExporter:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("gRPC OTLP export is unavailable on this machine (Smart App Control).")


def ensure_chromadb_importable() -> bool:
    """Return True if the stand-in was installed, False if real grpc loaded."""
    try:
        import grpc  # noqa: F401
    except ImportError:
        stub = types.ModuleType(_EXPORTER)
        stub.OTLPSpanExporter = _DisabledOTLPSpanExporter  # type: ignore[attr-defined]
        sys.modules[_EXPORTER] = stub
        return True
    return False
