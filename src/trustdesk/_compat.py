"""Workarounds for Windows Smart App Control (roadblock R16, ADR 0002).

Smart App Control blocks some unsigned native wheels on the development laptop. Each shim
below replaces only the piece TrustDesk's dependencies actually use, and only when the real
library fails to load, so Linux CI and other machines keep the real implementations.

- grpcio's `cygrpc`: ChromaDB imports it only to export OpenTelemetry spans over gRPC, which
  TrustDesk never enables (tracing uses OTLP over HTTP). We register a stand-in exporter.
- uuid-utils' `_uuid_utils`: langchain-core and langsmith (pulled in by LangGraph) call only
  `uuid_utils.compat.uuid7(timestamp=None, nanos=None)`. We provide a pure-Python RFC 9562 uuid7.
"""

from __future__ import annotations

import os
import sys
import time
import types
import uuid
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


def uuid7(timestamp: float | None = None, nanos: int | None = None) -> uuid.UUID:
    """RFC 9562 UUID version 7: 48-bit Unix time in ms, version, variant, 74 random bits."""
    if timestamp is None:
        ms = time.time_ns() // 1_000_000
    else:
        ms = int(timestamp) * 1000 + (nanos or 0) // 1_000_000
    rand = int.from_bytes(os.urandom(10), "big")  # 80 bits, 74 used
    value = (ms & ((1 << 48) - 1)) << 80
    value |= 0x7 << 76  # version 7
    value |= ((rand >> 62) & 0xFFF) << 64  # rand_a, 12 bits
    value |= 0b10 << 62  # RFC 4122 variant
    value |= rand & ((1 << 62) - 1)  # rand_b, 62 bits
    return uuid.UUID(int=value)


def ensure_uuid_utils_importable() -> bool:
    """Return True if the pure-Python uuid7 stand-in was installed."""
    try:
        import uuid_utils.compat  # noqa: F401
    except ImportError:
        for name in [m for m in sys.modules if m == "uuid_utils" or m.startswith("uuid_utils.")]:
            del sys.modules[name]  # drop the half-imported real package
        pkg = types.ModuleType("uuid_utils")
        compat = types.ModuleType("uuid_utils.compat")
        compat.uuid7 = uuid7  # type: ignore[attr-defined]
        pkg.compat = compat  # type: ignore[attr-defined]
        pkg.uuid7 = uuid7  # type: ignore[attr-defined]
        sys.modules["uuid_utils"] = pkg
        sys.modules["uuid_utils.compat"] = compat
        return True
    return False


def apply_all() -> None:
    ensure_chromadb_importable()
    ensure_uuid_utils_importable()
