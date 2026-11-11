"""Keyword search (BM25) over the same chunks that are in ChromaDB.

Policy questions are full of exact terms (AES-256, TLS 1.2, SOC 2, MFA) that dense embeddings
can blur. BM25 scores exact token overlap, so the two retrievers fail differently and fusing
them helps. The index is built from Chroma's stored documents and cached on disk.
"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from typing import Any

from rank_bm25 import BM25Okapi

from trustdesk.config import Settings, path
from trustdesk.ingest.index import client

_STOP = """a an and are as at be by do does for from has have how in is it its of on or our
that the their this to was we what when where which who why will with you your"""
STOPWORDS = frozenset(_STOP.split())
TOKEN = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens; keeps terms like 'tls-1.2' or 'aes-256' together."""
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


@dataclass
class Bm25Index:
    ids: list[str]
    docs: list[str]
    metas: list[dict[str, Any]]
    bm25: BM25Okapi

    def top(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(i, float(scores[i])) for i in order if scores[i] > 0]


def build(cfg: Settings) -> Bm25Index:
    col = client(cfg).get_collection(cfg.collection)
    data = col.get(include=["documents", "metadatas"])
    index = Bm25Index(
        ids=list(data["ids"]),
        docs=list(data["documents"]),
        metas=[dict(m or {}) for m in data["metadatas"]],
        bm25=BM25Okapi([tokenize(d) for d in data["documents"]]),
    )
    file = path(cfg.data_dir) / "bm25.pkl"
    file.write_bytes(pickle.dumps((col.count(), index)))
    return index


_CACHE: dict[tuple[str, str, int], Bm25Index] = {}


def load(cfg: Settings) -> Bm25Index:
    """Cached index, rebuilt when the Chroma collection's size changes."""
    file = path(cfg.data_dir) / "bm25.pkl"
    count = client(cfg).get_collection(cfg.collection).count()
    key = (cfg.data_dir, cfg.collection, count)
    if key in _CACHE:
        return _CACHE[key]
    index: Bm25Index | None = None
    if file.exists():
        cached_count, cached = pickle.loads(file.read_bytes())  # our own file, from build()
        if cached_count == count:
            index = cached
    _CACHE[key] = index or build(cfg)
    return _CACHE[key]
