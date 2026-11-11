"""Shared retrieval types. The pipeline (dense, BM25, RRF, reranking) is in retrieval.search."""

from __future__ import annotations

from pydantic import BaseModel


class ChunkHit(BaseModel):
    chunk_id: str
    text: str
    url: str
    title: str
    heading_path: str
    score: float  # cosine similarity, RRF score or reranker score, depending on the stage
