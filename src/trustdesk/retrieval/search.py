"""Retrieval entry point: dense, or hybrid (dense + BM25 fused with RRF), optionally reranked.

    dense top-N ─┐
                 ├─ reciprocal rank fusion ─> candidates ─> cross-encoder rerank ─> top-k
    BM25  top-N ─┘

Each stage is a setting in config/settings.yaml (retrieval.mode, retrieval.rerank), so the
experiment grid can compare them without code changes.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Any

from trustdesk.config import Settings
from trustdesk.ingest.index import client, embed
from trustdesk.retrieval import bm25
from trustdesk.retrieval.dense import ChunkHit
from trustdesk.telemetry import INPUT, span


def rrf(rankings: Sequence[Sequence[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal rank fusion: score(d) = sum over lists of 1 / (k + rank of d)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def _hit(cid: str, doc: str, meta: dict[str, Any], score: float) -> ChunkHit:
    return ChunkHit(
        chunk_id=cid,
        text=doc,
        url=str(meta.get("url", "")),
        title=str(meta.get("title", "")),
        heading_path=str(meta.get("heading_path", "")),
        score=score,
    )


def dense(query: str, cfg: Settings, n: int) -> list[ChunkHit]:
    col = client(cfg).get_collection(cfg.collection)
    res = col.query(query_embeddings=[embed([query], cfg.embedding.model)[0]], n_results=n)
    return [
        _hit(cid, doc, meta or {}, 1.0 - float(dist))
        for cid, doc, meta, dist in zip(
            res["ids"][0],
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0],
            strict=True,
        )
    ]


def keyword(query: str, cfg: Settings, n: int) -> list[ChunkHit]:
    idx = bm25.load(cfg)
    return [_hit(idx.ids[i], idx.docs[i], idx.metas[i], s) for i, s in idx.top(query, n)]


@lru_cache(maxsize=2)
def _cross_encoder(model: str) -> Any:
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model_name=model)


def rerank(query: str, hits: list[ChunkHit], model: str) -> list[ChunkHit]:
    if not hits:
        return hits
    scores = list(_cross_encoder(model).rerank(query, [h.text for h in hits]))
    ranked = sorted(zip(hits, scores, strict=True), key=lambda p: float(p[1]), reverse=True)
    return [h.model_copy(update={"score": float(s)}) for h, s in ranked]


def retrieve(query: str, cfg: Settings, k: int | None = None) -> list[ChunkHit]:
    """Run the configured pipeline without tracing (used by the experiment grid)."""
    r = cfg.retrieval
    k = k or r.top_k
    n = max(r.candidates, k)
    if r.mode == "hybrid":
        d, b = dense(query, cfg, n), keyword(query, cfg, n)
        pool = {h.chunk_id: h for h in [*b, *d]}
        fused = rrf([[h.chunk_id for h in d], [h.chunk_id for h in b]], r.rrf_k)[:n]
        hits = [pool[cid].model_copy(update={"score": s}) for cid, s in fused]
    else:
        hits = dense(query, cfg, n if r.rerank else k)
    if r.rerank:
        hits = rerank(query, hits, r.reranker_model)
    return hits[:k]


def search(query: str, cfg: Settings, k: int | None = None) -> list[ChunkHit]:
    """Traced retrieval used by the answerer."""
    r = cfg.retrieval
    with span(
        "retrieve",
        "RETRIEVER",
        **{
            INPUT: query,
            "retrieval.top_k": k or r.top_k,
            "retrieval.mode": r.mode,
            "retrieval.rerank": r.rerank,
        },
    ) as s:
        hits = retrieve(query, cfg, k)
        # O3: log chunk ids and scores so a bad answer can be traced to bad retrieval.
        for i, h in enumerate(hits):
            s.set_attribute(f"retrieval.documents.{i}.document.id", h.chunk_id)
            s.set_attribute(f"retrieval.documents.{i}.document.score", round(h.score, 4))
            s.set_attribute(f"retrieval.documents.{i}.document.metadata", h.url)
        return hits
