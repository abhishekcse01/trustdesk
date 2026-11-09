"""Dense retrieval over the Chroma index. BM25 + RRF + reranking arrive in phase 3."""

from __future__ import annotations

from pydantic import BaseModel

from trustdesk.config import Settings
from trustdesk.ingest.index import client, embed
from trustdesk.telemetry import INPUT, span


class ChunkHit(BaseModel):
    chunk_id: str
    text: str
    url: str
    title: str
    heading_path: str
    score: float  # cosine similarity, higher is better


def search(query: str, cfg: Settings, k: int | None = None) -> list[ChunkHit]:
    k = k or cfg.retrieval.top_k
    with span("retrieve", "RETRIEVER", **{INPUT: query, "retrieval.top_k": k}) as s:
        col = client(cfg).get_collection(cfg.collection)
        vec = embed([query], cfg.embedding.model)[0]
        res = col.query(query_embeddings=[vec], n_results=k)
        hits: list[ChunkHit] = []
        for cid, doc, meta, dist in zip(
            res["ids"][0],
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0],
            strict=True,
        ):
            hits.append(
                ChunkHit(
                    chunk_id=cid,
                    text=doc,
                    url=str(meta.get("url", "")),
                    title=str(meta.get("title", "")),
                    heading_path=str(meta.get("heading_path", "")),
                    score=1.0 - float(dist),
                )
            )
        # O3: log chunk ids and scores so a bad answer can be traced to bad retrieval.
        for i, h in enumerate(hits):
            s.set_attribute(f"retrieval.documents.{i}.document.id", h.chunk_id)
            s.set_attribute(f"retrieval.documents.{i}.document.score", round(h.score, 4))
            s.set_attribute(f"retrieval.documents.{i}.document.metadata", h.url)
        return hits
