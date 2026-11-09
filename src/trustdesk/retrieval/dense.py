"""Dense retrieval over the Chroma index. BM25 + RRF + reranking arrive in phase 3."""

from __future__ import annotations

from pydantic import BaseModel

from trustdesk.config import Settings
from trustdesk.ingest.index import client, embed


class ChunkHit(BaseModel):
    chunk_id: str
    text: str
    url: str
    title: str
    heading_path: str
    score: float  # cosine similarity, higher is better


def search(query: str, cfg: Settings, k: int | None = None) -> list[ChunkHit]:
    col = client(cfg).get_collection(cfg.collection)
    vec = embed([query], cfg.embedding.model)[0]
    res = col.query(query_embeddings=[vec], n_results=k or cfg.retrieval.top_k)
    hits: list[ChunkHit] = []
    for cid, doc, meta, dist in zip(
        res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0], strict=True
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
    return hits
