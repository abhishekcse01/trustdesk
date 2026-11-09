"""Embed chunks and persist them to ChromaDB (D2 steps 7-8).

Phase 1 is a full rebuild; incremental re-ingest by content hash arrives in phase 3.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from trustdesk._compat import ensure_chromadb_importable

ensure_chromadb_importable()

import chromadb

from trustdesk.config import Settings, path
from trustdesk.ingest.chunk import Chunk, chunk_page, read_page


@lru_cache(maxsize=2)
def embedder(model: str) -> Any:
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=model)


def embed(texts: Sequence[str], model: str, batch_size: int = 64) -> list[list[float]]:
    return [list(map(float, v)) for v in embedder(model).embed(list(texts), batch_size=batch_size)]


def client(cfg: Settings) -> Any:
    return chromadb.PersistentClient(path=str(path(cfg.chroma_dir)))


def load_chunks(raw_dir: Path, cfg: Settings) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in sorted(raw_dir.glob("*.md")):
        meta, body = read_page(page)
        c = cfg.chunking
        chunks.extend(chunk_page(meta, body, c.target_tokens, c.overlap_tokens, c.min_tokens))
    return chunks


def build_index(cfg: Settings, log: Callable[[str], None] = print) -> int:
    raw_dir = path(cfg.raw_dir)
    chunks = load_chunks(raw_dir, cfg)
    if not chunks:
        raise SystemExit(f"No pages in {raw_dir}. Run `trustdesk fetch` first.")
    # De-duplicate identical chunk ids (same page reached by two URLs).
    unique = list({c.id: c for c in chunks}.values())
    log(f"{len(unique)} chunks from {len(list(raw_dir.glob('*.md')))} pages; embedding...")

    db = client(cfg)
    with contextlib.suppress(Exception):  # collection may not exist yet
        db.delete_collection(cfg.collection)
    col = db.create_collection(cfg.collection, metadata={"hnsw:space": "cosine"})

    step = 256
    for i in range(0, len(unique), step):
        batch = unique[i : i + step]
        vectors = embed([c.text for c in batch], cfg.embedding.model, cfg.embedding.batch_size)
        col.add(
            ids=[c.id for c in batch],
            documents=[c.text for c in batch],
            metadatas=[c.as_metadata() for c in batch],
            embeddings=vectors,
        )
        log(f"  indexed {min(i + step, len(unique))}/{len(unique)}")
    return len(unique)
