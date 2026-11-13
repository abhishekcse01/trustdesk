"""Wire the graph to the real retriever, drafter, reranker, audit log and SQLite checkpoints."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver

from trustdesk.agents.answer import SYSTEM_PROMPT, DraftAnswer, build_user_prompt
from trustdesk.config import Settings, path
from trustdesk.graph.build import Deps, build_graph
from trustdesk.llm import structured
from trustdesk.retrieval.dense import ChunkHit
from trustdesk.retrieval.search import _cross_encoder, search
from trustdesk.store import audit


def real_deps(cfg: Settings) -> Deps:
    def retrieve(question: str, k: int) -> list[ChunkHit]:
        return search(question, cfg, k)

    def draft(question: str, hits: list[ChunkHit]) -> tuple[DraftAnswer | None, str | None]:
        result, error, _, _ = structured(
            cfg.llm.drafter,
            SYSTEM_PROMPT,
            build_user_prompt(question, hits),
            DraftAnswer,
            cfg.llm,
            name="draft",
        )
        return result, error

    def scorer(query: str, passages: Sequence[str]) -> list[float]:
        return [
            float(s)
            for s in _cross_encoder(cfg.retrieval.reranker_model).rerank(query, list(passages))
        ]

    def save(entry: dict[str, Any]) -> str:
        return audit.record(path(cfg.graph.audit_db), entry)

    return Deps(
        retrieve=retrieve,
        draft=draft,
        scorer=scorer,
        save=save,
        cfg=cfg.graph,
        model=cfg.llm.drafter,
        base_top_k=cfg.retrieval.top_k,
    )


def open_graph(cfg: Settings) -> tuple[Any, sqlite3.Connection]:
    """Compiled graph with a SQLite checkpointer; close the returned connection when done."""
    db = path(cfg.graph.checkpoint_db)
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db, check_same_thread=False)
    return build_graph(real_deps(cfg), SqliteSaver(conn)), conn
