"""Retrieval experiment grid (E5): compare retrieval configs on the golden references.

Retrieval-only, so it needs no LLM and no API tokens: for every reference with supporting
pages, check where those pages rank. Caveat: references were drafted from dense top-4 results,
so they can only cite pages dense search found. That biases the comparison toward dense; a
config that finds a different but valid page is scored as a miss.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from trustdesk.config import Settings, path
from trustdesk.evals.golden import load_references
from trustdesk.evals.metrics import hit_at_k, reciprocal_rank
from trustdesk.retrieval.search import retrieve

CONFIGS: dict[str, dict[str, Any]] = {
    "dense": {"mode": "dense", "rerank": False},
    "dense + rerank": {"mode": "dense", "rerank": True},
    "hybrid (RRF)": {"mode": "hybrid", "rerank": False},
    "hybrid + rerank": {"mode": "hybrid", "rerank": True},
}


def with_retrieval(cfg: Settings, **changes: Any) -> Settings:
    return cfg.model_copy(update={"retrieval": cfg.retrieval.model_copy(update=changes)})


def run_grid(
    cfg: Settings, names: list[str] | None = None, log: Callable[[str], None] = print
) -> tuple[list[dict[str, Any]], Path]:
    refs = [
        r for r in load_references().values() if r.supporting_urls and r.label != "unanswerable"
    ]
    results = []
    for name, changes in CONFIGS.items():
        if names and name not in names:
            continue
        variant = with_retrieval(cfg, **changes)
        retrieve("warm up models and indexes", variant, k=10)
        h1, h3, rr, secs = [], [], [], []
        for ref in refs:
            t0 = time.perf_counter()
            urls = [h.url for h in retrieve(ref.question, variant, k=10)]
            secs.append(time.perf_counter() - t0)
            h1.append(hit_at_k(urls, ref.supporting_urls, 1))
            h3.append(hit_at_k(urls, ref.supporting_urls, 3))
            rr.append(reciprocal_rank(urls, ref.supporting_urls))
        row = {
            "config": name,
            "questions": len(refs),
            "hit_at_1": round(sum(h1) / len(h1), 3),
            "hit_at_3": round(sum(h3) / len(h3), 3),
            "mrr_at_10": round(sum(rr) / len(rr), 3),
            "ms_per_query": round(1000 * sum(secs) / len(secs)),
        }
        results.append(row)
        log(
            f"  {name:<16} hit@1 {row['hit_at_1']:.0%}  hit@3 {row['hit_at_3']:.0%}  "
            f"MRR {row['mrr_at_10']:.2f}  {row['ms_per_query']} ms"
        )
    return results, write_grid_report(results)


def write_grid_report(results: list[dict[str, Any]]) -> Path:
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
    lines = [
        f"# Retrieval grid {stamp}",
        "",
        f"{results[0]['questions'] if results else 0} answerable/partial golden questions; "
        "a hit means a reference page is in the top results. No LLM calls.",
        "",
        "| Config | hit@1 | hit@3 | MRR@10 | ms / query |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {r['config']} | {r['hit_at_1']:.0%} | {r['hit_at_3']:.0%} | {r['mrr_at_10']:.2f} | "
        f"{r['ms_per_query']} |"
        for r in results
    ]
    lines += ["", "Caveat: references were drafted from dense top-4 results, which favours dense."]
    file = path("reports/evals") / f"grid-{stamp}.md"
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    file.with_suffix(".json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return file
