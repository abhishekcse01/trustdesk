"""Unbiased retrieval benchmark with hand-marked gold pages (TREC-style pooling).

The golden references were drafted from dense-retrieval results, so scoring retrievers against
them only measures agreement with dense search. Here, for a sample of questions, the top pages
from every retrieval setup are pooled and a human marks which pages actually answer the
question. Every setup is then scored against those gold pages. No LLM calls.
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from trustdesk.config import Settings, path
from trustdesk.evals.golden import load_references
from trustdesk.evals.grid import CONFIGS, with_retrieval
from trustdesk.evals.metrics import hit_at_k, reciprocal_rank
from trustdesk.retrieval.search import retrieve

DIR = "evals/gold_pages"
SHEET = "label-gold-pages.csv"
RANKINGS = "rankings.json"
# Blind labelling: the sheet never says which setup found a page, and pages are shuffled within
# each question, so the labeller cannot favour a retriever. Rankings stay in rankings.json.
FIELDS = ["qid", "question", "page_title", "section", "snippet", "page_url", "is_gold"]


def pick_questions(n: int, seed: int = 7) -> list[str]:
    """Spread n answerable/partial questions across domains, deterministic for a seed."""
    refs = [r for r in load_references().values() if r.label != "unanswerable"]
    by_domain: dict[str, list[str]] = defaultdict(list)
    for r in sorted(refs, key=lambda r: r.id):
        by_domain[r.domain].append(r.id)
    rng = random.Random(seed)
    for ids in by_domain.values():
        rng.shuffle(ids)
    picked: list[str] = []
    while len(picked) < n and any(by_domain.values()):
        for domain in sorted(by_domain):
            if by_domain[domain] and len(picked) < n:
                picked.append(by_domain[domain].pop())
    return sorted(picked)


def pages_in_order(hits: list[Any]) -> list[str]:
    """Collapse chunk hits to unique page URLs, keeping first-seen order."""
    return list(dict.fromkeys(h.url for h in hits))


def make_sheet(
    cfg: Settings, n: int = 30, depth: int = 5, log: Callable[[str], None] = print
) -> Path:
    refs = load_references()
    out_dir = path(DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    rankings: dict[str, dict[str, list[str]]] = {}
    rows: list[dict[str, str]] = []
    for qid in pick_questions(n):
        ref = refs[qid]
        found: dict[str, list[str]] = defaultdict(list)
        best: dict[str, Any] = {}
        rankings[qid] = {}
        for name, changes in CONFIGS.items():
            hits = retrieve(ref.question, with_retrieval(cfg, **changes), k=10)
            rankings[qid][name] = pages_in_order(hits)
            for h in hits:
                if h.url in rankings[qid][name][:depth]:
                    best.setdefault(h.url, h)
            for url in rankings[qid][name][:depth]:
                found[url].append(name)
        pool = list(found)
        random.Random(f"{qid}-shuffle").shuffle(pool)
        for url in pool:
            h = best[url]
            rows.append(
                {
                    "qid": qid,
                    "question": ref.question,
                    "page_title": h.title,
                    "section": h.heading_path,
                    "snippet": " ".join(h.text.split())[:280],
                    "page_url": url,
                    "is_gold": "",
                }
            )
        log(f"  {qid}: {len(found)} pooled pages")
    sheet = out_dir / SHEET
    with sheet.open("w", newline="", encoding="utf-8-sig") as f:  # BOM so Excel reads UTF-8
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    (out_dir / RANKINGS).write_text(json.dumps(rankings, indent=1), encoding="utf-8")
    return sheet


def score_sheet(sheet: Path | None = None) -> list[dict[str, Any]]:
    """Score each setup against the hand-marked gold pages (is_gold = y / yes / 1)."""
    sheet = sheet or path(DIR) / SHEET
    rankings = json.loads((sheet.parent / RANKINGS).read_text(encoding="utf-8"))
    gold: dict[str, set[str]] = defaultdict(set)
    labelled: set[str] = set()
    with sheet.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            mark = row["is_gold"].strip().lower()
            if mark:
                labelled.add(row["qid"])
            if mark in {"y", "yes", "1", "true"}:
                gold[row["qid"]].add(row["page_url"])
    scored = [q for q in rankings if q in labelled and gold[q]]
    results = []
    for name in CONFIGS:
        h1 = [hit_at_k(rankings[q][name], gold[q], 1) for q in scored]
        h3 = [hit_at_k(rankings[q][name], gold[q], 3) for q in scored]
        rr = [reciprocal_rank(rankings[q][name], gold[q]) for q in scored]
        results.append(
            {
                "config": name,
                "questions": len(scored),
                "hit_at_1": round(sum(h1) / len(h1), 3) if scored else None,
                "hit_at_3": round(sum(h3) / len(h3), 3) if scored else None,
                "mrr_at_10": round(sum(rr) / len(rr), 3) if scored else None,
            }
        )
    results.append(
        {
            "config": "_coverage",
            "labelled": len(labelled),
            "with_gold": len(scored),
            "no_gold_found": sorted(q for q in labelled if not gold[q]),
        }
    )
    return results
