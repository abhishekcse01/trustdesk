"""Pure metric functions (E2-E4), kept free of I/O so they are easy to test."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from statistics import mean, median
from typing import Any

CORRECTNESS_SCORE = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}


def hit_at_k(retrieved: Sequence[str], expected: Iterable[str], k: int) -> float:
    exp = set(expected)
    return 1.0 if exp and any(u in exp for u in retrieved[:k]) else 0.0


def reciprocal_rank(retrieved: Sequence[str], expected: Iterable[str]) -> float:
    exp = set(expected)
    for i, u in enumerate(retrieved, start=1):
        if u in exp:
            return 1.0 / i
    return 0.0


def _avg(values: list[float]) -> float | None:
    return round(mean(values), 3) if values else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-question eval rows into headline metrics.

    Each row has: label, retrieved_urls, expected_urls, abstained, citations_verified,
    citations_total, seconds, error, and optionally judge {correctness, faithfulness}.
    """
    answerable = [r for r in rows if r["label"] != "unanswerable"]
    unanswerable = [r for r in rows if r["label"] == "unanswerable"]
    with_refs = [r for r in answerable if r["expected_urls"]]
    judged = [r for r in rows if r.get("judge")]
    cit_total = sum(r["citations_total"] for r in rows)
    out: dict[str, Any] = {
        "questions": len(rows),
        "valid_output_rate": _avg([0.0 if r["error"] else 1.0 for r in rows]),
        "retrieval_hit_at_3": _avg(
            [hit_at_k(r["retrieved_urls"], r["expected_urls"], 3) for r in with_refs]
        ),
        "retrieval_mrr": _avg(
            [reciprocal_rank(r["retrieved_urls"], r["expected_urls"]) for r in with_refs]
        ),
        "citation_accuracy": round(sum(r["citations_verified"] for r in rows) / cit_total, 3)
        if cit_total
        else None,
        "correct_abstention_rate": _avg([1.0 if r["abstained"] else 0.0 for r in unanswerable]),
        "false_abstention_rate": _avg([1.0 if r["abstained"] else 0.0 for r in answerable]),
        "median_seconds": round(median(r["seconds"] for r in rows), 1) if rows else None,
        "judged": len(judged),
        "judge_failures": sum(1 for r in rows if r.get("judge_error")),
    }
    if judged:
        out["correctness"] = _avg(
            [
                CORRECTNESS_SCORE[r["judge"]["correctness"]]
                for r in judged
                if r["judge"]["correctness"] in CORRECTNESS_SCORE
            ]
        )
        faith = [
            r["judge"]["faithfulness"] for r in judged if r["judge"]["faithfulness"] is not None
        ]
        out["faithfulness"] = _avg(faith)
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in judged:
        by_label[r["label"]].append(r)
    out["correctness_by_label"] = {
        label: _avg([CORRECTNESS_SCORE.get(r["judge"]["correctness"], 0.0) for r in rs])
        for label, rs in sorted(by_label.items())
    }
    return out


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Agreement between two raters beyond chance (E6)."""
    if not a or len(a) != len(b):
        raise ValueError("need two equal, non-empty label lists")
    labels = sorted(set(a) | set(b))
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    expected = sum((a.count(lbl) / n) * (b.count(lbl) / n) for lbl in labels)
    return 1.0 if expected == 1 else round((observed - expected) / (1 - expected), 3)
