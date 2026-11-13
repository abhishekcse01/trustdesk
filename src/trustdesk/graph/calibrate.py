"""Calibrate the Verifier's abstain threshold on the golden set (no LLM, no API tokens).

For each reference, compute the evidence score (best question-passage cross-encoder score over
the retrieved chunks). Answerable/partial questions should score high, unanswerable ones low.
Pick the threshold that maximises balanced accuracy between the two groups.
Caveat: labels are unverified Gemini drafts, so re-run after human verification.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from itertools import pairwise
from typing import Any

from trustdesk.config import Settings, path
from trustdesk.evals.golden import load_references
from trustdesk.graph.runtime import real_deps
from trustdesk.graph.verify import score_evidence


def best_threshold(pos: list[float], neg: list[float]) -> tuple[float, float]:
    """Threshold t (abstain when score < t) with the highest balanced accuracy."""
    values = sorted(set(pos + neg))
    # Midpoints between neighbouring scores: the threshold sits in the gap, not on a sample.
    candidates = [(a + b) / 2 for a, b in pairwise(values)]
    best_t, best_acc = 0.5, 0.0
    for t in [0.0, *candidates, 1.0]:
        tpr = sum(s >= t for s in pos) / len(pos)  # answerable kept
        tnr = sum(s < t for s in neg) / len(neg)  # unanswerable abstained
        acc = (tpr + tnr) / 2
        if acc > best_acc:
            best_t, best_acc = t, acc
    return round(best_t, 3), round(best_acc, 3)


def calibrate(cfg: Settings, log: Callable[[str], None] = print) -> dict[str, Any]:
    deps = real_deps(cfg)
    pos: list[float] = []
    neg: list[float] = []
    for ref in sorted(load_references().values(), key=lambda r: r.id):
        hits = deps.retrieve(ref.question, cfg.retrieval.top_k)
        score = score_evidence(ref.question, [h.text for h in hits], deps.scorer)
        (neg if ref.label == "unanswerable" else pos).append(score)
    t, acc = best_threshold(pos, neg)
    current = cfg.graph.abstain_threshold
    out = path("reports/evals/calibration.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"answerable_or_partial": pos, "unanswerable": neg}, indent=1), encoding="utf-8"
    )

    def rates(th: float) -> tuple[float, float]:
        return (
            round(sum(s < th for s in neg) / len(neg), 3),
            round(sum(s < th for s in pos) / len(pos), 3),
        )

    result = {
        "answerable_or_partial": len(pos),
        "unanswerable": len(neg),
        "median_evidence_answerable": round(sorted(pos)[len(pos) // 2], 3),
        "median_evidence_unanswerable": round(sorted(neg)[len(neg) // 2], 3),
        "best_threshold": t,
        "balanced_accuracy": acc,
        "at_best": {"unanswerable_abstained": rates(t)[0], "answerable_abstained": rates(t)[1]},
        "at_current": {
            "threshold": current,
            "unanswerable_abstained": rates(current)[0],
            "answerable_abstained": rates(current)[1],
        },
        "curve": [
            {
                "threshold": th,
                "unanswerable_abstained": rates(th)[0],
                "answerable_abstained": rates(th)[1],
            }
            for th in (0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95)
        ],
    }
    log(
        f"  answerable median {result['median_evidence_answerable']} vs unanswerable median "
        f"{result['median_evidence_unanswerable']}"
    )
    return result
