"""Verifier scoring and routing (phase 4).

Three signals, all local and token-free:

- evidence:  how well the best retrieved passage matches the QUESTION (cross-encoder score,
             squashed to 0-1). Low evidence means the policies probably do not cover it, so
             the graph abstains instead of letting the model over-answer (phase 2's main error).
- citations: share of quotes found word for word in a retrieved chunk.
- support:   how well the answer text matches the passages it cites (cross-encoder, 0-1).

The cross-encoder is a relevance model (MS MARCO), used here as a proxy for entailment; a
dedicated NLI model is a later upgrade. The abstain threshold is calibrated on the golden set
(`trustdesk calibrate`).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

Scorer = Callable[[str, Sequence[str]], list[float]]  # (query, passages) -> raw scores
Route = Literal["review", "retry", "abstain"]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass(frozen=True)
class Verification:
    evidence: float
    citation_ratio: float
    support: float
    confidence: float
    route: Route
    reason: str


def score_evidence(question: str, passages: Sequence[str], scorer: Scorer) -> float:
    if not passages:
        return 0.0
    return round(max(sigmoid(s) for s in scorer(question, passages)), 4)


def score_support(answer_text: str, cited: Sequence[str], scorer: Scorer) -> float:
    if not answer_text.strip() or not cited:
        return 0.0
    scores = [sigmoid(s) for s in scorer(answer_text, cited)]
    return round(sum(scores) / len(scores), 4)


def decide(
    *,
    abstained: bool,
    evidence: float,
    citations_ok: int,
    citations_total: int,
    support: float,
    attempts: int,
    abstain_threshold: float,
    support_threshold: float,
    max_retries: int,
) -> Verification:
    """Pure routing logic, kept separate so it is easy to test and explain."""
    ratio = citations_ok / citations_total if citations_total else 0.0
    confidence = round(0.4 * evidence + 0.3 * ratio + 0.3 * support, 3)

    def v(route: Route, reason: str) -> Verification:
        return Verification(evidence, round(ratio, 3), support, confidence, route, reason)

    if abstained:
        return v("review", "drafter abstained")
    if evidence < abstain_threshold:
        return v(
            "abstain",
            f"evidence {evidence:.2f} below {abstain_threshold:.2f}: policies "
            "do not appear to address the question",
        )
    weak = citations_total == 0 or ratio < 1.0 or support < support_threshold
    if weak and attempts < max_retries:
        return v(
            "retry",
            f"weak support (citations {citations_ok}/{citations_total}, "
            f"support {support:.2f}): retry with wider context",
        )
    if citations_total == 0:
        return v("abstain", "no citations after retry")
    return v("review", "supported" if not weak else "weak but out of retries: flagged for review")
