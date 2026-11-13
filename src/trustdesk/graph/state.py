"""Graph state for one questionnaire question."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

Status = Literal["drafting", "needs_review", "abstained", "approved", "rejected"]


class QuestionState(TypedDict, total=False):
    question_id: str
    question: str
    guard_flags: list[str]
    top_k: int
    hits: list[dict[str, Any]]  # ChunkHit dumps
    draft: dict[str, Any] | None  # DraftAnswer dump
    draft_error: str | None
    checks: list[dict[str, Any]]  # CitationCheck dumps
    verification: dict[str, Any]
    attempts: int
    status: Status
    review: dict[str, Any]  # {"decision", "text", "reviewer"} from the human
    final_answer: str
    trace_id: str
