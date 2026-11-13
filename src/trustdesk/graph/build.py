"""Per-question LangGraph state machine.

    START → guard → retrieve → draft → verify ─┬─ review ──────────────┐
                       ▲                        ├─ retry ─→ (retrieve)  │
                       └────────────────────────┘                       ▼
                                                └─ abstain → abstain → human_review → save → END

`human_review` calls `interrupt()`: the run pauses, its state is checkpointed (SQLite), and
it resumes only when a person sends `Command(resume={"decision": ..., ...})`. Nothing reaches
the audit log without that decision (governance requirement G1). Side effects live in `save`,
after the interrupt, because LangGraph re-runs an interrupted node from its start on resume.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from trustdesk.agents.answer import PROMPT_VERSION, CitationCheck, DraftAnswer, check_citations
from trustdesk.config import GraphCfg
from trustdesk.graph.state import QuestionState
from trustdesk.graph.verify import Scorer, decide, score_evidence, score_support
from trustdesk.retrieval.dense import ChunkHit

INJECTION = re.compile(
    r"ignore (all|any|previous|prior) (instructions|rules)|system prompt|you are now|"
    r"disregard (the|all) (above|rules)",
    re.I,
)


@dataclass
class Deps:
    """Everything the graph touches outside itself, injectable for tests."""

    retrieve: Callable[[str, int], list[ChunkHit]]
    draft: Callable[[str, list[ChunkHit]], tuple[DraftAnswer | None, str | None]]
    scorer: Scorer
    save: Callable[[dict[str, Any]], str]
    cfg: GraphCfg
    model: str = ""
    base_top_k: int = 3


def build_graph(deps: Deps, checkpointer: Any) -> Any:
    def guard(state: QuestionState) -> dict[str, Any]:
        flags = ["possible_prompt_injection"] if INJECTION.search(state["question"]) else []
        return {"guard_flags": flags, "attempts": 0, "top_k": deps.base_top_k, "status": "drafting"}

    def retrieve(state: QuestionState) -> dict[str, Any]:
        hits = deps.retrieve(state["question"], state["top_k"])
        return {"hits": [h.model_dump() for h in hits]}

    def draft(state: QuestionState) -> dict[str, Any]:
        hits = [ChunkHit(**h) for h in state["hits"]]
        result, error = deps.draft(state["question"], hits)
        checks = check_citations(result, hits) if result else []
        return {
            "draft": result.model_dump() if result else None,
            "draft_error": error,
            "checks": [c.model_dump() for c in checks],
        }

    def verify(state: QuestionState) -> dict[str, Any]:
        hits = [ChunkHit(**h) for h in state["hits"]]
        raw = state.get("draft")
        d = DraftAnswer(**raw) if raw else None
        checks = [CitationCheck(**c) for c in state.get("checks", [])]
        by_id = {h.chunk_id: h.text for h in hits}
        cited = [by_id[c.chunk_id] for c in (d.citations if d else []) if c.chunk_id in by_id]
        v = decide(
            abstained=bool(d and d.abstain),
            evidence=score_evidence(state["question"], [h.text for h in hits], deps.scorer),
            citations_ok=sum(c.in_context and c.quote_found for c in checks),
            citations_total=len(checks),
            support=score_support(d.answer_text if d and not d.abstain else "", cited, deps.scorer),
            attempts=state.get("attempts", 0),
            abstain_threshold=deps.cfg.abstain_threshold,
            support_threshold=deps.cfg.support_threshold,
            max_retries=deps.cfg.max_retries,
        )
        update: dict[str, Any] = {"verification": v.__dict__}
        if v.route == "retry":
            update |= {
                "attempts": state.get("attempts", 0) + 1,
                "top_k": state["top_k"] + deps.cfg.retry_extra_k,
            }
        return update

    def route_after_verify(state: QuestionState) -> str:
        return str(state["verification"]["route"])

    def abstain(state: QuestionState) -> dict[str, Any]:
        reason = state["verification"]["reason"]
        return {
            "draft": DraftAnswer(
                answer_choice=None,
                answer_text="",
                citations=[],
                abstain=True,
                abstain_reason=f"Needs human input: {reason}",
            ).model_dump(),
            "checks": [],
            "status": "abstained",
        }

    def human_review(state: QuestionState) -> dict[str, Any]:
        decision = interrupt(
            {
                "question_id": state.get("question_id", ""),
                "question": state["question"],
                "draft": state.get("draft"),
                "checks": state.get("checks", []),
                "verification": state.get("verification", {}),
                "guard_flags": state.get("guard_flags", []),
                "sources": [
                    {"chunk_id": h["chunk_id"], "url": h["url"], "section": h["heading_path"]}
                    for h in state.get("hits", [])
                ],
            }
        )
        if decision.get("decision") not in {"approve", "edit", "reject"}:
            raise ValueError("review decision must be approve, edit or reject")
        return {"review": decision}

    def save(state: QuestionState) -> dict[str, Any]:
        review = state["review"]
        d = state.get("draft") or {}
        if review["decision"] == "reject":
            text, status = "", "rejected"
        elif review["decision"] == "edit":
            text, status = str(review.get("text", "")), "approved"
        else:
            text = d.get("answer_text") or d.get("abstain_reason") or ""
            status = "approved"
        deps.save(
            {
                "question_id": state.get("question_id", ""),
                "question": state["question"],
                "status": status,
                "decision": review["decision"],
                "reviewer": str(review.get("reviewer", "unknown")),
                "answer_text": text,
                "citations": d.get("citations", []),
                "verification": state.get("verification", {}),
                "model": deps.model,
                "prompt_version": PROMPT_VERSION,
                "trace_id": state.get("trace_id", ""),
            }
        )
        return {"status": status, "final_answer": text}

    g = StateGraph(QuestionState)
    for name, fn in [
        ("guard", guard),
        ("retrieve", retrieve),
        ("draft", draft),
        ("verify", verify),
        ("abstain", abstain),
        ("human_review", human_review),
        ("save", save),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "guard")
    g.add_edge("guard", "retrieve")
    g.add_edge("retrieve", "draft")
    g.add_edge("draft", "verify")
    g.add_conditional_edges(
        "verify",
        route_after_verify,
        {"review": "human_review", "retry": "retrieve", "abstain": "abstain"},
    )
    g.add_edge("abstain", "human_review")
    g.add_edge("human_review", "save")
    g.add_edge("save", END)
    return g.compile(checkpointer=checkpointer)
