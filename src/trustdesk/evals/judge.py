"""LLM-as-judge (E3): correctness against the reference, faithfulness against retrieved context.

Faithfulness follows the RAGAS idea (split the answer into claims, check each against the
context) but is implemented here because RAGAS needs pandas, which Smart App Control blocks on
the development laptop. Judge output that fails to parse is recorded as missing, never as 0 (R3).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from trustdesk.agents.answer import AnswerResult
from trustdesk.config import Settings
from trustdesk.evals.golden import Reference
from trustdesk.llm import structured

JUDGE_VERSION = "judge-v1"

JUDGE_PROMPT = """You grade answers to security questionnaire questions. Be strict and consistent.

You get the QUESTION, a verified REFERENCE (label, answer, key facts), the CANDIDATE answer, and \
the CONTEXT passages the candidate was given.

correctness, judged against the REFERENCE:
- Reference label "unanswerable": "correct" if the candidate abstained or clearly said the \
information is not available; "incorrect" if it asserted an answer.
- Otherwise: "correct" if the candidate states the key facts with no contradiction; \
"partially_correct" if it states some key facts or is vaguer than the reference; "incorrect" if \
it contradicts the reference, misses all key facts, or wrongly abstains.

claims: split the candidate answer into atomic factual claims (max 8). For each, supported = true \
only if the CONTEXT states it. Leave claims empty if the candidate abstained.

Give a one-sentence rationale."""


class Claim(BaseModel):
    claim: str
    supported: bool


class Verdict(BaseModel):
    correctness: Literal["correct", "partially_correct", "incorrect"]
    claims: list[Claim] = Field(max_length=8)
    rationale: str


class JudgeResult(BaseModel):
    correctness: str
    faithfulness: float | None  # supported claims / claims; None when there are no claims
    claims: list[Claim]
    rationale: str
    model: str
    version: str = JUDGE_VERSION


def judge(
    ref: Reference, res: AnswerResult, cfg: Settings, model: str
) -> tuple[JudgeResult | None, str | None]:
    d = res.draft
    if d is None:
        candidate = "(no valid answer produced)"
    elif d.abstain:
        candidate = f"ABSTAINED: {d.abstain_reason}"
    else:
        candidate = f"{d.answer_choice or ''} {d.answer_text}".strip()
    facts = "\n".join(f"- {f}" for f in ref.key_facts) or "(none)"
    cut = cfg.eval.judge_chunk_chars
    context = "\n\n".join(f"[{h.chunk_id}] {h.text[:cut]}" for h in res.hits)
    user = (
        f"QUESTION: {ref.question}\n\nREFERENCE (label: {ref.label}):\n{ref.reference_answer}\n"
        f"Key facts:\n{facts}\n\nCANDIDATE:\n{candidate}\n\nCONTEXT:\n{context}"
    )
    verdict, error, _, _ = structured(model, JUDGE_PROMPT, user, Verdict, cfg.llm, name="judge")
    if verdict is None:
        return None, error
    supported = [c.supported for c in verdict.claims]
    return (
        JudgeResult(
            correctness=verdict.correctness,
            faithfulness=round(sum(supported) / len(supported), 3) if supported else None,
            claims=verdict.claims,
            rationale=verdict.rationale,
            model=model,
        ),
        None,
    )
