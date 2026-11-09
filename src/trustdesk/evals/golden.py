"""Golden set (E1): questions plus reference answers drafted by a strong model, verified by a human.

references.yaml is the source of truth. `trustdesk golden build` only fills entries that are
missing or not yet verified, so human edits (verified: true) are never overwritten.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from trustdesk.config import Settings, api_model, path
from trustdesk.llm import structured
from trustdesk.retrieval.dense import search

Label = Literal["answerable", "partial", "unanswerable"]

QUESTIONS = "evals/golden/questions.yaml"
REFERENCES = "evals/golden/references.yaml"

REFERENCE_PROMPT = """You are a security compliance expert building ground truth for an \
evaluation. Using ONLY the policy passages in CONTEXT (from the company's public handbook), \
decide how well they answer the QUESTION and write the reference answer a careful analyst would \
give.

- label: "answerable" if the passages fully answer it, "partial" if they answer part of it or \
only imply the answer, "unanswerable" if they do not address it.
- reference_answer: 1-4 sentences, first person plural ("We ..."), only facts from CONTEXT. For \
unanswerable questions, say what is not covered.
- key_facts: the 1-5 atomic facts a correct answer must contain (empty if unanswerable).
- supporting: chunk_id and an exact quote (max 30 words, copied word for word) for each key fact.
Never use outside knowledge about the company."""


class GoldenQuestion(BaseModel):
    id: str
    domain: str
    expect: Label
    question: str


class Support(BaseModel):
    chunk_id: str
    quote: str


class ReferenceDraft(BaseModel):
    label: Label
    reference_answer: str
    key_facts: list[str] = Field(max_length=5)
    supporting: list[Support] = Field(max_length=6)


class Reference(BaseModel):
    id: str
    question: str
    domain: str
    label: Label
    reference_answer: str
    key_facts: list[str] = []
    supporting_urls: list[str] = []
    supporting_quotes: list[str] = []
    drafted_by: str = ""
    verified: bool = False
    notes: str = ""


def load_questions(root: Path | None = None) -> list[GoldenQuestion]:
    data = yaml.safe_load((root or path(".")).joinpath(QUESTIONS).read_text(encoding="utf-8"))
    return [GoldenQuestion.model_validate(q) for q in data["questions"]]


def load_references() -> dict[str, Reference]:
    file = path(REFERENCES)
    if not file.exists():
        return {}
    data = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    return {r["id"]: Reference.model_validate(r) for r in data.get("references", [])}


def save_references(refs: dict[str, Reference]) -> None:
    header = (
        "# Reference answers for the golden set. Drafted by an API model from the GitLab Handbook\n"
        "# (CC BY-SA 4.0; quotes are short excerpts with attribution), then checked by a human.\n"
        "# Set verified: true after checking an entry; `golden build` never overwrites those.\n"
    )
    ordered = [refs[k].model_dump() for k in sorted(refs)]
    body = yaml.safe_dump({"references": ordered}, sort_keys=False, allow_unicode=True, width=100)
    path(REFERENCES).write_text(header + body, encoding="utf-8")


def draft_reference(q: GoldenQuestion, cfg: Settings, model: str) -> Reference | None:
    hits = search(q.question, cfg, k=cfg.eval.reference_top_k)
    by_id = {h.chunk_id: h for h in hits}
    context = "\n\n".join(
        f'<chunk id="{h.chunk_id}" source="{h.heading_path}" url="{h.url}">\n{h.text}\n</chunk>'
        for h in hits
    )
    user = f"QUESTION: {q.question}\n\nCONTEXT:\n{context}"
    draft, _error, _, _ = structured(
        model, REFERENCE_PROMPT, user, ReferenceDraft, cfg.llm, "reference"
    )
    if draft is None:
        return None
    urls = list(
        dict.fromkeys(by_id[s.chunk_id].url for s in draft.supporting if s.chunk_id in by_id)
    )
    return Reference(
        id=q.id,
        question=q.question,
        domain=q.domain,
        label=draft.label,
        reference_answer=draft.reference_answer,
        key_facts=draft.key_facts,
        supporting_urls=urls,
        supporting_quotes=[s.quote for s in draft.supporting],
        drafted_by=model,
    )


def build_references(
    cfg: Settings, ids: set[str] | None = None, log: Callable[[str], None] = print
) -> dict[str, Reference]:
    model = api_model("reference")
    refs = load_references()
    for q in load_questions():
        if ids and q.id not in ids:
            continue
        if q.id in refs and refs[q.id].verified:
            continue
        ref = draft_reference(q, cfg, model)
        if ref is None:
            log(f"  {q.id} failed to draft")
            continue
        refs[q.id] = ref
        save_references(refs)  # save as we go so a crash loses nothing
        log(f"  {q.id} {ref.label:<12} (expected {q.expect})")
    return refs
