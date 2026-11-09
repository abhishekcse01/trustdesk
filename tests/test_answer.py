import pytest
from pydantic import ValidationError

from trustdesk.agents.answer import (
    Citation,
    DraftAnswer,
    build_user_prompt,
    check_citations,
    parse_draft,
)
from trustdesk.retrieval.dense import ChunkHit

HIT = ChunkHit(
    chunk_id="c1",
    text="All customer data is **encrypted at rest** using AES-256 in production databases.",
    url="https://example.com",
    title="Encryption Policy",
    heading_path="Encryption Policy > Data at rest",
    score=0.8,
)


def draft(*cites: Citation) -> DraftAnswer:
    return DraftAnswer(
        answer_choice="Yes", answer_text="We encrypt.", citations=list(cites), abstain=False
    )


def test_citation_passes_when_quote_in_chunk_ignoring_markdown() -> None:
    [chk] = check_citations(
        draft(Citation(chunk_id="c1", quote="encrypted at rest using AES-256")), [HIT]
    )
    assert chk.in_context and chk.quote_found


def test_citation_fails_for_unknown_chunk_or_invented_quote() -> None:
    checks = check_citations(
        draft(
            Citation(chunk_id="zz", quote="encrypted at rest"),
            Citation(chunk_id="c1", quote="encrypted with RSA-4096"),
        ),
        [HIT],
    )
    assert not checks[0].in_context
    assert checks[1].in_context and not checks[1].quote_found


def test_parse_draft_accepts_fenced_json() -> None:
    raw = '```json\n{"answer_choice": null, "answer_text": "", "citations": [], "abstain": true, "abstain_reason": "not covered"}\n```'
    d = parse_draft(raw)
    assert d.abstain and d.abstain_reason == "not covered"


def test_parse_draft_rejects_bad_choice() -> None:
    with pytest.raises(ValidationError):
        parse_draft(
            '{"answer_choice": "Maybe", "answer_text": "x", "citations": [], "abstain": false}'
        )


def test_prompt_marks_context_as_data() -> None:
    prompt = build_user_prompt("Is data encrypted?", [HIT])
    assert prompt.startswith("QUESTION: Is data encrypted?")
    assert '<chunk id="c1"' in prompt
