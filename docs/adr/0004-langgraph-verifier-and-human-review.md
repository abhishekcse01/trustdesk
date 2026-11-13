# ADR 0004: LangGraph state machine, local Verifier and human review

Date: 2026-11-13 · Status: accepted

## Context

Phase 2 showed over-answering as the main failure: 7 of 9 wrong answers were the model answering
questions the policies do not cover (correct abstention 65%). Better retrieval in phase 3 did not
fix it. Requirements G1/G2 also demand that no answer is final without a human decision and that
every decision is auditable. The Gemini budget rules out an LLM verifier call per answer.

## Decision

- **One LangGraph run per question** with deterministic edges:
  guard → retrieve → draft → verify → {review | retry → retrieve | abstain} → human_review → save.
  No LLM decides the next step (small models route badly; fixed edges are testable).
- **Local Verifier, three signals, no API tokens:**
  - *evidence*: best cross-encoder score between the question and retrieved passages (0–1);
    below the abstain threshold the graph replaces any draft with "Needs human input";
  - *citations*: share of quotes found word for word in retrieved chunks;
  - *support*: cross-encoder score between the answer text and its cited passages.
  Weak citations or support trigger one retry with 2 more chunks of context.
  The cross-encoder is an MS MARCO relevance model used as an entailment proxy; a true NLI model
  is a later upgrade.
- **Threshold from data:** `trustdesk calibrate` scores evidence for all golden references and
  reports the trade-off between abstaining on unanswerable and answerable questions.
- **Human review via `interrupt()`** with a SQLite checkpointer: the run pauses with the draft,
  citations, verifier scores and guard flags, and resumes on `approve | edit | reject`.
- **Append-only audit log** in SQLite with a SHA-256 of each row's payload. The save step runs
  after the interrupt because LangGraph re-runs an interrupted node from its start.
- **Guard (light):** flags instruction-like text in the question; full guardrails are phase 5.

## Consequences

- Over-answering is now caught by a measurable, tunable gate instead of prompt wording.
- Every component is injectable (`Deps`), so the graph is tested with fakes: interrupt/resume,
  forced abstain, retry, edit, reject and audit hashing all run in milliseconds.
- Costs ~2 extra cross-encoder passes per question (about 1–2 s on CPU).
- The threshold is calibrated on unverified Gemini labels; re-run after human verification.
