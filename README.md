# TrustDesk

A multi-agent assistant that drafts answers to vendor security questionnaires from a company's own
policy documents. Every answer cites the exact policy passage, flags questions the policies do not
cover, and waits for a human to approve it.

> Status: **phase 1 of 6 complete (RAG baseline); phase 2 (golden set, LLM judge, tracing) in
> progress.** Multi-agent LangGraph orchestration, guardrails and the review UI follow in later
> phases. Progress: [build log](docs/BUILD_LOG.md). Full plan:
> [requirements and design doc](https://claude.ai/artifact/EHk26QLd69ek2ciwzUrN5M).

## Quick start (Windows, macOS or Linux)

```bash
pip install uv                      # once
uv sync                             # create .venv and install
ollama pull qwen3:4b                # local model, ~2.5 GB (https://ollama.com)
uv run trustdesk fetch              # crawl GitLab handbook security pages into data/raw/
uv run trustdesk index              # chunk, embed (bge-small, CPU) and store in ChromaDB
uv run trustdesk ask "Is customer data encrypted at rest?"
uv run trustdesk batch              # run the 20-question phase 1 set
```

Measure your machine's model speed with `uv run python scripts/bench_ollama.py qwen3:4b`.

## How phase 1 works

1. **Fetch** — a polite crawler (robots.txt, 1 s delay) saves handbook pages as Markdown with URL,
   title, fetch time and content hash.
2. **Chunk** — heading-aware chunks of ~300 tokens with 50-token overlap; each keeps its heading path.
3. **Index** — `BAAI/bge-small-en-v1.5` embeddings via fastembed (ONNX, CPU) into persistent ChromaDB.
4. **Answer** — top-3 chunks go to the drafter (Qwen3 4B on Ollama) with a JSON schema; the reply is
   validated with Pydantic (one repair retry) and every citation is checked: its chunk must have been
   in context and its quote must appear in that chunk.

## Results

Phase 1 baseline on 20 questions ([full report](reports/phase1.md)):

| Metric | Result |
| --- | --- |
| Valid structured output | 20 / 20 |
| Out-of-scope questions correctly declined | 3 / 4 |
| Citations verified verbatim in the cited chunk | 17 / 22 (77%) |
| Median time per question (Qwen3 4B, laptop iGPU) | 33.5 s |

## Data and licences

- Policy text: [GitLab Handbook](https://handbook.gitlab.com), CC BY-SA 4.0. Pages are fetched at run
  time into `data/` (git-ignored) rather than committed.
- The CSA CAIQ questionnaire is licensed for personal, non-commercial use and may not be
  redistributed, so it is never committed. The phase 1 questions are written in our own words.

## Development

```bash
uv run python -m pytest
uv run ruff check . && uv run python -m mypy
uv run pre-commit install
```

On Windows with Smart App Control on, run everything as `python -m ...` (including
`uv run python -m trustdesk ask "..."`); see [ADR 0002](docs/adr/0002-windows-smart-app-control.md).
