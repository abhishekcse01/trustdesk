# TrustDesk

A multi-agent assistant that drafts answers to vendor security questionnaires from a company's own
policy documents. Every answer cites the exact policy passage, flags questions the policies do not
cover, and waits for a human to approve it.

> Status: **phases 1 and 3 complete, phase 2 measured (references awaiting human review).**
> Next: the LangGraph multi-agent graph with a Verifier and human approval (phase 4), then
> guardrails and the review UI. Progress: [build log](docs/BUILD_LOG.md). Full plan:
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
uv run trustdesk golden build       # draft reference answers (needs an API key in .env)
uv run trustdesk eval --limit 10    # score answers: retrieval, citations, abstention, LLM judge
uv run trustdesk grid               # compare retrieval setups, no API calls
```

Put `GEMINI_API_KEY` (free tier works), `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in `.env`
(git-ignored); `uv run trustdesk models` lists the Gemini models your key can use.

Measure your machine's model speed with `uv run python scripts/bench_ollama.py qwen3:4b`.

## How it works today

1. **Fetch** — a polite crawler (robots.txt, 1 s delay) saves handbook pages as Markdown with URL,
   title, fetch time and content hash.
2. **Chunk** — heading-aware chunks of ~300 tokens with 50-token overlap; each keeps its heading path.
3. **Index** — `BAAI/bge-small-en-v1.5` embeddings via fastembed (ONNX, CPU) into persistent ChromaDB,
   plus a BM25 keyword index over the same chunks.
4. **Retrieve** — dense and BM25 results fused with reciprocal rank fusion, then reranked by a
   cross-encoder; the best 3 chunks go on.
5. **Answer** — the drafter (Qwen3 4B on Ollama) replies against a JSON schema; Pydantic validates it
   (one repair retry) and every citation is checked: its chunk must have been retrieved and its quote
   must appear in that chunk.
6. **Measure** — a golden set, an LLM judge and OpenTelemetry traces show where answers go wrong.

## Results

Baseline on a balanced 50 golden questions, dense retrieval ([phase 2 report](reports/phase2.md)):

| Metric | Result |
| --- | --- |
| Correctness (LLM judge) | 73% |
| Faithfulness (claims supported by retrieved text) | 77% |
| Citations verified word for word | 70% |
| Unanswerable questions correctly declined | 65% |
| Answerable questions wrongly declined | 3% |
| Median time per question (Qwen3 4B, laptop iGPU) | 24 s |

Phase 3 switched retrieval to hybrid + rerank after it found the right policy in cases dense search
missed ([phase 3 report](reports/phase3.md)). References are unverified Gemini Flash-Lite drafts, so
treat these numbers as provisional. Phase 1 results: [reports/phase1.md](reports/phase1.md).

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
