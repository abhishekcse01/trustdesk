# Build log

A running record of what was built, measured and decided. Newest first.

## 2026-11-09 · Phase 2 (in progress): golden set, evals, tracing

**Done**

- `evals/golden/questions.yaml`: 100 questions written for this project across the 17 control
  domains used by the CSA Cloud Controls Matrix (prior labels: 34 answerable, 52 partial,
  14 unanswerable). No CAIQ text, so it is safe to publish.
- `trustdesk golden build`: an API model drafts a reference answer, key facts, supporting quotes
  and a label per question into `evals/golden/references.yaml`. Entries marked `verified: true`
  by a human are never overwritten.
- `trustdesk eval [--mode local|api] [--limit N] [--ids ...] [--no-judge] [--verified-only]`:
  - retrieval hit@3 and MRR against the reference pages
  - verbatim citation accuracy
  - correct and false abstention rates
  - LLM judge: correctness against the reference (1 / 0.5 / 0) and faithfulness (claims supported
    by the retrieved context, the RAGAS idea implemented in-house)
  - writes `reports/evals/eval-<time>.md` and `.json`, raw rows to `reports/runs/` (git-ignored)
- `trustdesk agreement <sheet.csv>`: judge-vs-human agreement and Cohen's kappa (E6). Each eval
  run exports a 50-row labelling sheet to `evals/judge_validation/`.
- OpenTelemetry tracing with OpenInference attribute names: each answer is a CHAIN span with
  RETRIEVER and LLM children, written to `reports/traces/` (git-ignored); set
  `OTEL_EXPORTER_OTLP_ENDPOINT` to export to Phoenix Cloud or Langfuse. Verified on a live
  question (3 linked spans).
- Shared `llm.structured()` helper: JSON-schema output, Pydantic validation, one repair retry,
  `$defs` inlined so every provider accepts the schema.
- 20 unit tests pass; ruff and strict mypy clean.
- [ADR 0003](adr/0003-tracing-and-eval-without-pandas.md): Phoenix, RAGAS and DeepEval all need
  pandas, which Smart App Control blocks, so tracing stays vendor-neutral and metrics are in-house.

**Waiting on**

- [ ] API key in `.env` (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`; with OpenAI also set model
      names under `eval:` in `config/settings.yaml`).

**Next steps once the key is in**

1. `uv run python -m trustdesk golden build` (drafts 100 references)
2. Review `evals/golden/references.yaml`, fix mistakes, set `verified: true`
3. `uv run python -m trustdesk golden stats`
4. `uv run python -m trustdesk eval --mode local` (about 1 hour) and `--mode api` for comparison
5. Fill `human_correctness` in the exported sheet, then `uv run python -m trustdesk agreement <sheet>`
6. Write `reports/phase2.md`, update the README results table, push, post the LinkedIn update

## 2026-11-09 · Phase 1 complete: RAG baseline

- Crawler, heading-aware chunker, bge-small embeddings, ChromaDB, cited-answer CLI.
- First crawl was skewed: 144 of 203 pages were IT how-to guides. Added deny lists and
  policy-first ordering; the re-crawl gave 150 pages / 1,674 chunks with the right policies.
- Qwen3 4B on the laptop: 30.7 tok/s prompt and 5.3 tok/s generation on CPU; 53.9 and 5.0 on the
  integrated Radeon via Vulkan (`OLLAMA_IGPU_ENABLE=1`). See [ADR 0001](adr/0001-local-model-and-native-ollama-client.md).
- Windows Smart App Control blocks venv launchers, grpc and compiled mypy; workarounds in
  [ADR 0002](adr/0002-windows-smart-app-control.md).
- Results on 20 questions ([report](../reports/phase1.md)): 20/20 valid output, 17/22 citations
  verified, 3/4 out-of-scope questions declined, 33.5 s median.
- Published at https://github.com/abhishekcse01/trustdesk.
