# Build log

A running record of what was built, measured and decided. Newest first.
Visual version: [TrustDesk System Map](https://claude.ai/artifact/J9jzu6bcA2yayhfQ67JAge) (private).

## Backlog from community feedback

- **Unbiased retrieval benchmark** (LinkedIn comment on post 2, 11 Nov 2026). The grid tie (~98%
  hit@3 for every setup) only measures agreement with the dense retriever, because references
  were drafted from dense results. Fix:
  1. Pick 30 answerable/partial questions.
  2. Pool the top 10 pages from all four retrieval setups per question (TREC-style pooling), so
     no single retriever decides what counts as gold.
  3. Hand-mark every pooled page as answer-bearing or not.
  4. Score each setup on "any gold page in the top 3" (and MRR against gold pages).
  5. Keep unanswerable questions separate: for those, "no gold page exists" is correct.
  Cost: no API tokens; about 45–60 min of hand labelling. Report the before/after in the next
  LinkedIn update (promised in the reply).
  - [x] Sheet generated (11 Nov): `evals/gold_pages/label-gold-pages.csv`, 30 questions across
        domains, 225 pooled pages (top-5 pages per setup), blind: no "found by" column, shuffled.
        Built by `uv run python -m trustdesk gold sheet`.
  - [ ] Mark `is_gold` y/n for every row
  - [ ] Score: `uv run python -m trustdesk gold score`

## 2026-11-13 · Phase 4 (in progress): LangGraph workflow, Verifier, human review

- **Per-question LangGraph state machine** (`src/trustdesk/graph/build.py`): guard → retrieve →
  draft → verify → {review | retry with +2 chunks | abstain} → human_review → save. Fixed edges,
  every dependency injectable for tests ([ADR 0004](adr/0004-langgraph-verifier-and-human-review.md)).
- **Local Verifier** (`graph/verify.py`), no API tokens: evidence (question vs best passage,
  cross-encoder), citation ratio, support (answer vs cited passages). Low evidence forces
  "Needs human input"; weak support triggers one retry.
- **Threshold from data** (`trustdesk calibrate`, 97 references): evidence median 0.985 for
  answerable vs 0.036 for unanswerable. Chose **0.3**: abstains on 68% of unanswerable and 7.6%
  of answerable questions (0.5 gave 68% / 10.6%; 0.95 gave 94% / 29%). Scores saved to
  `reports/evals/calibration.json`.
- **Human review:** `interrupt()` pauses with draft, citations, scores and guard flags; resumes
  on approve / edit / reject; state checkpointed in SQLite so a pending review survives restarts.
- **Append-only audit log** (`store/audit.py`) with a SHA-256 per row; `trustdesk audit`.
- **CLI:** `trustdesk run "question"` (review in the terminal), `audit`, `calibrate`.
- End-to-end on 3 questions: encryption answered (confidence 1.0); laptop brand abstained
  (evidence 0.002); audit findings passed with evidence 0.88 although it answers with
  vulnerability tracking: the relevance model is only a proxy for entailment (known limit).
- Smart App Control blocked `uuid_utils` (needed by LangGraph): pure-Python uuid7 shim, applied
  from `trustdesk/__init__.py`; pytest runs with `-p no:langsmith_plugin` (ADR 0002 update).
- 34 tests pass (8 new for the graph); ruff and strict mypy clean.

**Still to do in phase 4:** questionnaire-level parent graph (fan-out with `Send`), XLSX import
and export, FastAPI service and Streamlit review screen, before/after eval of the Verifier, and a
stronger entailment check for related-but-wrong passages.

## 2026-11-11 · Phase 3 complete: hybrid search + reranking ([report](../reports/phase3.md))

- BM25 keyword index (`retrieval/bm25.py`), hybrid retrieval with reciprocal rank fusion and a
  cross-encoder reranker (`retrieval/search.py`, MiniLM via fastembed on CPU, ~1.8 s/question).
- `trustdesk grid` compares retrieval setups with no LLM calls
  ([results](../reports/evals/grid-retrieval.md)): all four keep ~98% hit@3, but the benchmark
  favours dense because references were drafted from dense results.
- A qualitative check on hard questions showed hybrid + rerank finding the right policy where
  dense missed it (SDLC standard for change approval; Encryption Policy for TLS).
- 10-question confirmation ([results](../reports/evals/confirm-10-hybrid-rerank.md)): of 9
  baseline failures, 3 now correct and 1 partly correct (selection bias stated).
- **Default is now `retrieval.mode: hybrid`, `retrieval.rerank: true`.**
- Deferred to phase 4: query rewriting, confidence calibration, fixing over-answering (Verifier).

## 2026-11-10 · Phase 2 measured ([report](../reports/phase2.md))

- Judge and reference model: Gemini 3.5 Flash-Lite on the free tier. `trustdesk models` lists
  usable names; calls throttled to 4.5 s, retried with back-off, capped at 400 output tokens,
  4 chunks per reference, judge chunks cut to 800 characters. About 170 Gemini calls used.
- 97 of 100 references drafted (3 overran the token cap and were skipped); none verified yet.
- Baseline on a balanced 50 across 17 domains ([results](../reports/evals/baseline-50-dense.md)):
  correctness 73%, faithfulness 77%, citation accuracy 70%, correct abstention 65%,
  false abstention 3%, median 24 s. Main failure: over-answering.
- A DNS failure crashed the run at question 48: rows are now flushed per question, connection
  errors retry, judge failures are recorded, and `trustdesk report` combines partial runs.

**Waiting on you**

- [ ] Review `evals/golden/references.yaml` and set `verified: true` on checked entries
- [ ] Fill `human_correctness` (correct / partially_correct / incorrect) in
      `evals/judge_validation/label-me-50.csv`, then run
      `uv run python -m trustdesk agreement evals/judge_validation/label-me-50.csv`
- [ ] Approve pushing phases 2 and 3 to GitHub

## 2026-11-09 · Phase 2 (started): golden set, evals, tracing

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
