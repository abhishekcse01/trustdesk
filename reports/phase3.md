# Phase 3 results: hybrid search and reranking

Run on 11 Nov 2026 · same corpus (150 pages, 1,674 chunks) · Qwen3 4B drafter · Gemini 3.5
Flash-Lite judge · decision: **hybrid (dense + BM25, RRF) + cross-encoder rerank is now the default.**

## What was built

- **BM25 keyword index** over the same chunks as ChromaDB (`rank_bm25`), with a tokenizer that
  keeps technical terms such as `aes-256` and `1.2` together; cached on disk.
- **Hybrid retrieval:** dense top-20 and BM25 top-20 fused with reciprocal rank fusion (k = 60).
- **Reranker:** `Xenova/ms-marco-MiniLM-L-6-v2` cross-encoder (ONNX via fastembed, CPU, ~90 MB)
  reorders the candidate pool; the best 3 reach the drafter.
- **Experiment grid:** `trustdesk grid` scores each retrieval setup on the golden references
  (hit@1, hit@3, MRR@10, latency) with no LLM calls.
- Every setting is in `config/settings.yaml` (`retrieval.mode`, `retrieval.rerank`, ...).

## Grid on 66 answerable/partial golden questions (no API tokens)

| Config | hit@1 | hit@3 | MRR@10 | ms / query |
| --- | --- | --- | --- | --- |
| dense | 85% | 98% | 0.92 | 29 |
| dense + rerank | 86% | 100% | 0.92 | 1,790 |
| hybrid (RRF) | 83% | 98% | 0.90 | 55 |
| hybrid + rerank | 80% | 98% | 0.89 | 1,787 |

**This table is biased toward dense.** The references were drafted from dense top-4 results, so
they can only cite pages dense search already found. A setup that finds a different but correct
page scores a miss. The table shows that no setup loses the reference pages; it cannot show that
one finds better pages.

## What each setup actually retrieves

Top 3 pages on questions where dense search was known to struggle:

| Question | dense | hybrid + rerank |
| --- | --- | --- |
| G019 change approval before deployment | Third Party Risk, Vuln Code Review, Patch Release | **Software Development Lifecycle Standard**, Vulnerability triage, AppSec Review |
| G026 allowed TLS versions | Cryptography Standard, Google Cloud practices, AppSec Review | Cryptography Standard ×2, **Encryption Policy** |
| G029 weak cryptography prohibited | Cryptography Standard ×2, Security Architecture | Cryptography Standard ×3 |
| G058 access removal on leaving | Access Review, Access Management, Access Review | Access Review ×2, Access Management |

Hybrid alone sometimes adds noise (password guidelines for G058); the reranker removes it.

## Confirmation eval: the 9 questions wrong at baseline (+ G058)

| Setup | Correct | Partly correct | Correctness | Citation accuracy |
| --- | --- | --- | --- | --- |
| dense (phase 2 baseline, same 9) | 0 | 0 | 0% | 54% (7/13) |
| hybrid + rerank (same 9) | 3 | 1 | 39% | 58% (all 10) |

Fixed: G019 (now cites the SDLC standard), G049, G085; G096 partly. Caveats: these questions
were chosen *because* they failed, which exaggerates any gain (regression to the mean), and the
judge and references are unverified Flash-Lite outputs. A fair before/after needs the verified
golden set.

## Decision

Make **hybrid + rerank** the default. It retrieves the right policy in cases dense search misses,
costs ~1.8 s per question (next to ~24 s of generation), and keeps every reference page in the top
3 on the grid.

## What phase 3 did not fix

- **Over-answering:** the model still answers questions the references mark unanswerable (G003,
  G011, G021). That is a generation problem; phase 4 adds a Verifier (NLI check per citation) and
  confidence-based abstention.
- **Deferred to phase 4:** query rewriting (too slow on the local model for its expected gain) and
  confidence calibration (needs the Verifier's scores).
- **Next measurement step:** verify references, then re-run the grid and a fair 50-question eval.
- **Unbiased retrieval benchmark (from reader feedback):** hand-mark answer-bearing pages for 30
  questions, using pages pooled from all four setups so no retriever defines the gold set, then
  score "any gold page in the top 3". This separates retrieval quality from agreement with the
  dense-built references. Planned; see the backlog in `docs/BUILD_LOG.md`.
