# Phase 2 results: golden set, LLM judge, tracing

Run on 10–11 Nov 2026 · drafter Qwen3 4B (local, Ollama) · judge and reference drafter
Gemini 3.5 Flash-Lite (free tier) · prompt `answer-v1` · judge `judge-v1` · dense top-3 retrieval.

## What was built

- **Golden set:** 100 questions in our own words across 17 control domains.
- **References:** 97 drafted by Gemini from the top 4 retrieved chunks (3 skipped: their replies
  overran the 400-token cap). Drafted labels: 56 answerable, 10 partial, 31 unanswerable.
  **None are human-verified yet.**
- **Eval harness:** retrieval hit@3 and MRR, verbatim citation accuracy, abstention rates, and an
  LLM judge for correctness (vs reference) and faithfulness (claims supported by context).
- **Tracing:** every answer is an OpenTelemetry trace (CHAIN → RETRIEVER + LLM spans).
- **Token budget:** about 1.5k input and ≤400 output tokens per Gemini call, 4.5 s apart.

## Baseline on a balanced 50 (all 17 domains)

| Metric | Result |
| --- | --- |
| Correctness (judge: 1 / 0.5 / 0) | **73%** |
| Faithfulness (claims supported by retrieved context) | **77%** |
| Citation accuracy (quote found word for word) | 70% |
| Correct abstention on unanswerable questions | 65% (13 of 20) |
| False abstention on answerable/partial questions | 3% (1 of 30) |
| Valid structured output | 100% |
| Judge parse failures | 0 |
| Retrieval hit@3 / MRR | 100% / 0.97 (inflated, see below) |
| Median seconds per question | 24.4 |

| Label | Questions | Correctness |
| --- | --- | --- |
| answerable | 24 | 83% |
| partial | 6 | 50% |
| unanswerable | 20 | 68% |

## What the numbers mean

1. **Over-answering is the main failure.** 7 of 9 incorrect answers are the model answering a
   question the reference marked unanswerable. The model abstains rarely (false abstention 3%) but
   too often answers from nearby text, such as treating vulnerability tracking as audit tracking
   (G003).
2. **Some references are probably wrong.** Flash-Lite saw only 4 chunks per question. Several
   "unanswerable" references look answerable from the policies (G019 change approval, G029 weak
   cryptography). Human verification of `evals/golden/references.yaml` will move these numbers.
3. **Retrieval hit@3 is inflated by construction.** References cite only pages that dense
   retrieval already returned, so dense retrieval cannot miss them. Phase 3 compares retrievers
   with that bias stated.
4. **Citation accuracy dropped from 77% (phase 1, 20 questions) to 70%** on harder, broader
   questions: the citation checker remains essential.
5. **Free-tier engineering:** one network drop (DNS failure at question 48) crashed the first run.
   Rows are now flushed per question, connection errors retry with back-off, a judge failure is
   recorded instead of stopping the run, and `trustdesk report` combines partial runs.

## Still to do (needs a human)

- Verify the 97 references (`verified: true`), then re-score with `--verified-only`.
- Label `human_correctness` in the exported sheet in `evals/judge_validation/`, then run
  `trustdesk agreement <sheet>` for agreement and Cohen's kappa.

Policy text: [GitLab Handbook](https://handbook.gitlab.com), CC BY-SA 4.0.
