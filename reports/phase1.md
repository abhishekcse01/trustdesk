# Phase 1 results: RAG baseline

Run on 2026-11-09 · Qwen3 4B (Ollama, integrated Radeon via Vulkan) · bge-small-en-v1.5 · ChromaDB ·
150 GitLab handbook security pages → 1,674 chunks · top 3 chunks per question · prompt `answer-v1`.

## Headline numbers (20 questions)

| Metric | Result |
| --- | --- |
| Valid structured output | 20 / 20 (no repair retries needed) |
| Unanswerable questions correctly abstained | 3 / 4 (P18 answered "N/A" with an explanation instead of abstaining) |
| Citations whose quote was found verbatim in the cited chunk | 17 / 22 (77%) |
| Answered questions with every citation verified | 13 / 17 |
| Median time per question | 33.5 s (11.4 min for all 20) |

Answer *correctness* is not scored yet: that needs the labelled golden set and judge from phase 2.

## Per-question results

| ID | Expected | Result | Citations verified | Seconds | Top retrieved section |
| --- | --- | --- | --- | --- | --- |
| P01 | answerable | Yes | 1/1 | 14 | Encryption Policy > Encryption |
| P02 | answerable | Yes | 1/1 | 32 | Encryption Policy > Encryption |
| P03 | answerable | Yes | 0/1 | 22 | Penetration Testing Policy > Execution |
| P04 | answerable | Yes | 0/1 | 32 | Password Standards > Multi Factor Authentication |
| P05 | answerable | Partial | 1/1 | 32 | Access Review Procedure > Purpose |
| P06 | answerable | Yes | 2/2 | 43 | Security Awareness Training Program |
| P07 | answerable | Yes | 2/2 | 38 | Security Incident Response Guide |
| P08 | answerable | Yes | 2/2 | 40 | Infrastructure Vulnerability Management Procedure |
| P09 | answerable | Yes | 1/1 | 43 | Security and Technology Policies Management |
| P10 | answerable | N/A | 1/1 | 45 | Security Compliance Controls > GitLab Control Framework |
| P11 | partial | Yes | 1/1 | 35 | Technical and Organizational Security Measures |
| P12 | partial | Yes | 2/2 | 56 | Third Party Risk Management |
| P13 | partial | Partial | 1/1 | 27 | Records Retention & Disposal |
| P14 | partial | Yes | 1/1 | 31 | HackerOne Process |
| P15 | partial | Yes | 0/1 | 29 | Access Review Procedure > Lookback Review |
| P16 | partial | Yes | 0/1 | 36 | Security Logging Standards |
| P17 | unanswerable | Abstain | n/a | 26 | Records Retention & Disposal |
| P18 | unanswerable | N/A | 1/2 | 41 | Third Party Risk Management |
| P19 | unanswerable | Abstain | n/a | 24 | USRM Program > References |
| P20 | unanswerable | Abstain | n/a | 40 | Risk-based Control Testing |

## What we learned

1. **Corpus selection matters more than the model.** The first crawl spent 144 of 203 pages on IT
   how-to guides, and "Is customer data encrypted at rest?" retrieved laptop disk-encryption pages.
   Blocking those sections and fetching policy pages first fixed retrieval before any model change.
2. **The citation checker catches real problems.** 5 of 22 quotes were not verbatim. In P15 the right
   policy (Access Management, deprovisioning) was not retrieved at all, and the model cited text from
   memory. Hybrid search and reranking (phase 3) target exactly this.
3. **Small models abstain reasonably.** 3 of 4 out-of-scope questions were declined with a clear
   reason; P18 declined in substance but chose "N/A" instead of the abstain flag. The Verifier in
   phase 4 should treat that pattern as an abstention.
4. **Speed is acceptable for development.** About 34 s per question locally with short prompts
   (~600 tokens); generation at 5–8 tokens/s is the bottleneck. Bulk evals move to an API model.

## Next (phase 2)

- Label answer correctness for these 20 and grow to 100 questions with reference answers.
- Add RAGAS / DeepEval metrics, an API-model judge, and Phoenix tracing.
- Track the citation-verified rate as a first-class metric.

Policy text cited in runs comes from the [GitLab Handbook](https://handbook.gitlab.com), CC BY-SA 4.0.
