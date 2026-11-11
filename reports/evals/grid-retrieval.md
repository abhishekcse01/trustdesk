# Retrieval grid 20261111-155546

66 answerable/partial golden questions; a hit means a reference page is in the top results. No LLM calls.

| Config | hit@1 | hit@3 | MRR@10 | ms / query |
| --- | --- | --- | --- | --- |
| dense | 85% | 98% | 0.92 | 29 |
| dense + rerank | 86% | 100% | 0.92 | 1790 |
| hybrid (RRF) | 83% | 98% | 0.90 | 55 |
| hybrid + rerank | 80% | 98% | 0.89 | 1787 |

Caveat: references were drafted from dense top-4 results, which favours dense.
