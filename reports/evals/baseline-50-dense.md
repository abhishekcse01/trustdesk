# Eval run 20261111-155038

Drafter `ollama/qwen3:4b` · judge `gemini/gemini-3.5-flash-lite` · prompt `answer-v1` · 50 questions (0 with human-verified references)

| Metric | Value |
| --- | --- |
| Correctness (judge, 1 / 0.5 / 0) | 73% |
| Faithfulness (claims supported by context) | 77% |
| Citation accuracy (verbatim) | 70% |
| Retrieval hit@3 (expected page in top 3) | 100% |
| Retrieval MRR | 97% |
| Correct abstention (unanswerable) | 65% |
| False abstention (answerable/partial) | 3% |
| Valid structured output | 100% |
| Median seconds per question | 24.4 |
| Judge parse failures (counted as missing) | 0 |

| Label | Correctness |
| --- | --- |
| answerable | 83% |
| partial | 50% |
| unanswerable | 68% |

## Incorrect answers

| ID | Label | Judge rationale |
| --- | --- | --- |
| G003 | unanswerable | The candidate incorrectly asserted an answer to an unanswerable reference question about audit findings, confusing vulnerability tracking with audit finding tracking. |
| G011 | unanswerable | The candidate answered an unanswerable question by asserting details about vetting and vulnerability scanners, making it incorrect according to the reference label. |
| G019 | unanswerable | The candidate incorrectly asserted an answer while the reference label states the question is unanswerable based on the provided context. |
| G021 | unanswerable | The candidate asserts an answer when the reference states the information is unanswerable based on the policy passages. |
| G029 | unanswerable | The reference label is unanswerable as the provided texts do not state whether weak or deprecated algorithms are strictly prohibited, making the candidate's assertion incorrect. |
| G049 | partial | The candidate incorrectly abstained, whereas the reference notes that the provided passages do indicate it is an annual requirement. |
| G085 | answerable | The candidate incorrectly provides a specific schedule based on risk levels that applies to finding reviews rather than vendor assessments, contradicting the reference policy. |
| G092 | answerable | The candidate incorrectly asserts that scans are run on a schedule, whereas the reference states there is no configured schedule and they occur on a less than daily basis. |
| G096 | unanswerable | The candidate incorrectly asserts an answer to an unanswerable question and includes claims not supported by the provided context. |
