# Eval run 20261111-155701

Drafter `ollama/qwen3:4b` · judge `gemini/gemini-3.5-flash-lite` · prompt `answer-v1` · 10 questions (0 with human-verified references)

| Metric | Value |
| --- | --- |
| Correctness (judge, 1 / 0.5 / 0) | 40% |
| Faithfulness (claims supported by context) | 63% |
| Citation accuracy (verbatim) | 58% |
| Retrieval hit@3 (expected page in top 3) | 100% |
| Retrieval MRR | 100% |
| Correct abstention (unanswerable) | 0% |
| False abstention (answerable/partial) | 0% |
| Valid structured output | 100% |
| Median seconds per question | 25.3 |
| Judge parse failures (counted as missing) | 0 |

| Label | Correctness |
| --- | --- |
| answerable | 50% |
| partial | 100% |
| unanswerable | 25% |

## Incorrect answers

| ID | Label | Judge rationale |
| --- | --- | --- |
| G003 | unanswerable | The candidate is incorrect because the reference specifies the question is unanswerable, even though the context discusses vulnerability findings rather than specifically audit findings. |
| G011 | unanswerable | The candidate incorrectly asserts an answer to a question that is unanswerable based on the reference (how dependencies are vetted, rather than how vulnerabilities in them are processed). |
| G021 | unanswerable | The candidate incorrectly answered yes and asserted specific emergency change procedures that are not supported by the reference or context, which only mention general change management without addressing emergency changes. |
| G029 | unanswerable | The candidate incorrectly asserted an answer with specific details not found in the reference or context, whereas the reference states the question is unanswerable. |
| G092 | answerable | The candidate incorrectly asserts that scans are run on a schedule, directly contradicting the reference and context which state there is no configured schedule. |
