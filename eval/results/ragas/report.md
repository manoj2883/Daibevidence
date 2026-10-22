# Ragas scores: dev set (43 questions)

Judge `claude-haiku-5-5` (effort medium), embeddings `sentence-transformers/all-MiniLM-L6-v2`, ragas 0.4.3. Built by `scripts/run_ragas_eval.py`; inputs and rules are in its docstring. Held-out set not touched.

Faithfulness and answer relevancy: answered rows only (any scope). Context precision and recall: in-scope rows with a reference, any status; `not_covered` rows are scored on the parents retrieved but not used, `answered_partial` rows on the judge-filtered parents the generator saw.

| Status | faithfulness | answer_relevancy | context_precision | context_recall |
|---|---|---|---|---|
| answered | 0.739 (n=9) | 0.908 (n=9) | 0.730 (n=6) | 0.458 (n=6) |
| answered_partial | 0.694 (n=15) | 0.781 (n=15) | 0.771 (n=11) | 0.386 (n=11) |
| not_covered | — | — | 0.259 (n=6) | 0.167 (n=6) |
| all | 0.711 (n=24) | 0.829 (n=24) | 0.627 (n=23) | 0.348 (n=23) |

## 5 lowest-scoring ids per metric

- **faithfulness**: 34 (0.14), 25 (0.47), 15 (0.53), 10 (0.56), 33 (0.62)
- **answer_relevancy**: 16 (0.53), 8 (0.63), 37 (0.64), 10 (0.66), 5 (0.69)
- **context_precision**: 7 (0.00), 9 (0.00), 20 (0.00), 21 (0.00), 24 (0.00)
- **context_recall**: 3 (0.00), 7 (0.00), 9 (0.00), 10 (0.00), 13 (0.00)

## Ragas faithfulness vs existing grades

Mapping fixed before the run: 1.0 = SUPPORTED, 0.5 to <1.0 = PARTIALLY_SUPPORTED, <0.5 = UNSUPPORTED.

- Three-class agreement: **7/24 (29%)**
- Binary (SUPPORTED vs not): **8/24 (33%)**
- Three-class by grader: manual 5/17 (29%), automated 2/7 (29%)

Confusion (existing -> ragas):

- PARTIALLY_SUPPORTED -> PARTIALLY_SUPPORTED: 7
- SUPPORTED -> PARTIALLY_SUPPORTED: 14
- SUPPORTED -> UNSUPPORTED: 2
- UNSUPPORTED -> PARTIALLY_SUPPORTED: 1

| id | status | ragas | ragas label | existing | grader |
|---|---|---|---|---|---|
| 1 | answered | 0.812 | PARTIALLY_SUPPORTED | SUPPORTED | automated |
| 2 | answered | 0.739 | PARTIALLY_SUPPORTED | SUPPORTED | automated |
| 3 | answered_partial | 0.773 | PARTIALLY_SUPPORTED | SUPPORTED | automated |
| 4 | answered | 0.778 | PARTIALLY_SUPPORTED | SUPPORTED | automated |
| 5 | answered_partial | 0.769 | PARTIALLY_SUPPORTED | PARTIALLY_SUPPORTED | automated |
| 7 | answered_partial | 0.875 | PARTIALLY_SUPPORTED | SUPPORTED | automated |
| 8 | answered_partial | 0.773 | PARTIALLY_SUPPORTED | PARTIALLY_SUPPORTED | automated |
| 10 | answered_partial | 0.556 | PARTIALLY_SUPPORTED | UNSUPPORTED | manual |
| 11 | answered_partial | 0.778 | PARTIALLY_SUPPORTED | PARTIALLY_SUPPORTED | manual |
| 12 | answered_partial | 0.667 | PARTIALLY_SUPPORTED | SUPPORTED | manual |
| 13 | answered | 0.84 | PARTIALLY_SUPPORTED | SUPPORTED | manual |
| 14 | answered | 0.85 | PARTIALLY_SUPPORTED | SUPPORTED | manual |
| 15 | answered_partial | 0.526 | PARTIALLY_SUPPORTED | SUPPORTED | manual |
| 16 | answered_partial | 0.667 | PARTIALLY_SUPPORTED | PARTIALLY_SUPPORTED | manual |
| 17 | answered_partial | 0.87 | PARTIALLY_SUPPORTED | PARTIALLY_SUPPORTED | manual |
| 19 | answered_partial | 0.75 | PARTIALLY_SUPPORTED | PARTIALLY_SUPPORTED | manual |
| 22 | answered | 0.8 | PARTIALLY_SUPPORTED | SUPPORTED | manual |
| 25 | answered | 0.467 | UNSUPPORTED | SUPPORTED | manual |
| 31 | answered_partial | 0.739 | PARTIALLY_SUPPORTED | SUPPORTED | manual |
| 32 | answered | 0.667 | PARTIALLY_SUPPORTED | SUPPORTED | manual |
| 33 | answered_partial | 0.619 | PARTIALLY_SUPPORTED | PARTIALLY_SUPPORTED | manual |
| 34 | answered_partial | 0.143 | UNSUPPORTED | SUPPORTED | manual |
| 36 | answered | 0.7 | PARTIALLY_SUPPORTED | SUPPORTED | manual |
| 37 | answered_partial | 0.9 | PARTIALLY_SUPPORTED | SUPPORTED | manual |
