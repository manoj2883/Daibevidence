# Ragas faithfulness without definition, gap and safety statements

Judge `claude-haiku-5-5`; method in the `scripts/run_ragas_eval.py` docstring (`--filtered`). 24 answered rows. Statements are re-extracted, so counts differ slightly from the first run.

Statement kinds (unsupported / total): finding 21/283, definition 50/103, gap 16/47, safety 17/18

| Variant | answered | answered_partial | all | agreement 3-class | agreement binary |
|---|---|---|---|---|---|
| first run (abstract text, all statements) | 0.739 (n=9) | 0.694 (n=15) | 0.711 (n=24) | 7/24 (29%) | 8/24 (33%) |
| generator contexts, all statements | 0.800 (n=9) | 0.725 (n=15) | 0.753 (n=24) | 7/24 (29%) | 8/24 (33%) |
| generator contexts, findings only | 0.955 (n=9) | 0.909 (n=15) | 0.926 (n=24) | 16/24 (67%) | 17/24 (71%) |

Rows with no finding statements (excluded from findings-only): none

## Per id

| id | status | first run | all statements | findings only | findings label | existing | unsupported findings |
|---|---|---|---|---|---|---|---|
| 1 | answered | 0.81 | 0.71 | 0.92 | PARTIALLY_SUPPORTED | SUPPORTED | Research on eating patterns like a Mediterranean-style diet suggests that a Mediterranean-style diet can lower the risk of developing type 2 diabetes in adults who are at higher risk. |
| 2 | answered | 0.74 | 0.81 | 1.00 | SUPPORTED | SUPPORTED | — |
| 3 | answered_partial | 0.77 | 0.82 | 1.00 | SUPPORTED | SUPPORTED | — |
| 4 | answered | 0.78 | 0.77 | 0.90 | PARTIALLY_SUPPORTED | SUPPORTED | The improvement in HbA1c was bigger when people followed a vegetarian or vegan diet for more than 12 weeks. / Reducing body weight, body mass index, and waist size can further support blood sugar management in people with type 2 diabetes. |
| 5 | answered_partial | 0.77 | 0.80 | 0.88 | PARTIALLY_SUPPORTED | PARTIALLY_SUPPORTED | Studies disagree on whether time-restricted eating actually lowers HbA1c. / This evidence comes from studies in adults with type 2 diabetes. |
| 7 | answered_partial | 0.88 | 0.88 | 1.00 | SUPPORTED | SUPPORTED | — |
| 8 | answered_partial | 0.77 | 0.78 | 0.94 | PARTIALLY_SUPPORTED | PARTIALLY_SUPPORTED | A more recent and larger review of eleven trials found that continuous glucose monitoring (CGM) did not change HbA1c compared with self-monitoring of blood glucose (SMBG). |
| 10 | answered_partial | 0.56 | 0.68 | 0.92 | PARTIALLY_SUPPORTED | UNSUPPORTED | This evidence comes from research involving people with type 2 diabetes. |
| 11 | answered_partial | 0.78 | 0.86 | 0.90 | PARTIALLY_SUPPORTED | PARTIALLY_SUPPORTED | In one large trial of military veterans with type 2 diabetes, bigger swings in average blood sugar test results predicted a higher chance of dying from any cause. |
| 12 | answered_partial | 0.67 | 0.67 | 1.00 | SUPPORTED | SUPPORTED | — |
| 13 | answered | 0.84 | 0.83 | 1.00 | SUPPORTED | SUPPORTED | — |
| 14 | answered | 0.85 | 0.86 | 0.95 | PARTIALLY_SUPPORTED | SUPPORTED | All of the research on intensive low-calorie diets for remission was done in adults with type 2 diabetes. |
| 15 | answered_partial | 0.53 | 0.50 | 0.82 | PARTIALLY_SUPPORTED | SUPPORTED | The evidence comes from research on people with type 2 diabetes. / The comparison between visceral fat and subcutaneous fat is pieced together from several separate studies. |
| 16 | answered_partial | 0.67 | 0.55 | 0.75 | PARTIALLY_SUPPORTED | PARTIALLY_SUPPORTED | Large reviews of clinical trial data give a mixed picture on whether muscle loss from GLP-1 receptor agonists is out of proportion to overall weight loss. / All of this research was done in adults with type 2 diabetes who were overweight or living with obesity. |
| 17 | answered_partial | 0.87 | 0.80 | 0.83 | PARTIALLY_SUPPORTED | PARTIALLY_SUPPORTED | The combination of sarcopenia and excess body fat is considered especially harmful in older adults compared with having only muscle loss or only excess weight. / In older adults, the combination of sarcopenia and excess body fat is tied to worse insulin resistance. / In older adults, the combination of sarcopenia and excess body fat is tied to ongoing low-level inflammation in the body. |
| 19 | answered_partial | 0.75 | 0.68 | 0.93 | PARTIALLY_SUPPORTED | PARTIALLY_SUPPORTED | In the year-long clinical trial, blood vitamin B12 levels in people with type 2 diabetes on metformin rose sharply, from a low level back to a normal level. |
| 22 | answered | 0.80 | 0.76 | 0.91 | PARTIALLY_SUPPORTED | SUPPORTED | A review focused on the surgery setting noted that changes in eating around operations can raise the risk of ketoacidosis in people taking SGLT2 inhibitors. |
| 25 | answered | 0.47 | 0.88 | 1.00 | SUPPORTED | SUPPORTED | — |
| 31 | answered_partial | 0.74 | 0.80 | 1.00 | SUPPORTED | SUPPORTED | — |
| 32 | answered | 0.67 | 0.74 | 0.92 | PARTIALLY_SUPPORTED | SUPPORTED | The information in the answer comes mainly from research in people with type 2 diabetes. |
| 33 | answered_partial | 0.62 | 0.73 | 0.92 | PARTIALLY_SUPPORTED | PARTIALLY_SUPPORTED | Debridement helps the healthy tissue underneath a diabetic foot ulcer heal. |
| 34 | answered_partial | 0.14 | 0.43 | 0.75 | PARTIALLY_SUPPORTED | SUPPORTED | One study found that wearable, specialized footwear lowered the chance of getting a diabetic foot ulcer. |
| 36 | answered | 0.70 | 0.84 | 1.00 | SUPPORTED | SUPPORTED | — |
| 37 | answered_partial | 0.90 | 0.90 | 1.00 | SUPPORTED | SUPPORTED | — |
