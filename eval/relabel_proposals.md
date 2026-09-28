# Relabel proposals

Diabetes-adjacent questions (outside the four charter areas) where the corpus produced an answer the
faithfulness grader rated SUPPORTED — meaning the corpus actually has good evidence for these, even though
they fall outside `config/scope_charter.md`'s four core areas. **These are proposals for the maintainer,
not changes** — `eval/eval_set_v2.json`'s labels are untouched.

No candidates found in this run: no adjacent-labeled question was answered by the new pipeline (Stage B returned `partial` or `insufficient` for all 17, which the decision matrix routes to `out_of_scope`).

## Named in the task: ids 25, 33, 37 (evidence from the previous pipeline)

Answers and sentence verdicts below are from `data/eval_v1_faithfulness_v2_5k.json` (previous pipeline, previous grader), not from this run. That file grades each sentence separately and has no overall verdict, so the tally is shown as-is. Passages are cited in each verdict note; full passage text is in that file's source run.

### id 25 — out_of_scope_diabetes_adjacent / retinopathy

**Question:** What eye complications require laser treatment in diabetic retinopathy?

**Previous-pipeline answer:** Laser treatment, known as panretinal photocoagulation, is used to treat proliferative diabetic retinopathy, the advanced stage of diabetic retinopathy in which abnormal new blood vessels grow in the retina. Laser photocoagulation is also named, alongside anti-VEGF injections, corticosteroids, and vitreoretinal surgery, among the treatments that have improved outcomes for diabetic retinopathy in general. However, none of these treatments, including laser therapy, can permanently stop diabetic retinopathy from getting worse. Even with treatments like panretinal photocoagulation, some eyes with proliferative diabetic retinopathy still go on to develop serious complications such as bleeding inside the eye (vitreous hemorrhage), scar-tissue-related retinal detachment, or a severe form of glaucoma linked to abnormal blood vessels. This evidence on laser treatment comes from reviews covering mixed diabetes populations as well as studies specific to type 2 diabetes.

**Sentence verdicts (previous grader):** PARTIALLY_SUPPORTED 1, SUPPORTED 4

- [1] SUPPORTED: Excerpt 5 directly states that panretinal photocoagulation (PRP) is used for proliferative diabetic retinopathy, which involves abnormal new blood vessel growth.
- [2] SUPPORTED: Excerpt 3 explicitly lists 'laser photocoagulation' alongside anti-VEGF therapy, corticosteroids, and vitreoretinal surgery as treatments that have improved outcomes for diabetic retinopathy.
- [3] SUPPORTED: Excerpt 3 states that none of these treatments 'can permanently prevent disease progression,' directly supporting this claim about laser therapy's limitations.
- [4] SUPPORTED: Excerpt 5 explicitly lists vitreous hemorrhage, tractional retinal detachment, and neovascular glaucoma as complications that can occur 'despite apparently adequate therapy' including PRP.
- [5] PARTIALLY_SUPPORTED: While Excerpts 3 and 5 do discuss laser treatment and diabetic retinopathy, they do not specifically characterize the populations as 'mixed diabetes populations' versus 'type 2 diabetes specific,' making this framing partially inferred.

**This run:** scope=adjacent, evidence verdict=partial, status=out_of_scope. Stage B reason: C5 mentions panretinal photocoagulation for PDR, but the passages do not directly specify which other eye complications (DME, severe NPDR) require laser treatment, and do not explicitly state that PDR, DME, or severe NPDR 'require' laser as opposed to being treated with other modalities.

**For relabeling to in_scope:** the previous pipeline produced a mostly sentence-supported answer, so the corpus demonstrably holds relevant evidence, and users with type 2 diabetes plausibly ask this.

**Against relabeling:** it falls in an area the charter lists as adjacent by design (complications management), the new Stage B judged the retrieved evidence only partial, and broadening the charter to fit incidental corpus coverage would drift from the four-area product intent.

### id 33 — out_of_scope_diabetes_adjacent / foot_care

**Question:** What wound care protocol is typically used for treating a diabetic foot ulcer?

**Previous-pipeline answer:** Typical wound care for a diabetic foot ulcer usually combines controlling blood sugar, treating any infection, relieving pressure on the wound, and using dressings or other methods that help the wound heal. This general approach is described in diabetes research that covers a mix of diabetes types, with some studies focused specifically on type 2 diabetes. Pressure relief, called offloading, can be done without surgery, but non-surgical methods are sometimes linked to poor patient compliance, skin irritation, and limited ability to check on the wound. Because of this, surgical offloading is sometimes used instead, and it tends to work best when combined with good blood sugar control, careful patient selection, and coordinated care after the procedure. Advanced wound dressings, including hydrogels, are commonly studied and compared against standard care as part of building treatment guidelines for diabetic foot ulcers. One randomized trial in people with type 2 diabetes found that alternating honey and hydrogel treatment healed foot ulcers faster, with an average healing time of 10.83 days, compared with 12.20 days for honey alone, 13.97 days for hydrogel alone, and 14.03 days for a fucidin ointment control group, and this difference was statistically significant. Another randomized trial tested a hydrogel enriched with sodium alginate and vitamins A and E, describing it as a newer option that helps clean and prepare the wound bed for healing. Topical Chinese herbal medicine formulas have also been reviewed as a possible treatment for diabetic foot ulcers, valued in Traditional Chinese Medicine theory for detoxifying the wound, improving blood flow, and supporting healing.

**Sentence verdicts (previous grader):** PARTIALLY_SUPPORTED 1, SUPPORTED 7

- [1] SUPPORTED: Excerpt 2 explicitly states common therapeutic strategies include glycemic control, infection eradication, pressure alleviation, and wound healing facilitation.
- [2] PARTIALLY_SUPPORTED: Excerpt 2 discusses DFU broadly but doesn't specifically characterize the research as covering 'a mix of diabetes types' or emphasize type 2 as a focus.
- [3] SUPPORTED: Excerpt 4 directly states nonsurgical offloading is associated with noncompliance, skin irritation, and limited wound assessment capability.
- [4] SUPPORTED: Excerpt 4 states surgical offloading success depends on proper patient selection, glycemic control, and postoperative care, and it works best with multidisciplinary management.
- [5] SUPPORTED: Excerpt 3 discusses advanced wound dressings used in DFU treatment and mentions a meta-analysis comparing them to standard care for guideline development.
- [6] SUPPORTED: Excerpt 1 reports the exact healing times and statistical significance for all four treatment groups in the randomized trial.
- [7] SUPPORTED: Excerpt 5 describes hydrogel enriched with sodium alginate and vitamins A and E as a new treatment that aids in debridement and wound bed preparation.
- [8] SUPPORTED: Excerpt 2 discusses topical Chinese herbal medicine for DFU treatment and mentions its functions of detoxification, blood circulation promotion, blood stasis removal, and wound healing enhancement.

**This run:** scope=adjacent, evidence verdict=partial, status=out_of_scope. Stage B reason: Individual protocol components are documented across passages (dressing, debridement, offloading, glycemic control, infection management), but no single passage or source presents these as a unified, integrated 'typical' protocol with explicit stepwise guidance for implementing wound care.

**For relabeling to in_scope:** the previous pipeline produced a mostly sentence-supported answer, so the corpus demonstrably holds relevant evidence, and users with type 2 diabetes plausibly ask this.

**Against relabeling:** it falls in an area the charter lists as adjacent by design (complications management), the new Stage B judged the retrieved evidence only partial, and broadening the charter to fit incidental corpus coverage would drift from the four-area product intent.

### id 37 — out_of_scope_diabetes_adjacent / dka

**Question:** How is diabetic ketoacidosis typically managed in the emergency department?

**Previous-pipeline answer:** Emergency treatment for diabetic ketoacidosis generally includes aggressive fluid resuscitation, insulin therapy, electrolyte management, and treating whatever triggered the episode. This description comes from research on diabetic ketoacidosis that happens after surgery in adults with type 2 diabetes. In hospitals more broadly, doctors also use insulin infusion protocols, sometimes guided by computer-based electronic glycemic management systems, to control blood sugar during treatment. A 2026 meta-analysis of 3,874 hospitalized patients with diabetic ketoacidosis found that these electronic systems did not speed up the time it took for DKA to resolve, and did not shorten ICU stay, hospital stay, or the length of insulin infusion compared with standard paper-based protocols. The same review found lower recorded rates of low blood sugar with the electronic systems, but the researchers said this finding is uncertain because of weaknesses and inconsistency across the underlying studies. This evidence comes from a mix of general diabetes populations for the insulin-protocol research and specifically from people with type 2 diabetes for the postoperative management description.

**Sentence verdicts (previous grader):** NOT_APPLICABLE 1, PARTIALLY_SUPPORTED 1, SUPPORTED 4

- [1] SUPPORTED: Excerpt 3 explicitly states management requires 'aggressive fluid resuscitation, insulin therapy, electrolyte management, and addressing any precipitating factors.'
- [2] PARTIALLY_SUPPORTED: Excerpt 3 discusses postoperative DKA in type 2 diabetes, but the research review examines both type 1 and type 2 diabetes, so limiting the description to type 2 is somewhat restrictive.
- [3] SUPPORTED: Excerpt 2 discusses insulin infusion protocols and electronic glycemic management systems for hospitalized DKA patients, supporting this statement.
- [4] SUPPORTED: Excerpt 2 states the pooled analysis 'showed no statistically significant difference between groups' for time to DKA resolution and reports on ICU/hospital LOS and insulin infusion duration.
- [5] SUPPORTED: Excerpt 2 explicitly states 'Lower recorded hypoglycemia was observed with eGMSs, but this finding should be considered hypothesis-generating because of serious-to-critical risk of bias.'
- [6] NOT_APPLICABLE: This is a framing/caveat statement about the sources of evidence rather than a factual claim about DKA management.

**This run:** scope=adjacent, evidence verdict=partial, status=out_of_scope. Stage B reason: Only one management component (insulin infusion via eGMS) has direct support; fundamental ED management approaches (fluid resuscitation, insulin, electrolytes) are mentioned only for postoperative type 2 DKA, not for typical ED presentation across all DKA populations.

**For relabeling to in_scope:** the previous pipeline produced a mostly sentence-supported answer, so the corpus demonstrably holds relevant evidence, and users with type 2 diabetes plausibly ask this.

**Against relabeling:** it falls in an area the charter lists as adjacent by design (complications management), the new Stage B judged the retrieved evidence only partial, and broadening the charter to fit incidental corpus coverage would drift from the four-area product intent.
