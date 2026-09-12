"""
Curated background-tier source documents: authoritative patient-education
content from ADA, NIDDK, and CDC public patient pages, covering definitions,
diagnostic criteria, and general disease mechanism — the "reference layer"
alongside the PubMed evidence corpus.

Text was fetched from the live pages listed below and lightly cleaned of
navigation/boilerplate; it is a faithful extraction of the page's substance,
not necessarily a byte-exact quote (comparable in spirit to how any citation
tool renders a source, but worth being explicit about here since these are
attributed to real public-health organizations). Fetched 2026-09.

Background chunks are for definitional/explanatory support only — see the
usage restriction enforced in src/rag/chain.py's SYSTEM_PROMPT.
"""
from typing import Any, Dict, List

BACKGROUND_DOCUMENTS: List[Dict[str, Any]] = [
    {
        "publisher": "NIDDK",
        "title": "What Is Diabetes?",
        "source_url": "https://www.niddk.nih.gov/health-information/diabetes/overview/what-is-diabetes",
        "population": "mixed",
        "text": (
            "Diabetes is a disease that occurs when blood glucose (blood sugar) becomes too high. "
            "Glucose serves as the body's primary energy source, produced internally and obtained from food. "
            "The pancreas produces insulin, a hormone that facilitates glucose entry into cells for energy use. "
            "In diabetes, the body either produces insufficient or no insulin, or fails to use insulin properly. "
            "Consequently, glucose remains in the bloodstream rather than reaching cells. Elevated blood glucose "
            "increases damage risk to the eyes, kidneys, nerves, and heart. The disease also correlates with "
            "certain cancer types. Preventive and management measures may reduce diabetes-related health "
            "complications.\n\n"
            "Diabetes Types. Type 1: The body produces minimal or no insulin as the immune system attacks and "
            "destroys insulin-producing pancreatic cells. Usually diagnosed in children and young adults, though "
            "it can develop at any age. Daily insulin is necessary for survival. Type 2: Body cells don't utilize "
            "insulin properly. The pancreas may produce insulin but insufficient amounts to maintain normal blood "
            "glucose levels. This represents the most common diabetes form, more likely developing in those with "
            "overweight or obesity and family history. Type 2 can appear at any age, including childhood. "
            "Gestational: Develops during pregnancy and typically resolves post-delivery, though increases future "
            "type 2 diabetes risk. Prediabetes: Blood glucose exceeds normal levels but remains below type 2 "
            "diabetes diagnostic thresholds, elevating future type 2 diabetes and heart disease risk. Other "
            "types: Monogenic diabetes results from single gene mutations. Diabetes can also result from "
            "pancreas removal or damage from conditions like cystic fibrosis or pancreatitis.\n\n"
            "Prevalence: Over 133 million Americans have diabetes or prediabetes. As of 2019, 37.3 million "
            "people (11.3% of the population) had diabetes, with over 25% of those aged 65+ affected. Nearly 25% "
            "of adults with diabetes were unaware of their condition. Type 2 comprises 90-95% of cases. "
            "Approximately 96 million U.S. adults (38%) had prediabetes in 2019.\n\n"
            "Complications: High blood glucose damages the heart, kidneys, feet, and eyes over time. Managing "
            "blood glucose, blood pressure, and cholesterol levels helps prevent future complications."
        ),
    },
    {
        "publisher": "ADA",
        "title": "Type 1 Diabetes",
        "source_url": "https://diabetes.org/about-diabetes/type-1",
        "population": "type1",
        "text": (
            "Type 1 diabetes is an autoimmune disease where the immune system mistakenly destroys beta cells in "
            "the pancreas that produce insulin. When sufficient beta cells are destroyed, the pancreas cannot "
            "make insulin or produces insufficient amounts, requiring insulin therapy for survival. Insulin "
            "functions as a hormone enabling blood glucose to enter body cells for energy use. Without adequate "
            "insulin, blood glucose accumulates in the bloodstream, causing hyperglycemia. Prolonged high blood "
            "glucose damages the body and can lead to diabetes-related complications if untreated.\n\n"
            "Symptoms: Individuals should contact healthcare providers if experiencing urinating often, extreme "
            "thirst, persistent hunger despite eating, severe fatigue, blurry vision, slow-healing cuts or "
            "bruises, or weight loss despite increased food intake. Notably, people may lack symptoms entirely "
            "when type 1 diabetes first develops.\n\n"
            "Diagnosis and Special Considerations: Type 1 diabetes typically appears in young people but can "
            "develop at any age. Scientists remain uncertain about prevention methods or triggering mechanisms. "
            "Adults are sometimes misdiagnosed with type 2 diabetes, particularly those with additional risk "
            "factors like overweight status, physical inactivity, or high blood pressure. For those with family "
            "history, healthcare providers may recommend blood testing for islet autoantibodies. Positive "
            "results indicate risk but not current diabetes; counseling about symptoms and diabetic ketoacidosis "
            "prevention is recommended.\n\n"
            "Honeymoon Phase: Some newly diagnosed patients experience a temporary period where the body "
            "produces sufficient insulin to lower blood glucose, potentially requiring less insulin therapy. "
            "This honeymoon phase lasts from one week to one year. Symptom absence does not indicate diabetes "
            "resolution; the pancreas eventually loses insulin production capacity."
        ),
    },
    {
        "publisher": "ADA",
        "title": "Type 2 Diabetes",
        "source_url": "https://diabetes.org/about-diabetes/type-2",
        "population": "type2",
        "text": (
            "In type 2 diabetes, the body does not use insulin properly — a condition referred to as insulin "
            "resistance. Initially, beta cells produce extra insulin to compensate for this resistance. However, "
            "over time the pancreas cannot generate sufficient insulin to maintain normal blood glucose levels. "
            "Type 2 diabetes typically develops in middle-aged and older adults, though it is increasingly "
            "appearing in younger populations.\n\n"
            "Common Symptoms: Typical symptoms include urinating frequently, experiencing excessive thirst, "
            "feeling hungry despite eating, extreme fatigue, blurry vision, cuts or bruises that heal slowly, "
            "and tingling, pain, or numbness in the hands or feet. Some individuals with type 2 diabetes "
            "experience symptoms so mild they may go unnoticed.\n\n"
            "Treatment Approach: Treatment for people with type 2 diabetes can include an eating plan, physical "
            "activity, and oral or injectable medications (including insulin) to help achieve target blood "
            "glucose levels.\n\n"
            "Clinical Significance: Early detection and treatment can decrease the risk of developing diabetes "
            "complications. Type 1 and type 2 diabetes have different causes and typically require different "
            "treatment approaches, though newly diagnosed adults with type 1 may sometimes exhibit symptoms "
            "resembling type 2, creating potential diagnostic confusion."
        ),
    },
    {
        "publisher": "ADA",
        "title": "Gestational Diabetes",
        "source_url": "https://diabetes.org/about-diabetes/gestational-diabetes",
        "population": "gestational",
        "text": (
            "Gestational diabetes (GDM) is diabetes diagnosed during pregnancy, affecting up to 9% of "
            "pregnancies annually in the U.S. A diagnosis does not indicate pre-existing diabetes nor does it "
            "guarantee post-pregnancy diabetes development.\n\n"
            "Mechanism: The placenta's hormones, which support the baby's growth, can sometimes block the "
            "mother's insulin, leading to insulin resistance. This makes it harder for the body to use insulin "
            "effectively, requiring increased insulin production. When the body cannot produce sufficient "
            "insulin during pregnancy, glucose accumulates in the blood, resulting in elevated blood sugar "
            "levels.\n\n"
            "Management Approach: Treatment centers on maintaining normal blood glucose levels through several "
            "interventions: personalized meal plans, regular physical activity, daily blood glucose testing, and "
            "potentially insulin injections. Early screening and treatment are emphasized as critical for "
            "preventing health complications for both mother and baby.\n\n"
            "Dietary and Lifestyle Recommendations: Diet serves as a crucial management tool, with healthcare "
            "providers developing individualized meal plans. The Diabetes Plate method is suggested as an "
            "accessible starting point. Exercise is described as vital, with recommendations to collaborate "
            "with healthcare providers to determine safe activity levels during pregnancy.\n\n"
            "Post-Pregnancy Risk: Approximately 50% of women with GDM subsequently develop type 2 diabetes. The "
            "National Diabetes Prevention Program, a lifestyle intervention program, demonstrates 58% risk "
            "reduction for type 2 diabetes development in qualifying individuals with GDM history."
        ),
    },
    {
        "publisher": "ADA",
        "title": "Prediabetes",
        "source_url": "https://diabetes.org/about-diabetes/prediabetes",
        "population": "prediabetes",
        "text": (
            "Over 115 million Americans have prediabetes, and approximately 80% remain unaware of their "
            "condition. The disease is characterized by blood glucose levels that are elevated above normal "
            "range but insufficient for a type 2 diabetes diagnosis.\n\n"
            "Prediabetes typically presents with no clear symptoms, meaning individuals may carry the condition "
            "without knowing it. However, before people develop type 2 diabetes, they almost always have "
            "prediabetes — where blood glucose levels are higher than normal but not yet high enough to be "
            "diagnosed as diabetes.\n\n"
            "The condition does not inevitably lead to type 2 diabetes. Through early treatment combined with "
            "moderate lifestyle modifications, for some people with prediabetes, early treatment as well as "
            "moderate lifestyle changes can actually return blood glucose (blood sugar) levels to a normal "
            "range, effectively preventing or delaying type 2 diabetes.\n\n"
            "Key preventive strategies include adapting food choices and increasing daily physical activity to "
            "achieve weight loss if necessary. A CDC-recognized lifestyle change program, guided by a trained "
            "lifestyle coach using an approved curriculum, could reduce diabetes development risk by "
            "approximately half. Even modest changes in behavior can significantly impact outcomes.\n\n"
            "Individuals should consult their healthcare provider for testing if they suspect prediabetes. "
            "Working collaboratively with a healthcare team to develop a personalized treatment and lifestyle "
            "plan is recommended for optimal disease management and prevention."
        ),
    },
    {
        "publisher": "ADA",
        "title": "Diabetes Diagnosis and Prediabetes Overview",
        "source_url": "https://diabetes.org/about-diabetes/diagnosis",
        "population": "mixed",
        "text": (
            "There are several ways to diagnose diabetes, with each method typically requiring confirmation "
            "through repeat testing on a second day. Testing should occur in a healthcare setting, though "
            "doctors may waive the second test if blood glucose levels are very high or classic symptoms "
            "accompany one positive result.\n\n"
            "A1C Test: This measures average blood glucose over two to three months and requires no fasting. "
            "Diabetes is diagnosed at an A1C of greater than or equal to 6.5%. Normal results are less than "
            "5.7%, while prediabetes ranges from 5.7% to 6.4%.\n\n"
            "Fasting Plasma Glucose (FPG): After fasting for at least 8 hours, diabetes is diagnosed at fasting "
            "blood glucose of greater than or equal to 126 mg/dl. Normal findings are less than 100 mg/dL, with "
            "prediabetes between 100 mg/dl to 125 mg/dL.\n\n"
            "Oral Glucose Tolerance Test (OGTT): This two-hour test checks glucose before and after consuming a "
            "sweet drink. Diabetes is diagnosed at two-hour blood glucose of greater than or equal to 200 mg/dl. "
            "Normal is less than 140 mg/dL; prediabetes ranges from 140 to 199 mg/dL.\n\n"
            "Random Plasma Glucose Test: Diabetes is diagnosed at blood glucose of greater than or equal to 200 "
            "mg/dL when symptoms are severe.\n\n"
            "Prediabetes: Before type 2 diabetes develops, prediabetes typically emerges — elevated glucose that "
            "hasn't reached diabetic levels. There are no clear symptoms of prediabetes, so you may have it and "
            "not know it. Prediabetic results include an A1C of 5.7 to 6.4%, fasting blood glucose of 100 to 125 "
            "mg/dL, or an OGTT two-hour blood glucose of 140 to 199 mg/dL. Annual screening is recommended. "
            "Prevention research demonstrates that you can lower your risk for type 2 diabetes by 58% by losing "
            "7% of body weight and exercising moderately 30 minutes daily, five days weekly."
        ),
    },
    {
        "publisher": "CDC",
        "title": "About Type 1 Diabetes",
        "source_url": "https://www.cdc.gov/diabetes/about/about-type-1-diabetes.html",
        "population": "type1",
        "text": (
            "With type 1 diabetes, the pancreas either fails to produce insulin or produces very little. Insulin "
            "is essential for enabling blood sugar to enter cells for energy use. When insulin is absent, blood "
            "sugar accumulates in the bloodstream, causing damage and triggering diabetes symptoms and "
            "complications.\n\n"
            "Previously termed insulin-dependent or juvenile diabetes, type 1 typically emerges in children, "
            "adolescents, and young adults, though it can develop at any age. It accounts for approximately "
            "5-10% of diabetes cases and is less prevalent than type 2 diabetes.\n\n"
            "Currently, type 1 diabetes can't be prevented, but it can be treated effectively through "
            "doctor-recommended healthy lifestyles, blood sugar management, regular health checkups, and "
            "diabetes self-management education.\n\n"
            "Early-stage type 1 often presents no symptoms. As the condition progresses, symptoms can appear "
            "suddenly, in just a few weeks or months, and can be severe. Untreated diabetes risks serious or "
            "fatal health complications.\n\n"
            "Type 1 stems from an autoimmune reaction where the body mistakenly attacks beta cells in the "
            "pancreas that produce insulin. This destructive process may continue for months or years before "
            "symptoms appear. Genetic factors and environmental triggers like viruses may contribute, but diet "
            "and lifestyle do not cause the condition.\n\n"
            "Diagnosis involves a blood test. Doctors may screen for diabetes-related autoantibodies to detect "
            "early-stage type 1 before symptoms develop. Urine testing for ketones may also indicate type 1 "
            "rather than type 2.\n\n"
            "Management requires daily insulin injections or pump therapy, as stomach acid destroys oral "
            "insulin. Regular blood sugar monitoring, stress management, and healthy lifestyle habits — "
            "including proper nutrition, physical activity, blood pressure and cholesterol management — support "
            "treatment success."
        ),
    },
    {
        "publisher": "CDC",
        "title": "About Type 2 Diabetes",
        "source_url": "https://www.cdc.gov/diabetes/about/about-type-2-diabetes.html",
        "population": "type2",
        "text": (
            "More than 40 million Americans have diabetes, with approximately 90-95% having type 2 diabetes. "
            "While it most commonly develops in people 45 or older, increasing numbers of children, teens, and "
            "young adults are developing the condition.\n\n"
            "Symptoms and Detection: Type 2 diabetes symptoms often develop gradually over several years and may "
            "go unnoticed for extended periods. Sometimes no noticeable symptoms appear at all. A simple blood "
            "test can determine if someone has diabetes, though results from health fairs or pharmacies should "
            "be confirmed at a clinic or doctor's office.\n\n"
            "Risk Factors: People face increased risk if they have prediabetes, carry extra weight, are 45 or "
            "older, have a family history of type 2 diabetes, exercise fewer than three times weekly, have "
            "experienced gestational diabetes, or gave birth to a baby weighing 9 pounds or more. Individuals "
            "who are African American, Hispanic or Latino, American Indian, Alaska Native, Pacific Islander, or "
            "Asian American face higher risk. Non-alcoholic fatty liver disease also increases risk.\n\n"
            "Biological Mechanism: Insulin is a hormone made by your pancreas. It acts like a key to let blood "
            "sugar into cells in your body for use as energy. In type 2 diabetes, cells don't respond normally "
            "to insulin — called insulin resistance. The pancreas produces more insulin attempting to "
            "compensate, but eventually cannot maintain this response, causing blood sugar to rise and "
            "potentially leading to prediabetes and type 2 diabetes.\n\n"
            "Prevention and Management: Type 2 diabetes can be prevented or delayed through lifestyle "
            "modifications including weight loss, healthy eating, and regular physical activity. Management "
            "primarily involves personal responsibility with healthcare team support, potentially including "
            "insulin or other diabetes medications alongside continued healthy habits. Regular blood sugar "
            "monitoring and stress management through exercise, sleep, and relaxation techniques are "
            "recommended."
        ),
    },
    {
        "publisher": "CDC",
        "title": "Gestational Diabetes",
        "source_url": "https://www.cdc.gov/diabetes/about/gestational-diabetes.html",
        "population": "gestational",
        "text": (
            "Gestational diabetes can develop during pregnancy in women who don't already have diabetes. Every "
            "year, 5% to 9% of U.S. pregnancies are affected by gestational diabetes. Managing gestational "
            "diabetes can help ensure a healthy pregnancy and healthy baby.\n\n"
            "Gestational diabetes often doesn't produce symptoms. If symptoms occur, they may be mild, such as "
            "increased thirst or more frequent urination. Testing is necessary to confirm diagnosis.\n\n"
            "Higher risk exists for those who had gestational diabetes in a previous pregnancy, gave birth to a "
            "baby weighing over 9 pounds, are overweight, have a family history of type 2 diabetes, have "
            "polycystic ovary syndrome (PCOS), or are African American, Hispanic or Latino, American Indian, "
            "Alaska Native, Native Hawaiian, or Pacific Islander.\n\n"
            "The condition develops when your body can't make enough insulin during your pregnancy. Insulin is "
            "a hormone produced by the pancreas that allows blood sugar to enter cells for energy use. During "
            "pregnancy, the body produces more hormones and undergoes changes like weight gain, causing cells "
            "to use insulin less effectively — a condition called insulin resistance. This increases insulin "
            "requirements. All pregnant women experience some insulin resistance in late pregnancy, but some "
            "have it beforehand, increasing their likelihood of developing gestational diabetes.\n\n"
            "Lifestyle modifications before pregnancy may prevent gestational diabetes, including weight loss if "
            "overweight, healthy eating, and regular physical activity. Approximately half of women with "
            "gestational diabetes later develop type 2 diabetes.\n\n"
            "Gestational diabetes usually develops around the 24th week of pregnancy, with testing typically "
            "occurring between 24 and 28 weeks. Those at higher risk may be tested earlier.\n\n"
            "Management involves checking blood sugar levels, staying active, and eating healthy food in "
            "appropriate amounts at proper times. If lifestyle measures prove insufficient, doctors may "
            "prescribe insulin, metformin, or other medications."
        ),
    },
    {
        "publisher": "CDC",
        "title": "Prediabetes",
        "source_url": "https://www.cdc.gov/diabetes/basics/prediabetes.html",
        "population": "prediabetes",
        "text": (
            "With prediabetes, blood sugar levels are higher than normal but not high enough for a type 2 "
            "diabetes diagnosis. This condition elevates risks for type 2 diabetes, heart disease, and "
            "stroke.\n\n"
            "Mechanism: Insulin normally acts as a key allowing blood sugar into cells for energy. In "
            "prediabetes, cells don't respond normally to insulin. The pancreas produces additional insulin "
            "attempting to restore cellular response, but eventually cannot sustain this effort, causing blood "
            "sugar to rise and setting the stage for type 2 diabetes.\n\n"
            "Symptoms and Detection: Prediabetes often remains asymptomatic for years and frequently goes "
            "undetected until serious complications like type 2 diabetes emerge. Individuals should discuss "
            "blood sugar testing with their doctor if they have risk factors including being overweight, age 45 "
            "or older, family history of type 2 diabetes, low physical activity levels (less than three times "
            "weekly), prior gestational diabetes, delivery of a baby weighing over nine pounds, or polycystic "
            "ovary syndrome. Certain racial and ethnic groups face higher risk.\n\n"
            "Prevention: Weight loss of approximately 5-7% of body weight combined with regular physical "
            "activity — at least 150 minutes weekly of brisk walking or similar activity — can significantly "
            "reduce type 2 diabetes risk. The National Diabetes Prevention Program can lower risk by 58% "
            "overall and 71% for those over 60."
        ),
    },
]
