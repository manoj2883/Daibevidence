from src.ingest.population import classify_population, classify_population_two_pass


def test_classify_type1():
    assert classify_population("A cohort of adolescents with type 1 diabetes (T1DM)...") == "type1"


def test_classify_type2():
    assert classify_population("Patients with type 2 diabetes mellitus (T2DM) were enrolled...") == "type2"


def test_classify_gestational():
    assert classify_population("Women diagnosed with gestational diabetes (GDM) during pregnancy...") == "gestational"


def test_classify_prediabetes():
    assert classify_population("Adults with prediabetes and impaired glucose tolerance...") == "prediabetes"


def test_classify_mixed_when_multiple_populations_mentioned():
    text = "Comparing outcomes between type 1 diabetes and type 2 diabetes cohorts..."
    assert classify_population(text) == "mixed"


def test_classify_mixed_when_unclear():
    assert classify_population("A general review of dietary patterns and metabolic health.") == "mixed"


def test_classify_empty_text():
    assert classify_population("") == "mixed"


def test_two_pass_mesh_match_is_high_confidence():
    population, confidence = classify_population_two_pass(["Diabetes Mellitus, Type 2", "Humans"], "some abstract text")
    assert population == "type2"
    assert confidence == "high"


def test_two_pass_multiple_mesh_types_is_mixed_high_confidence():
    population, confidence = classify_population_two_pass(
        ["Diabetes Mellitus, Type 1", "Diabetes Mellitus, Type 2"], "text"
    )
    assert population == "mixed"
    assert confidence == "high"


def test_two_pass_falls_back_to_keyword_heuristic_low_confidence():
    population, confidence = classify_population_two_pass(["Humans", "Diet, Food, and Nutrition"], "Adults with prediabetes and impaired glucose tolerance...")
    assert population == "prediabetes"
    assert confidence == "low"


def test_two_pass_no_mesh_and_no_keyword_match_is_mixed_low_confidence():
    population, confidence = classify_population_two_pass([], "A general review of dietary patterns.")
    assert population == "mixed"
    assert confidence == "low"


def test_two_pass_mesh_match_is_case_insensitive():
    population, confidence = classify_population_two_pass(["diabetes, gestational"], "text")
    assert population == "gestational"
    assert confidence == "high"
