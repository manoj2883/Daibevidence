from src.ingest.population import classify_population


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
