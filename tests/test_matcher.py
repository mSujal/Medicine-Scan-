from pathlib import Path

from src.matcher import extract_strengths, load_name_index, normalize, text_scores

REF = {
    "paracetamol": ["paracetamol", "acetaminophen"],
    "ibuprofen": ["ibuprofen"],
    "hyoscine butylbromide": ["hyoscine butylbromide"],
    "domperidone": ["domperidone"],
    "ondansetron": ["ondansetron"],
    "fexofenadine": ["fexofenadine"],
}
REAL_DB = Path(__file__).resolve().parents[1] / "data" / "reference_db.csv"

def test_normalize_lowercases_and_collapses_spaces():
    assert normalize("  PARACETAMOL   500 MG ") == "paracetamol 500 mg"

def test_extract_strengths_simple():
    assert extract_strengths("Paracetamol 500mg") == ["500 mg"]


def test_extract_strengths_slash_form():
    assert extract_strengths("Syrup 125 mg/5 ml") == ["125 mg", "5 ml"]


def test_extract_strengths_percent():
    assert extract_strengths("Oxymetazoline 0.05 %") == ["0.05 %"]


def test_exact_match_scores_highest():
    scores = text_scores("Paracetamol 500 mg", REF)
    assert scores["paracetamol"] == 1.0
    assert max(scores, key=scores.get) == "paracetamol"


def test_alias_matches_ingredient():
    scores = text_scores("acetaminophen 500 mg", REF)
    assert "paracetamol" in scores


def test_truncated_name_resolves_but_scores_below_exact():
    trunc = text_scores("parace 500 mg", REF)["paracetamol"]
    exact = text_scores("paracetamol 500 mg", REF)["paracetamol"]
    assert 0.8 <= trunc < exact == 1.0


def test_ocr_digit_confusion_is_repaired():
    assert "paracetamol" in text_scores("paracetamo1 500 mg", REF)
    assert "ibuprofen" in text_scores("IBUPR0FEN 400 mg", REF)


def test_multiword_ingredient_matches():
    scores = text_scores("Hyoscine Butylbromide 10 mg", REF)
    assert scores.get("hyoscine butylbromide") == 1.0


def test_multiple_ingredients_in_one_text():
    scores = text_scores("Ibuprofen 200 mg and Paracetamol 500 mg", REF)
    assert "ibuprofen" in scores and "paracetamol" in scores


def test_unrelated_text_returns_empty():
    assert text_scores("Keep out of reach of children. Store below 25 C", REF) == {}


def test_short_tokens_are_ignored():
    assert text_scores("the of mg", REF) == {}


def test_real_reference_db_loads_and_matches():
    index = load_name_index(REAL_DB)
    assert "acetaminophen" in index["paracetamol"]
    scores = text_scores("parace 500 mg", index)
    assert "paracetamol" in scores


def test_no_false_positive_on_back_boilerplate():
    assert text_scores("Iron OXide Red and Titanium Dioxide, store in dry place", REF) == {} 


def test_longer_name_does_not_trigger_similar_ingredient():
    scores = text_scores("Fexofenadine Hydrochloride USP 180 mg", REF)
    assert list(scores) == ["fexofenadine"]