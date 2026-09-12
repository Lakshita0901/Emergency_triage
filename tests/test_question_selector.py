"""Comprehensive unit tests for the adaptive question selector engine.

Covers:
- is_contextually_relevant helper logic (direct chief complaint match, general fallback, symptom overlap).
- select_next_question returning None when:
    * No candidates remain (all fields already filled).
    * No candidates are contextually relevant to patient.
    * Question bank is empty or invalid.
- select_next_question selecting the correct highest-scored candidate across multiple
  hand-built state fixtures:
    * Direct chief complaint match (1.0 weight) vs. higher base general question (0.6 weight).
    * Skipping fields already populated in flat state or nested vitals/symptoms dicts.
    * Overlapping known symptoms triggering applicable questions.
    * End-to-end question prioritization on the production question_bank.json.
"""
import json
from pathlib import Path
import pytest

from agent.question_selector import select_next_question, is_contextually_relevant


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def production_question_bank():
    path = Path(__file__).resolve().parent.parent / "data" / "question_bank.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def controlled_question_bank():
    """Deterministic, hand-crafted question bank for exact mathematical score tests."""
    return {
        "questions": [
            {
                "id": "q_chest_pain",
                "field": "chest_pain",
                "question": "Are you having chest pain?",
                "risk_impact": 2.5,
                "uncertainty_reduction": 2.5,
                "applicable_to": ["cardiovascular", "chest_pain"],
            },
            {
                "id": "q_spo2",
                "field": "oxygen_saturation",
                "question": "What is your oxygen saturation?",
                "risk_impact": 3.0,
                "uncertainty_reduction": 3.0,
                "applicable_to": ["general", "respiratory"],
            },
            {
                "id": "q_pain_radiation",
                "field": "pain_radiation",
                "question": "Does pain radiate?",
                "risk_impact": 2.0,
                "uncertainty_reduction": 1.5,
                "applicable_to": ["chest_pain"],
            },
            {
                "id": "q_temperature",
                "field": "temperature",
                "question": "What is your temperature?",
                "risk_impact": 1.5,
                "uncertainty_reduction": 1.0,
                "applicable_to": ["general", "fever"],
            },
            {
                "id": "q_specialized_ortho",
                "field": "joint_swelling",
                "question": "Is your joint swollen?",
                "risk_impact": 1.0,
                "uncertainty_reduction": 1.0,
                "applicable_to": ["orthopedic", "joint_pain"],
            },
        ]
    }


# ---------------------------------------------------------------------------
# 1. is_contextually_relevant Tests
# ---------------------------------------------------------------------------

def test_relevance_general_always_applicable():
    q = {"applicable_to": ["general", "respiratory"]}
    state = {"chief_complaint": "abdominal_pain"}
    is_app, is_direct = is_contextually_relevant(state, q)
    assert is_app is True
    assert is_direct is False  # "abdominal_pain" is not "respiratory"


def test_relevance_general_with_direct_chief_complaint():
    q = {"applicable_to": ["general", "respiratory"]}
    state = {"chief_complaint": "respiratory"}
    is_app, is_direct = is_contextually_relevant(state, q)
    assert is_app is True
    assert is_direct is True  # "respiratory" directly matches


def test_relevance_direct_match_non_general():
    q = {"applicable_to": ["cardiovascular", "chest_pain"]}
    state = {"chief_complaint": "chest_pain"}
    is_app, is_direct = is_contextually_relevant(state, q)
    assert is_app is True
    assert is_direct is True


def test_relevance_overlap_via_known_symptom():
    q = {"applicable_to": ["chest_pain"]}
    # Chief complaint is shortness of breath, but chest_pain symptom is known
    state = {
        "chief_complaint": "shortness_of_breath",
        "chest_pain": True,
    }
    is_app, is_direct = is_contextually_relevant(state, q)
    assert is_app is True
    assert is_direct is False  # Overlap via symptom, not direct chief complaint


def test_relevance_overlap_via_nested_symptoms_dict():
    q = {"applicable_to": ["dizziness"]}
    state = {
        "chief_complaint": "fatigue",
        "symptoms": {"dizziness": True},
    }
    is_app, is_direct = is_contextually_relevant(state, q)
    assert is_app is True
    assert is_direct is False


def test_relevance_no_overlap_not_applicable():
    q = {"applicable_to": ["orthopedic", "fracture"]}
    state = {
        "chief_complaint": "chest_pain",
        "symptoms": {"chest_pain": True},
    }
    is_app, is_direct = is_contextually_relevant(state, q)
    assert is_app is False
    assert is_direct is False


# ---------------------------------------------------------------------------
# 2. select_next_question: Returning None when no candidates remain
# ---------------------------------------------------------------------------

def test_select_none_when_all_fields_filled_flat(controlled_question_bank):
    state = {
        "chief_complaint": "chest_pain",
        "chest_pain": True,
        "oxygen_saturation": 98.0,
        "pain_radiation": False,
        "temperature": 37.0,
        "joint_swelling": False,
    }
    selected = select_next_question(state, controlled_question_bank)
    assert selected is None


def test_select_none_when_all_fields_filled_in_nested_dicts(controlled_question_bank):
    state = {
        "chief_complaint": "chest_pain",
        "vitals": {
            "oxygen_saturation": 98.0,
            "temperature": 36.8,
        },
        "symptoms": {
            "chest_pain": True,
            "pain_radiation": True,
            "joint_swelling": False,
        },
    }
    selected = select_next_question(state, controlled_question_bank)
    assert selected is None


def test_select_none_when_no_candidate_is_applicable():
    bank = {
        "questions": [
            {
                "id": "q_eye",
                "field": "eye_redness",
                "question": "Is your eye red?",
                "risk_impact": 1.0,
                "uncertainty_reduction": 1.0,
                "applicable_to": ["ophthalmology", "eye_pain"],
            },
            {
                "id": "q_tooth",
                "field": "toothache",
                "question": "Do you have tooth pain?",
                "risk_impact": 1.0,
                "uncertainty_reduction": 1.0,
                "applicable_to": ["dental"],
            },
        ]
    }
    state = {
        "chief_complaint": "chest_pain",
        "symptoms": {"chest_pain": True},
    }
    selected = select_next_question(state, bank)
    assert selected is None


def test_select_none_on_empty_bank():
    assert select_next_question({"chief_complaint": "general"}, []) is None
    assert select_next_question({"chief_complaint": "general"}, {"questions": []}) is None
    assert select_next_question({"chief_complaint": "general"}, None) is None  # type: ignore


# ---------------------------------------------------------------------------
# 3. Hand-Built State Fixtures (Highest-Scored Question Verification)
# ---------------------------------------------------------------------------

def test_fixture_1_direct_match_trumps_higher_base_general(controlled_question_bank):
    """Fixture 1: Direct chief complaint match (1.0 weight) wins over general question (0.6 weight).

    q_chest_pain: risk 2.5 * unc 2.5 * 1.0 = 6.25
    q_spo2:       risk 3.0 * unc 3.0 * 0.6 = 5.40
    q_temp:       risk 1.5 * unc 1.0 * 0.6 = 0.90
    Expected winner: q_chest_pain (score 6.25)
    """
    state = {"chief_complaint": "chest_pain"}
    selected = select_next_question(state, controlled_question_bank)

    assert selected is not None
    assert selected["field"] == "chest_pain"
    assert selected["id"] == "q_chest_pain"
    assert selected["selection_score"] == 6.25
    assert "directly related to chief complaint" in selected["reason"]


def test_fixture_2_fallback_to_general_when_direct_field_filled(controlled_question_bank):
    """Fixture 2: When chest_pain is already answered, q_spo2 becomes highest candidate.

    q_chest_pain: already answered (skipped)
    q_spo2:       risk 3.0 * unc 3.0 * 0.6 = 5.40
    q_pain_radiation: risk 2.0 * unc 1.5 * 1.0 = 3.00 (direct match to chest_pain complaint)
    q_temp:       risk 1.5 * unc 1.0 * 0.6 = 0.90
    Expected winner: q_spo2 (score 5.40 > 3.00)
    """
    state = {
        "chief_complaint": "chest_pain",
        "chest_pain": True,
    }
    selected = select_next_question(state, controlled_question_bank)

    assert selected is not None
    assert selected["field"] == "oxygen_saturation"
    assert selected["id"] == "q_spo2"
    assert selected["selection_score"] == 5.40
    assert "oxygen_saturation" in selected["reason"]


def test_fixture_3_nested_vitals_skipping_and_symptom_overlap(controlled_question_bank):
    """Fixture 3: Nested vitals already contain SpO2, and known symptom activates relevant question.

    State: chief_complaint="trauma", vitals={"oxygen_saturation": 96.0}, symptoms={"chest_pain": True}
    q_chest_pain: skipped (in symptoms)
    q_spo2:       skipped (in vitals)
    q_pain_radiation: applicable via symptoms["chest_pain"] -> relevance 0.6 -> 2.0 * 1.5 * 0.6 = 1.80
    q_temperature:    applicable via general -> relevance 0.6 -> 1.5 * 1.0 * 0.6 = 0.90
    q_specialized_ortho: not applicable
    Expected winner: q_pain_radiation (score 1.80)
    """
    state = {
        "chief_complaint": "trauma",
        "vitals": {"oxygen_saturation": 96.0},
        "symptoms": {"chest_pain": True},
    }
    selected = select_next_question(state, controlled_question_bank)

    assert selected is not None
    assert selected["field"] == "pain_radiation"
    assert selected["id"] == "q_pain_radiation"
    assert selected["selection_score"] == 1.80


def test_fixture_4_respiratory_complaint_direct_multiplier():
    """Fixture 4: Hand-built respiratory fixture where SpO2 directly matches complaint."""
    bank = [
        {
            "id": "q_hr",
            "field": "heart_rate",
            "question": "What is your heart rate?",
            "risk_impact": 3.0,
            "uncertainty_reduction": 3.0,
            "applicable_to": ["general", "cardiovascular"],
        },
        {
            "id": "q_spo2",
            "field": "oxygen_saturation",
            "question": "What is your SpO2?",
            "risk_impact": 3.0,
            "uncertainty_reduction": 3.0,
            "applicable_to": ["general", "respiratory"],
        },
    ]
    # SpO2 gets 1.0 context_relevance (3.0 * 3.0 * 1.0 = 9.0)
    # HR gets 0.6 context_relevance (3.0 * 3.0 * 0.6 = 5.4)
    state = {"chief_complaint": "respiratory"}
    selected = select_next_question(state, bank)

    assert selected is not None
    assert selected["id"] == "q_spo2"
    assert selected["selection_score"] == 9.0
    assert "directly related to chief complaint" in selected["reason"]


# ---------------------------------------------------------------------------
# 4. Production question_bank.json Integration
# ---------------------------------------------------------------------------

def test_production_question_bank_initial_selection(production_question_bank):
    # For a general patient with no prior answers, questions like SpO2 or HR have 3.0 * 3.0 * 0.6 = 5.4
    state = {"chief_complaint": "general"}
    selected = select_next_question(state, production_question_bank)

    assert selected is not None
    assert selected["field"] in ["oxygen_saturation", "heart_rate", "respiratory_rate", "systolic_bp"]
    assert selected["selection_score"] >= 5.4
    assert "reason" in selected
    assert "selection_reason" in selected


def test_production_question_bank_exhaustion(production_question_bank):
    # When all 11 fields from the production bank are answered, selector returns None
    all_fields_state = {
        "oxygen_saturation": 98.0,
        "heart_rate": 75,
        "respiratory_rate": 16,
        "systolic_bp": 120,
        "temperature": 37.0,
        "chest_pain": False,
        "pain_radiation": False,
        "shortness_of_breath": False,
        "confusion": False,
        "dizziness": False,
        "age": 45,
        "chief_complaint": "general",
    }
    selected = select_next_question(all_fields_state, production_question_bank)
    assert selected is None
