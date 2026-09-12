"""Comprehensive unit tests for the deterministic risk scorer engine.

Covers:
- Boundary conditions for each individual score_* function (just below, at, just above each cutoff).
- Missing/None values for each scorer.
- calculate_risk boundary levels (LOW < 4, MODERATE 4-6, HIGH >= 7).
- calculate_risk end-to-end evaluation on the 3 synthetic cases from data/synthetic_cases.json.
"""
import json
from pathlib import Path
import pytest

from tools.risk_scorer import (
    score_oxygen_saturation,
    score_heart_rate,
    score_respiratory_rate,
    score_systolic_bp,
    score_temperature,
    score_symptoms,
    score_age,
    calculate_risk,
)


# ---------------------------------------------------------------------------
# Fixture: Load Synthetic Cases
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_cases():
    data_path = Path(__file__).resolve().parent.parent / "data" / "synthetic_cases.json"
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)["cases"]


# ---------------------------------------------------------------------------
# 1. score_oxygen_saturation Threshold Boundaries
# Rule: SpO2 <= 91% -> +3, 92-93% -> +2, >93% -> 0
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "spo2, expected_points, expect_factor",
    [
        (90.0, 3, True),   # Below 91
        (90.9, 3, True),   # Just below 91 cutoff
        (91.0, 3, True),   # At 91 cutoff
        (91.1, 2, True),   # Just above 91 cutoff (in 92-93 range)
        (92.0, 2, True),   # In 92-93 range
        (92.9, 2, True),   # Just below 93 cutoff
        (93.0, 2, True),   # At 93 cutoff
        (93.1, 0, False),  # Just above 93 cutoff (normal)
        (94.0, 0, False),  # Above 93
        (99.0, 0, False),  # Normal
        (None, 0, False),  # Missing value
    ],
)
def test_score_oxygen_saturation_boundaries(spo2, expected_points, expect_factor):
    pts, factor = score_oxygen_saturation(spo2)
    assert pts == expected_points
    if expect_factor:
        assert factor is not None
        assert "SpO2" in factor
    else:
        assert factor is None


# ---------------------------------------------------------------------------
# 2. score_heart_rate Threshold Boundaries
# Rule: HR >= 131 or HR <= 40 -> +3, 111-130 -> +2, 41-110 -> 0
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "hr, expected_points, expect_factor",
    [
        (39, 3, True),    # Just below 40 cutoff (severe bradycardia)
        (40, 3, True),    # At 40 cutoff
        (41, 0, False),   # Just above 40 cutoff (normal range)
        (75, 0, False),   # Normal
        (110, 0, False),  # Just below 111 cutoff (normal upper bound)
        (111, 2, True),   # At 111 cutoff (elevated tachycardia)
        (112, 2, True),   # Just above 111 cutoff
        (120, 2, True),   # Middle of 111-130
        (129, 2, True),   # Just below 130 cutoff
        (130, 2, True),   # At 130 cutoff
        (131, 3, True),   # At 131 cutoff (severe tachycardia)
        (132, 3, True),   # Just above 131 cutoff
        (180, 3, True),   # Extreme tachycardia
        (None, 0, False), # Missing value
    ],
)
def test_score_heart_rate_boundaries(hr, expected_points, expect_factor):
    pts, factor = score_heart_rate(hr)
    assert pts == expected_points
    if expect_factor:
        assert factor is not None
        assert "HR" in factor
    else:
        assert factor is None


# ---------------------------------------------------------------------------
# 3. score_respiratory_rate Threshold Boundaries
# Rule: RR >= 25 -> +3, < 25 -> 0
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "rr, expected_points, expect_factor",
    [
        (16, 0, False),   # Normal
        (24, 0, False),   # Just below 25 cutoff
        (25, 3, True),    # At 25 cutoff (tachypnea)
        (26, 3, True),    # Just above 25 cutoff
        (40, 3, True),    # Severe tachypnea
        (None, 0, False), # Missing value
    ],
)
def test_score_respiratory_rate_boundaries(rr, expected_points, expect_factor):
    pts, factor = score_respiratory_rate(rr)
    assert pts == expected_points
    if expect_factor:
        assert factor is not None
        assert "RR" in factor
    else:
        assert factor is None


# ---------------------------------------------------------------------------
# 4. score_systolic_bp Threshold Boundaries
# Rule: Systolic BP <= 90 -> +3, > 90 -> 0
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "sbp, expected_points, expect_factor",
    [
        (70, 3, True),    # Severe hypotension
        (89, 3, True),    # Just below 90 cutoff
        (90, 3, True),    # At 90 cutoff
        (91, 0, False),   # Just above 90 cutoff
        (120, 0, False),  # Normal
        (180, 0, False),  # High BP (not scored as hypotension)
        (None, 0, False), # Missing value
    ],
)
def test_score_systolic_bp_boundaries(sbp, expected_points, expect_factor):
    pts, factor = score_systolic_bp(sbp)
    assert pts == expected_points
    if expect_factor:
        assert factor is not None
        assert "Systolic BP" in factor
    else:
        assert factor is None


# ---------------------------------------------------------------------------
# 5. score_temperature Threshold Boundaries
# Rule: Temp >= 39.1 or Temp <= 35.0 -> +3, else 0
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "temp, expected_points, expect_factor",
    [
        (34.0, 3, True),   # Severe hypothermia
        (34.9, 3, True),   # Just below 35.0 cutoff
        (35.0, 3, True),   # At 35.0 cutoff
        (35.1, 0, False),  # Just above 35.0 cutoff (normal lower bound)
        (37.0, 0, False),  # Normal body temperature
        (39.0, 0, False),  # Just below 39.1 cutoff (high normal / mild)
        (39.1, 3, True),   # At 39.1 cutoff (hyperthermia)
        (39.2, 3, True),   # Just above 39.1 cutoff
        (40.5, 3, True),   # Severe fever
        (None, 0, False),  # Missing value
    ],
)
def test_score_temperature_boundaries(temp, expected_points, expect_factor):
    pts, factor = score_temperature(temp)
    assert pts == expected_points
    if expect_factor:
        assert factor is not None
        assert "Temp" in factor
    else:
        assert factor is None


# ---------------------------------------------------------------------------
# 6. score_symptoms Combinations & Individual Points
# Rule: chest_pain +2, pain_radiation +2, shortness_of_breath +2, confusion +3
# ---------------------------------------------------------------------------

def test_score_symptoms_empty_or_false():
    # Empty dictionary
    pts, factors = score_symptoms({})
    assert pts == 0
    assert factors == []

    # All explicitly False
    state_false = {
        "chest_pain": False,
        "pain_radiation": False,
        "shortness_of_breath": False,
        "confusion": False,
    }
    pts, factors = score_symptoms(state_false)
    assert pts == 0
    assert factors == []


@pytest.mark.parametrize(
    "symptom_key, expected_pts, expected_label",
    [
        ("chest_pain", 2, "Chest pain present"),
        ("pain_radiation", 2, "Pain radiation (arm/jaw/back)"),
        ("shortness_of_breath", 2, "Shortness of breath"),
        ("confusion", 3, "Confusion"),
    ],
)
def test_score_symptoms_individual(symptom_key, expected_pts, expected_label):
    state = {symptom_key: True}
    pts, factors = score_symptoms(state)
    assert pts == expected_pts
    assert len(factors) == 1
    assert expected_label in factors[0]


def test_score_symptoms_all_present():
    state = {
        "chest_pain": True,
        "pain_radiation": True,
        "shortness_of_breath": True,
        "confusion": True,
    }
    pts, factors = score_symptoms(state)
    assert pts == 9  # 2 + 2 + 2 + 3
    assert len(factors) == 4


# ---------------------------------------------------------------------------
# 7. score_age Threshold Boundaries
# Rule: Age >= 70 -> +2, 55-69 -> +1, < 55 -> 0
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "age, expected_points, expect_factor",
    [
        (18, 0, False),   # Young adult
        (54, 0, False),   # Just below 55 cutoff
        (55, 1, True),    # At 55 cutoff
        (56, 1, True),    # Just above 55 cutoff
        (68, 1, True),    # Within 55-69
        (69, 1, True),    # Just below 70 cutoff
        (70, 2, True),    # At 70 cutoff
        (71, 2, True),    # Just above 70 cutoff
        (90, 2, True),    # Advanced age
        (None, 0, False), # Missing value
    ],
)
def test_score_age_boundaries(age, expected_points, expect_factor):
    pts, factor = score_age(age)
    assert pts == expected_points
    if expect_factor:
        assert factor is not None
        assert "Age" in factor
    else:
        assert factor is None


# ---------------------------------------------------------------------------
# 8. calculate_risk Score Cutoff Boundaries
# Rule: score < 4 -> LOW, 4 <= score <= 6 -> MODERATE, score >= 7 -> HIGH
# ---------------------------------------------------------------------------

def test_calculate_risk_boundary_low():
    # Exactly 3 points (SpO2=91 -> +3) -> score < 4 -> LOW
    state = {"oxygen_saturation": 91.0}
    res = calculate_risk(state)
    assert res["score"] == 3
    assert res["level"] == "LOW"


def test_calculate_risk_boundary_moderate_lower():
    # Exactly 4 points (chest_pain=True (+2) + pain_radiation=True (+2)) -> score 4 -> MODERATE
    state = {"chest_pain": True, "pain_radiation": True}
    res = calculate_risk(state)
    assert res["score"] == 4
    assert res["level"] == "MODERATE"


def test_calculate_risk_boundary_moderate_upper():
    # Exactly 6 points (chest_pain=True (+2) + shortness_of_breath=True (+2) + age=70 (+2)) -> score 6 -> MODERATE
    state = {"chest_pain": True, "shortness_of_breath": True, "age": 70}
    res = calculate_risk(state)
    assert res["score"] == 6
    assert res["level"] == "MODERATE"


def test_calculate_risk_boundary_high():
    # Exactly 7 points (SpO2=91 (+3) + SBP=90 (+3) + age=55 (+1)) -> score 7 -> HIGH
    state = {"oxygen_saturation": 91.0, "systolic_bp": 90, "age": 55}
    res = calculate_risk(state)
    assert res["score"] == 7
    assert res["level"] == "HIGH"


# ---------------------------------------------------------------------------
# 9. calculate_risk End-to-End on the 3 Synthetic Cases
# Asserts risk level matches the clinical risk category for the expected route:
# - high_risk_chest_pain -> EMERGENCY -> HIGH
# - contradictory_breathing_case -> ESCALATE_TO_HUMAN -> HIGH (at presenting state)
# - low_risk_muscle_pain -> DISCHARGE -> LOW
# ---------------------------------------------------------------------------

def test_calculate_risk_case_high_risk_chest_pain(synthetic_cases):
    case = next(c for c in synthetic_cases if c["id"] == "high_risk_chest_pain")
    state = {"age": case["demographics"]["age"], "chief_complaint": case.get("chief_complaint")}
    for step in sorted(case["steps"], key=lambda s: s["order"]):
        state[step["field"]] = step["value"]

    result = calculate_risk(state)
    # 68yo (1) + chest_pain (2) + radiation (2) + SOB (2) + SpO2 90 (3) + HR 118 (2) + RR 26 (3) + SBP 88 (3) = 18
    assert result["score"] == 18
    assert result["level"] == "HIGH"
    assert case["expected_routing"] == "EMERGENCY"
    # Verify expected clinical factors captured
    assert len(result["risk_factors"]) >= 7


def test_calculate_risk_case_contradictory_breathing(synthetic_cases):
    case = next(c for c in synthetic_cases if c["id"] == "contradictory_breathing_case")
    
    # State with initial presenting symptoms (steps 1..6 prior to the contradictory denial):
    # 55yo (1) + SOB (2) + SpO2 91 (3) + HR 105 (0) + RR 22 (0) + SBP 110 (0) + chest_pain (2) = 8
    presenting_state = {"age": case["demographics"]["age"], "chief_complaint": case.get("chief_complaint")}
    for step in sorted(case["steps"], key=lambda s: s["order"]):
        if step["order"] < 7:
            presenting_state[step["field"]] = step["value"]

    result_presenting = calculate_risk(presenting_state)
    assert result_presenting["score"] == 8
    assert result_presenting["level"] == "HIGH"
    assert case["expected_routing"] == "ESCALATE_TO_HUMAN"

    # If step 7 is applied (shortness_of_breath denied -> False):
    full_state = dict(presenting_state)
    full_state["shortness_of_breath"] = False
    result_full = calculate_risk(full_state)
    assert result_full["score"] == 6
    assert result_full["level"] == "MODERATE"


def test_calculate_risk_case_low_risk_muscle_pain(synthetic_cases):
    case = next(c for c in synthetic_cases if c["id"] == "low_risk_muscle_pain")
    state = {"age": case["demographics"]["age"], "chief_complaint": case.get("chief_complaint")}
    for step in sorted(case["steps"], key=lambda s: s["order"]):
        state[step["field"]] = step["value"]

    result = calculate_risk(state)
    # 28yo (0) + chest_pain (2) + all vitals normal + no SOB/radiation/confusion = 2
    assert result["score"] == 2
    assert result["level"] == "LOW"
    assert case["expected_routing"] == "DISCHARGE"
    assert len(result["risk_factors"]) == 1
    assert "Chest pain present" in result["risk_factors"][0]


@pytest.mark.parametrize(
    "case_id, expected_routing, expected_risk_level",
    [
        ("high_risk_chest_pain", "EMERGENCY", "HIGH"),
        ("contradictory_breathing_case", "ESCALATE_TO_HUMAN", "HIGH"),
        ("low_risk_muscle_pain", "DISCHARGE", "LOW"),
    ],
)
def test_calculate_risk_all_synthetic_cases_parameterized(
    synthetic_cases, case_id, expected_routing, expected_risk_level
):
    case = next(c for c in synthetic_cases if c["id"] == case_id)
    assert case["expected_routing"] == expected_routing

    state = {"age": case["demographics"]["age"], "chief_complaint": case.get("chief_complaint")}
    # Build state representing the patient's positive clinical findings
    for step in sorted(case["steps"], key=lambda s: s["order"]):
        # For contradiction case, capture the presenting positive finding
        if case_id == "contradictory_breathing_case" and step["order"] == 7:
            continue
        state[step["field"]] = step["value"]

    result = calculate_risk(state)
    assert result["level"] == expected_risk_level
