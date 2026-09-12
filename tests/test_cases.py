"""Pytest coverage for all decide_routing branches + uncertainty + risk scorer.

Branches under test:
  1. EMERGENCY           — risk≥7, no uncertainty/contradictions/missing
  2. ESCALATE via uncertainty  — risk≥7, uncertainty≥0.50
  3. ESCALATE via contradiction — risk≥7, contradiction present
  4. ESCALATE via missing field — risk≥7, missing critical field
  5. PRIMARY_CARE        — 4≤risk≤6
  6. DISCHARGE           — risk<4, uncertainty≤0.25
  7. None (ask more)     — risk<4, uncertainty>0.25
"""
import pytest
from agent.decision_engine import calculate_uncertainty, decide_routing
from tools.risk_scorer import (
    calculate_risk,
    score_age,
    score_heart_rate,
    score_oxygen_saturation,
    score_respiratory_rate,
    score_symptoms,
    score_systolic_bp,
    score_temperature,
)


# ---------------------------------------------------------------------------
# Risk scorer unit tests
# ---------------------------------------------------------------------------

def test_score_spo2_critical():
    assert score_oxygen_saturation(90)[0] == 3

def test_score_spo2_elevated():
    assert score_oxygen_saturation(92)[0] == 2

def test_score_spo2_normal():
    assert score_oxygen_saturation(98)[0] == 0

def test_score_spo2_none():
    assert score_oxygen_saturation(None)[0] == 0

def test_score_hr_high_critical():
    assert score_heart_rate(135)[0] == 3

def test_score_hr_low_critical():
    assert score_heart_rate(38)[0] == 3

def test_score_hr_elevated():
    assert score_heart_rate(120)[0] == 2

def test_score_hr_normal():
    assert score_heart_rate(75)[0] == 0

def test_score_rr_critical():
    assert score_respiratory_rate(26)[0] == 3

def test_score_rr_normal():
    assert score_respiratory_rate(16)[0] == 0

def test_score_sbp_critical():
    assert score_systolic_bp(85)[0] == 3

def test_score_sbp_normal():
    assert score_systolic_bp(120)[0] == 0

def test_score_temp_high():
    assert score_temperature(39.5)[0] == 3

def test_score_temp_low():
    assert score_temperature(34.5)[0] == 3

def test_score_temp_normal():
    assert score_temperature(37.0)[0] == 0

def test_score_symptoms_all():
    state = {"chest_pain": True, "pain_radiation": True, "shortness_of_breath": True, "confusion": True}
    pts, factors = score_symptoms(state)
    assert pts == 9
    assert len(factors) == 4

def test_score_symptoms_none():
    pts, factors = score_symptoms({})
    assert pts == 0
    assert factors == []

def test_score_age_old():
    assert score_age(72)[0] == 2

def test_score_age_middle():
    assert score_age(60)[0] == 1

def test_score_age_young():
    assert score_age(30)[0] == 0

def test_calculate_risk_high():
    state = {
        "oxygen_saturation": 88, "heart_rate": 140, "chest_pain": True,
        "age": 72, "respiratory_rate": 28,
    }
    result = calculate_risk(state)
    assert result["score"] >= 7
    assert result["level"] == "HIGH"

def test_calculate_risk_moderate():
    state = {"oxygen_saturation": 92, "chest_pain": True}
    result = calculate_risk(state)
    assert 4 <= result["score"] <= 6
    assert result["level"] == "MODERATE"

def test_calculate_risk_low():
    state = {"oxygen_saturation": 98, "heart_rate": 70, "age": 30}
    result = calculate_risk(state)
    assert result["score"] < 4
    assert result["level"] == "LOW"


# ---------------------------------------------------------------------------
# Uncertainty tests
# ---------------------------------------------------------------------------

def test_uncertainty_all_missing():
    state = {}  # all 6 critical fields missing
    result = calculate_uncertainty(state)
    assert result["uncertainty"] == 1.0
    assert len(result["missing_fields"]) == 6

def test_uncertainty_none_missing():
    state = {
        "oxygen_saturation": 98, "heart_rate": 75, "respiratory_rate": 16,
        "systolic_bp": 120, "chest_pain": False, "shortness_of_breath": False,
    }
    result = calculate_uncertainty(state)
    assert result["uncertainty"] == 0.0
    assert result["missing_fields"] == []

def test_uncertainty_contradictions_capped():
    state = {
        "oxygen_saturation": 98, "heart_rate": 75, "respiratory_rate": 16,
        "systolic_bp": 120, "chest_pain": False, "shortness_of_breath": False,
        "contradictions": ["c1", "c2", "c3"],  # 3 × 0.15 = 0.45, capped at 0.30
    }
    result = calculate_uncertainty(state)
    assert result["uncertainty"] == 0.30

def test_uncertainty_partial_missing_with_contradiction():
    # 3 missing → 3/6=0.50, 1 contradiction → 0.15; total = 0.65
    state = {
        "oxygen_saturation": 98, "heart_rate": 75, "respiratory_rate": 16,
        "contradictions": ["conflict"],
    }
    result = calculate_uncertainty(state)
    assert abs(result["uncertainty"] - 0.65) < 1e-4


# ---------------------------------------------------------------------------
# Routing branch tests
# ---------------------------------------------------------------------------

def _make_high_risk_state(uncertainty=0.0, contradictions=None, missing=None):
    return {
        "risk_score": 9,
        "uncertainty": uncertainty,
        "contradictions": contradictions or [],
        "missing_critical_fields": missing or [],
    }

# Branch 1: EMERGENCY — high risk, clean data
def test_routing_emergency():
    state = _make_high_risk_state(uncertainty=0.0)
    result = decide_routing(state)
    assert result["routing"] == "EMERGENCY"

# Branch 2: ESCALATE — high risk + high uncertainty
def test_routing_escalate_uncertainty():
    state = _make_high_risk_state(uncertainty=0.55)
    result = decide_routing(state)
    assert result["routing"] == "ESCALATE_TO_HUMAN"
    assert "uncertainty" in result["reason"].lower()

# Branch 3: ESCALATE — high risk + contradiction present
def test_routing_escalate_contradiction():
    state = _make_high_risk_state(uncertainty=0.0, contradictions=["HR conflict"])
    result = decide_routing(state)
    assert result["routing"] == "ESCALATE_TO_HUMAN"
    assert "contradiction" in result["reason"].lower()

# Branch 4: ESCALATE — high risk + missing critical field
def test_routing_escalate_missing_field():
    state = _make_high_risk_state(uncertainty=0.0, missing=["oxygen_saturation"])
    result = decide_routing(state)
    assert result["routing"] == "ESCALATE_TO_HUMAN"
    assert "missing" in result["reason"].lower()

# Branch 5: PRIMARY_CARE — moderate risk
def test_routing_primary_care_low_bound():
    state = {"risk_score": 4, "uncertainty": 0.10, "contradictions": [], "missing_critical_fields": []}
    result = decide_routing(state)
    assert result["routing"] == "PRIMARY_CARE"

def test_routing_primary_care_high_bound():
    state = {"risk_score": 6, "uncertainty": 0.10, "contradictions": [], "missing_critical_fields": []}
    result = decide_routing(state)
    assert result["routing"] == "PRIMARY_CARE"

# Branch 6: DISCHARGE — low risk, low uncertainty
def test_routing_discharge():
    state = {"risk_score": 2, "uncertainty": 0.10, "contradictions": [], "missing_critical_fields": []}
    result = decide_routing(state)
    assert result["routing"] == "DISCHARGE"

# Branch 7: None — low risk but high uncertainty (needs more data)
def test_routing_none_low_risk_high_uncertainty():
    state = {"risk_score": 2, "uncertainty": 0.50, "contradictions": [], "missing_critical_fields": []}
    result = decide_routing(state)
    assert result["routing"] is None

# Edge: risk=3, uncertainty=0.25 → exactly at DISCHARGE boundary
def test_routing_discharge_boundary():
    state = {"risk_score": 3, "uncertainty": 0.25, "contradictions": [], "missing_critical_fields": []}
    result = decide_routing(state)
    assert result["routing"] == "DISCHARGE"

# Edge: risk=3, uncertainty=0.26 → just above boundary → None
def test_routing_none_just_above_boundary():
    state = {"risk_score": 3, "uncertainty": 0.26, "contradictions": [], "missing_critical_fields": []}
    result = decide_routing(state)
    assert result["routing"] is None
