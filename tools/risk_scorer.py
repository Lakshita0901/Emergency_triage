"""Deterministic risk scoring engine for TriageFlow AI.

All thresholds are sourced directly from the TriageFlow AI specification.
NO LLM calls. Pure deterministic Python.
"""
from typing import Any, TypedDict

CRITICAL_FIELDS: list[str] = [
    "oxygen_saturation",
    "heart_rate",
    "respiratory_rate",
    "systolic_bp",
    "chest_pain",
    "shortness_of_breath",
]


class RiskResult(TypedDict):
    score: int
    level: str  # "LOW" | "MODERATE" | "HIGH"
    risk_factors: list[str]


def score_oxygen_saturation(spo2: float | None) -> tuple[int, str | None]:
    """SpO2 ≤91% → +3, 92–93% → +2, else 0."""
    if spo2 is None:
        return 0, None
    if spo2 <= 91:
        return 3, f"SpO2 {spo2}% (≤91% critical)"
    if spo2 <= 93:
        return 2, f"SpO2 {spo2}% (92–93% elevated)"
    return 0, None


def score_heart_rate(hr: int | None) -> tuple[int, str | None]:
    """HR ≥131 or ≤40 → +3, 111–130 → +2, else 0."""
    if hr is None:
        return 0, None
    if hr >= 131 or hr <= 40:
        return 3, f"HR {hr} bpm (≥131 or ≤40 critical)"
    if hr >= 111:
        return 2, f"HR {hr} bpm (111–130 elevated)"
    return 0, None


def score_respiratory_rate(rr: int | None) -> tuple[int, str | None]:
    """RR ≥25 → +3, else 0."""
    if rr is None:
        return 0, None
    if rr >= 25:
        return 3, f"RR {rr} breaths/min (≥25 critical)"
    return 0, None


def score_systolic_bp(sbp: int | None) -> tuple[int, str | None]:
    """Systolic BP ≤90 → +3, else 0."""
    if sbp is None:
        return 0, None
    if sbp <= 90:
        return 3, f"Systolic BP {sbp} mmHg (≤90 critical)"
    return 0, None


def score_temperature(temp: float | None) -> tuple[int, str | None]:
    """Temp ≥39.1°C or ≤35°C → +3, else 0."""
    if temp is None:
        return 0, None
    if temp >= 39.1 or temp <= 35.0:
        return 3, f"Temp {temp}°C (≥39.1 or ≤35 critical)"
    return 0, None


def score_symptoms(state: dict[str, Any]) -> tuple[int, list[str]]:
    """
    Chest pain present → +2
    Pain radiation (arm/jaw/back) → +2
    Shortness of breath → +2
    Confusion → +3
    """
    points = 0
    factors: list[str] = []
    if state.get("chest_pain"):
        points += 2
        factors.append("Chest pain present (+2)")
    if state.get("pain_radiation"):
        points += 2
        factors.append("Pain radiation (arm/jaw/back) (+2)")
    if state.get("shortness_of_breath"):
        points += 2
        factors.append("Shortness of breath (+2)")
    if state.get("confusion"):
        points += 3
        factors.append("Confusion (+3)")
    return points, factors


def score_age(age: int | None) -> tuple[int, str | None]:
    """Age ≥70 → +2, 55–69 → +1, else 0."""
    if age is None:
        return 0, None
    if age >= 70:
        return 2, f"Age {age} (≥70 elevated)"
    if age >= 55:
        return 1, f"Age {age} (55–69 mild)"
    return 0, None


def calculate_risk(state: dict[str, Any]) -> RiskResult:
    """Aggregate all scoring functions into a total risk score and level.

    Returns:
        RiskResult with keys:
            score: int — total risk points
            level: "LOW" (score<4) | "MODERATE" (4–6) | "HIGH" (≥7)
            risk_factors: list of human-readable factor strings
    """
    risk_factors: list[str] = []
    total = 0

    def _add(points: int, factor: str | None) -> None:
        nonlocal total
        if points > 0 and factor:
            total += points
            risk_factors.append(factor)

    p, f = score_oxygen_saturation(state.get("oxygen_saturation"))
    _add(p, f)

    p, f = score_heart_rate(state.get("heart_rate"))
    _add(p, f)

    p, f = score_respiratory_rate(state.get("respiratory_rate"))
    _add(p, f)

    p, f = score_systolic_bp(state.get("systolic_bp"))
    _add(p, f)

    p, f = score_temperature(state.get("temperature"))
    _add(p, f)

    p, f = score_age(state.get("age"))
    _add(p, f)

    sym_points, sym_factors = score_symptoms(state)
    total += sym_points
    risk_factors.extend(sym_factors)

    if total >= 7:
        level = "HIGH"
    elif total >= 4:
        level = "MODERATE"
    else:
        level = "LOW"

    return RiskResult(score=total, level=level, risk_factors=risk_factors)
