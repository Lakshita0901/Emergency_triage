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


class RiskResult(TypedDict, total=False):
    score: int
    level: str  # "LOW" | "MODERATE" | "HIGH"
    risk_factors: list[str]
    invalid_fields: list[str]


PHYSIOLOGICAL_RANGES: dict[str, tuple[float, float, str]] = {
    "oxygen_saturation": (0.0, 100.0, "0–100%"),
    "heart_rate": (0.0, 300.0, "0–300 bpm"),
    "respiratory_rate": (0.0, 80.0, "0–80 breaths/min"),
    "systolic_bp": (0.0, 300.0, "0–300 mmHg"),
    "temperature": (25.0, 45.0, "25–45°C"),
    "age": (0.0, 130.0, "0–130 years"),
}

FIELD_ALIASES: dict[str, str] = {
    "oxygen_saturation": "oxygen_saturation",
    "spo2": "oxygen_saturation",
    "o2_sat": "oxygen_saturation",
    "heart_rate": "heart_rate",
    "hr": "heart_rate",
    "pulse": "heart_rate",
    "respiratory_rate": "respiratory_rate",
    "rr": "respiratory_rate",
    "resp_rate": "respiratory_rate",
    "systolic_bp": "systolic_bp",
    "sbp": "systolic_bp",
    "blood_pressure": "systolic_bp",
    "bp": "systolic_bp",
    "temperature": "temperature",
    "temp": "temperature",
    "age": "age",
}


def validate_vital(field: str, value: Any) -> tuple[bool, str]:
    """Validate whether a value is physiologically plausible for a given vital sign.

    Parameters:
        field: Name of the vital sign or demographic field (e.g. oxygen_saturation,
               heart_rate, respiratory_rate, systolic_bp, temperature).
        value: Numeric measurement value.

    Returns:
        (True, "") if the value is physiologically plausible for that field.
        (False, reason) otherwise.
    """
    if not isinstance(field, str):
        return False, "Field name must be a string."

    norm_field = FIELD_ALIASES.get(field.strip().lower(), field.strip().lower())

    if value is None:
        return False, f"Value for {field} cannot be None."

    if isinstance(value, bool):
        return False, f"Value for {field} must be numeric, not a boolean."

    try:
        val = float(value)
    except (TypeError, ValueError):
        return False, f"Value for {field} must be a valid number, got {value!r}."

    if norm_field not in PHYSIOLOGICAL_RANGES:
        return False, f"Unrecognized vital sign field: {field}."

    min_val, max_val, unit_str = PHYSIOLOGICAL_RANGES[norm_field]
    if not (min_val <= val <= max_val):
        return (
            False,
            f"{field} value {value} is outside plausible physiological range ({unit_str}).",
        )

    return True, ""


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

    Validates physiological plausibility of all vital signs before scoring.
    If a field fails validation, it is excluded from scoring and added to 'invalid_fields'.

    Returns:
        RiskResult with keys:
            score: int — total risk points
            level: "LOW" (score<4) | "MODERATE" (4–6) | "HIGH" (≥7)
            risk_factors: list of human-readable factor strings
            invalid_fields: list of field names that failed physiological validation
    """
    risk_factors: list[str] = []
    invalid_fields: list[str] = []
    total = 0

    # Validate each vital/demographic field present in state
    validated_state = dict(state)
    for field in ("oxygen_saturation", "heart_rate", "respiratory_rate", "systolic_bp", "temperature", "age"):
        val = state.get(field)
        if val is not None:
            is_valid, _reason = validate_vital(field, val)
            if not is_valid:
                invalid_fields.append(field)
                validated_state[field] = None  # Exclude from scoring

    def _add(points: int, factor: str | None) -> None:
        nonlocal total
        if points > 0 and factor:
            total += points
            risk_factors.append(factor)

    p, f = score_oxygen_saturation(validated_state.get("oxygen_saturation"))
    _add(p, f)

    p, f = score_heart_rate(validated_state.get("heart_rate"))
    _add(p, f)

    p, f = score_respiratory_rate(validated_state.get("respiratory_rate"))
    _add(p, f)

    p, f = score_systolic_bp(validated_state.get("systolic_bp"))
    _add(p, f)

    p, f = score_temperature(validated_state.get("temperature"))
    _add(p, f)

    p, f = score_age(validated_state.get("age"))
    _add(p, f)

    sym_points, sym_factors = score_symptoms(validated_state)
    total += sym_points
    risk_factors.extend(sym_factors)

    if total >= 7:
        level = "HIGH"
    elif total >= 4:
        level = "MODERATE"
    else:
        level = "LOW"

    return RiskResult(
        score=total,
        level=level,
        risk_factors=risk_factors,
        invalid_fields=invalid_fields,
    )
