"""Patient data handling and clinical reference range tool.

Merge, normalize, update patient clinical records, and classify vital signs
against adult reference ranges for simulated triage.

NOTE: All vital sign reference ranges and status classifications in this module
are SIMULATED REFERENCE VALUES for algorithmic prototyping and demonstration only.
They DO NOT constitute medical advice or clinical guidance.
"""
import copy
from typing import Any, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Simulated Adult Vital-Sign Reference Ranges & Thresholds
# NOT clinical guidance — software simulation only.
# ---------------------------------------------------------------------------
REFERENCE_RANGES: Dict[str, Dict[str, Any]] = {
    "oxygen_saturation": {
        "normal_range": (95.0, 100.0),
        "unit": "%",
        "status_thresholds": {
            "critical_low": 91.0,
            "low": 94.0,
            "normal": (95.0, 100.0),
            "high": 100.0,
            "critical_high": None,
        },
        "disclaimer": "Simulated adult reference range (SpO2). Not clinical guidance.",
    },
    "heart_rate": {
        "normal_range": (60.0, 100.0),
        "unit": "bpm",
        "status_thresholds": {
            "critical_low": 40.0,
            "low": 59.0,
            "normal": (60.0, 100.0),
            "high": 101.0,
            "critical_high": 131.0,
        },
        "disclaimer": "Simulated adult reference range (resting HR). Not clinical guidance.",
    },
    "respiratory_rate": {
        "normal_range": (12.0, 20.0),
        "unit": "/min",
        "status_thresholds": {
            "critical_low": 8.0,
            "low": 11.0,
            "normal": (12.0, 20.0),
            "high": 21.0,
            "critical_high": 25.0,
        },
        "disclaimer": "Simulated adult reference range (RR). Not clinical guidance.",
    },
    "systolic_bp": {
        "normal_range": (90.0, 120.0),
        "unit": "mmHg",
        "status_thresholds": {
            "critical_low": 90.0,
            "low": 99.0,
            "normal": (91.0, 120.0),
            "high": 140.0,
            "critical_high": 180.0,
        },
        "disclaimer": "Simulated adult reference range (systolic BP). Not clinical guidance.",
    },
    "temperature": {
        "normal_range": (36.1, 37.2),
        "unit": "°C",
        "status_thresholds": {
            "critical_low": 35.0,
            "low": 36.0,
            "normal": (36.1, 37.2),
            "high": 37.6,
            "critical_high": 39.1,
        },
        "disclaimer": "Simulated adult reference range (body temp °C). Not clinical guidance.",
    },
}

# Aliases for canonical fields
FIELD_ALIASES: Dict[str, str] = {
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
}


def lookup_reference_ranges(field: str) -> dict:
    """Look up standard adult vital-sign reference ranges and status thresholds.

    Parameters:
        field: Name of the vital sign field (e.g., 'oxygen_saturation', 'heart_rate',
               'respiratory_rate', 'systolic_bp', 'temperature').

    Returns:
        A dictionary containing:
            - 'normal_range': (low, high)
            - 'unit': string
            - 'status_thresholds': {
                'critical_low': x,
                'low': x,
                'normal': (x, x),
                'high': x,
                'critical_high': x
              }
        Returns an empty dict if the field has no reference range.

    Note:
        These values are simulated reference values for software demonstration
        and prototype testing, not clinical guidance.
    """
    if not isinstance(field, str):
        return {}

    normalized = FIELD_ALIASES.get(field.strip().lower(), field.strip().lower())
    if normalized in REFERENCE_RANGES:
        return copy.deepcopy(REFERENCE_RANGES[normalized])
    return {}


def classify_value(field: str, value: Optional[float]) -> str:
    """Classify a vital-sign value into 'critical', 'abnormal', 'normal', or 'unknown'.

    Parameters:
        field: Vital sign field name.
        value: Numeric value of the vital measurement (or None).

    Returns:
        One of 'critical', 'abnormal', 'normal', or 'unknown'.
        Returns 'unknown' if value is None, unparseable, or field has no reference range.
    """
    if value is None:
        return "unknown"

    try:
        val = float(value)
    except (TypeError, ValueError):
        return "unknown"

    ref = lookup_reference_ranges(field)
    if not ref or "status_thresholds" not in ref:
        return "unknown"

    thresholds = ref["status_thresholds"]
    crit_low = thresholds.get("critical_low")
    crit_high = thresholds.get("critical_high")
    normal_range = thresholds.get("normal") or ref.get("normal_range")

    # 1. Critical boundaries
    if crit_low is not None and val <= crit_low:
        return "critical"
    if crit_high is not None and val >= crit_high:
        return "critical"

    # 2. Normal range
    if normal_range is not None and len(normal_range) == 2:
        norm_low, norm_high = normal_range
        if norm_low <= val <= norm_high:
            return "normal"

    # 3. Non-critical abnormal (outside normal range)
    return "abnormal"


def update_patient_records(
    current_data: Dict[str, Any], new_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge and normalize patient record updates.

    Performs a clean dictionary merge preserving nested structures where appropriate.
    """
    merged = dict(current_data)
    for k, v in new_data.items():
        if k in ("vitals", "symptoms") and isinstance(v, dict) and isinstance(merged.get(k), dict):
            nested = dict(merged[k])
            nested.update(v)
            merged[k] = nested
        else:
            merged[k] = v
    return merged
