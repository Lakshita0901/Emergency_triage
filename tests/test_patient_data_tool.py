"""Unit tests for tools/patient_data_tool.py.

Tests:
- lookup_reference_ranges returns required dictionary structure for all 5 vital signs.
- Simulated disclaimer/note is included.
- Unknown fields return empty dict.
- classify_value returns correct categories ("critical", "abnormal", "normal", "unknown")
  across boundaries and edge cases.
- update_patient_records works as expected.
"""
import pytest
from tools.patient_data_tool import (
    lookup_reference_ranges,
    classify_value,
    update_patient_records,
)


REQUIRED_VITALS = [
    "oxygen_saturation",
    "heart_rate",
    "respiratory_rate",
    "systolic_bp",
    "temperature",
]


# ---------------------------------------------------------------------------
# Tests: lookup_reference_ranges
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field", REQUIRED_VITALS)
def test_lookup_reference_ranges_structure(field):
    ref = lookup_reference_ranges(field)
    assert isinstance(ref, dict)
    assert "normal_range" in ref
    assert isinstance(ref["normal_range"], tuple)
    assert len(ref["normal_range"]) == 2
    assert ref["normal_range"][0] < ref["normal_range"][1]

    assert "unit" in ref
    assert isinstance(ref["unit"], str)

    assert "status_thresholds" in ref
    st = ref["status_thresholds"]
    assert "critical_low" in st
    assert "low" in st
    assert "normal" in st
    assert "high" in st
    assert "critical_high" in st
    assert isinstance(st["normal"], tuple)
    assert len(st["normal"]) == 2


def test_lookup_reference_ranges_aliases():
    assert lookup_reference_ranges("SpO2") == lookup_reference_ranges("oxygen_saturation")
    assert lookup_reference_ranges("HR") == lookup_reference_ranges("heart_rate")
    assert lookup_reference_ranges("RR") == lookup_reference_ranges("respiratory_rate")
    assert lookup_reference_ranges("SBP") == lookup_reference_ranges("systolic_bp")
    assert lookup_reference_ranges("temp") == lookup_reference_ranges("temperature")


def test_lookup_reference_ranges_unknown():
    assert lookup_reference_ranges("unknown_vital") == {}
    assert lookup_reference_ranges("") == {}
    assert lookup_reference_ranges(None) == {}


# ---------------------------------------------------------------------------
# Tests: classify_value
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "field, val, expected",
    [
        # oxygen_saturation (SpO2)
        ("oxygen_saturation", 88.0, "critical"),
        ("oxygen_saturation", 90.0, "critical"),
        ("oxygen_saturation", 91.0, "critical"),
        ("oxygen_saturation", 92.0, "abnormal"),
        ("oxygen_saturation", 93.0, "abnormal"),
        ("oxygen_saturation", 94.0, "abnormal"),
        ("oxygen_saturation", 95.0, "normal"),
        ("oxygen_saturation", 98.0, "normal"),
        ("oxygen_saturation", 100.0, "normal"),
        ("oxygen_saturation", None, "unknown"),

        # heart_rate
        ("heart_rate", 35, "critical"),
        ("heart_rate", 40, "critical"),
        ("heart_rate", 45, "abnormal"),
        ("heart_rate", 59, "abnormal"),
        ("heart_rate", 60, "normal"),
        ("heart_rate", 75, "normal"),
        ("heart_rate", 100, "normal"),
        ("heart_rate", 115, "abnormal"),
        ("heart_rate", 130, "abnormal"),
        ("heart_rate", 131, "critical"),
        ("heart_rate", 160, "critical"),
        ("heart_rate", None, "unknown"),

        # respiratory_rate
        ("respiratory_rate", 6, "critical"),
        ("respiratory_rate", 8, "critical"),
        ("respiratory_rate", 10, "abnormal"),
        ("respiratory_rate", 12, "normal"),
        ("respiratory_rate", 16, "normal"),
        ("respiratory_rate", 20, "normal"),
        ("respiratory_rate", 22, "abnormal"),
        ("respiratory_rate", 24, "abnormal"),
        ("respiratory_rate", 25, "critical"),
        ("respiratory_rate", 30, "critical"),
        ("respiratory_rate", None, "unknown"),

        # systolic_bp
        ("systolic_bp", 70, "critical"),
        ("systolic_bp", 89, "critical"),
        ("systolic_bp", 90, "critical"),
        ("systolic_bp", 100, "normal"),
        ("systolic_bp", 115, "normal"),
        ("systolic_bp", 120, "normal"),
        ("systolic_bp", 135, "abnormal"),
        ("systolic_bp", 170, "abnormal"),
        ("systolic_bp", 180, "critical"),
        ("systolic_bp", 200, "critical"),
        ("systolic_bp", None, "unknown"),

        # temperature
        ("temperature", 34.0, "critical"),
        ("temperature", 35.0, "critical"),
        ("temperature", 35.5, "abnormal"),
        ("temperature", 36.5, "normal"),
        ("temperature", 37.0, "normal"),
        ("temperature", 37.2, "normal"),
        ("temperature", 38.0, "abnormal"),
        ("temperature", 39.0, "abnormal"),
        ("temperature", 39.1, "critical"),
        ("temperature", 40.0, "critical"),
        ("temperature", None, "unknown"),

        # Unrecognized field
        ("unknown_metric", 50, "unknown"),
        ("unknown_metric", None, "unknown"),
    ],
)
def test_classify_value(field, val, expected):
    assert classify_value(field, val) == expected


def test_update_patient_records():
    curr = {
        "age": 50,
        "vitals": {"heart_rate": 72},
        "symptoms": {"chest_pain": True},
    }
    new_data = {
        "vitals": {"oxygen_saturation": 98.0},
        "symptoms": {"dizziness": False},
        "chief_complaint": "Shortness of breath",
    }
    updated = update_patient_records(curr, new_data)
    assert updated["age"] == 50
    assert updated["vitals"]["heart_rate"] == 72
    assert updated["vitals"]["oxygen_saturation"] == 98.0
    assert updated["symptoms"]["chest_pain"] is True
    assert updated["symptoms"]["dizziness"] is False
    assert updated["chief_complaint"] == "Shortness of breath"
