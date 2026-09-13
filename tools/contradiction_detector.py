"""Contradiction detector and audit logging for patient-triage state machine.

Pure Python, stdlib only, dependency-free, and fully type-hinted.
"""
from datetime import datetime, timezone
from typing import Any, Dict, Optional, TypedDict

BOOLEAN_SYMPTOM_FIELDS = {
    "chest_pain",
    "shortness_of_breath",
    "dizziness",
    "confusion",
}


class ContradictionDetail(TypedDict):
    field: str
    old_value: bool
    new_value: bool
    severity: str


class AuditEvent(TypedDict):
    event_type: str
    timestamp: str
    payload: Dict[str, Any]


def detect_contradiction(
    state: Dict[str, Any], field: str, new_value: Any
) -> Optional[ContradictionDetail]:
    """Detect contradiction for boolean symptom fields.

    Only flags a contradiction for boolean symptom fields:
    ('chest_pain', 'shortness_of_breath', 'dizziness', 'confusion')
    where the new value differs from the previously recorded value for that field.

    Returns:
        {"field": field, "old_value": ..., "new_value": ..., "severity": "high"}
        or None if no prior value exists or values match.
    """
    if field not in BOOLEAN_SYMPTOM_FIELDS:
        return None

    # Retrieve prior value from state (or state["symptoms"] if nested)
    prior_value = state.get(field)
    if prior_value is None and isinstance(state.get("symptoms"), dict):
        prior_value = state["symptoms"].get(field)

    if prior_value is None:
        return None

    # Cast / compare boolean values if both are present
    if bool(prior_value) != bool(new_value):
        return {
            "field": field,
            "old_value": bool(prior_value),
            "new_value": bool(new_value),
            "severity": "high",
        }

    return None


def append_audit_event(
    state: Dict[str, Any], event_type: str, payload: Dict[str, Any]
) -> Dict[str, Any]:
    """Append a structured event dict to state['audit_log'] with an ISO timestamp.

    Events include:
    - 'contradiction_detected'
    - 'risk_recalculated'
    - 'question_asked'

    Returns the updated state.
    """
    if "audit_log" not in state or not isinstance(state["audit_log"], list):
        state["audit_log"] = []

    timestamp = datetime.now(timezone.utc).isoformat()
    event: Dict[str, Any] = {
        "event_type": event_type,
        "timestamp": timestamp,
        "payload": payload,
    }
    if isinstance(payload, dict):
        for k, v in payload.items():
            if k not in event:
                event[k] = v

    state["audit_log"].append(event)
    return state
