"""Patient state TypedDict definition."""
from typing import TypedDict, Optional, List, Dict, Any, Literal

RoutingDecision = Literal["DISCHARGE", "PRIMARY_CARE", "EMERGENCY", "ESCALATE_TO_HUMAN"]

class PatientVitals(TypedDict, total=False):
    oxygen_saturation: Optional[float]
    heart_rate: Optional[int]
    respiratory_rate: Optional[int]
    systolic_bp: Optional[int]
    temperature: Optional[float]
    age: Optional[int]

class PatientSymptoms(TypedDict, total=False):
    chest_pain: Optional[bool]
    pain_radiation: Optional[bool]
    shortness_of_breath: Optional[bool]
    confusion: Optional[bool]
    dizziness: Optional[bool]

class PatientState(TypedDict, total=False):
    # Demographics & Clinical Information
    age: Optional[int]
    chief_complaint: Optional[str]
    vitals: PatientVitals
    symptoms: PatientSymptoms
    
    # Specific fields mapped for convenience/triage calculation
    oxygen_saturation: Optional[float]
    heart_rate: Optional[int]
    respiratory_rate: Optional[int]
    systolic_bp: Optional[int]
    temperature: Optional[float]
    chest_pain: Optional[bool]
    pain_radiation: Optional[bool]
    shortness_of_breath: Optional[bool]
    confusion: Optional[bool]
    dizziness: Optional[bool]

    # Conversation & History
    messages: List[Dict[str, Any]]
    asked_questions: List[str]
    current_question: Optional[Dict[str, Any]]
    new_answer: Optional[Dict[str, Any]]  # e.g., {"field": ..., "value": ...}

    # Metrics & Routing
    risk_score: int
    risk_level: str
    risk_factors: List[str]
    invalid_fields: Optional[List[str]]
    last_validation_error: Optional[str]
    uncertainty: float
    contradictions: List[str]
    contradiction_details: List[Dict[str, Any]]
    audit_log: List[Dict[str, Any]]
    missing_critical_fields: List[str]
    routing_decision: Optional[RoutingDecision]
    routing_reason: Optional[str]
    rationale: Optional[str]
    turn_count: int
    is_complete: bool
