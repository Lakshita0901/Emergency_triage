"""Adaptive question selector for TriageFlow AI.

Pure Python, deterministic question prioritization.
"""
from typing import Any, Dict, List, Optional, Tuple, Union


def is_contextually_relevant(state: Dict[str, Any], question: Dict[str, Any]) -> Tuple[bool, bool]:
    """Check if question is contextually relevant to patient state.

    Returns:
        (is_applicable, is_direct_match):
        - is_applicable: whether the question overlaps with patient context (or is general)
        - is_direct_match: True if directly relevant to chief complaint, False otherwise
    """
    applicable_to: List[str] = [item.lower() for item in question.get("applicable_to", [])]

    # "general" is always relevant
    if "general" in applicable_to:
        # Check if chief complaint specifically matches as well
        chief_complaint = (state.get("chief_complaint") or "").lower()
        is_direct = bool(chief_complaint and chief_complaint in applicable_to)
        return True, is_direct

    chief_complaint = (state.get("chief_complaint") or "").lower()
    is_direct_match = bool(chief_complaint and chief_complaint in applicable_to)

    # Gather known positive symptoms or present conditions
    known_symptoms: set[str] = set()
    symptoms_dict = state.get("symptoms")
    if isinstance(symptoms_dict, dict):
        for k, v in symptoms_dict.items():
            if v:
                known_symptoms.add(k.lower())

    for k in ["chest_pain", "shortness_of_breath", "pain_radiation", "confusion", "dizziness"]:
        if state.get(k):
            known_symptoms.add(k.lower())

    overlaps_symptoms = any(app in known_symptoms for app in applicable_to)
    is_applicable = is_direct_match or overlaps_symptoms

    return is_applicable, is_direct_match


def select_next_question(
    state: Dict[str, Any],
    question_bank: Union[List[Dict[str, Any]], Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Select the next best question to ask the patient.

    Logic:
    - skip a question if state already has a non-null value for question["field"]
    - skip a question if question["applicable_to"] doesn't overlap with the patient's
      chief_complaint category or already-known symptoms ("general" is always relevant)
    - score = risk_impact * uncertainty_reduction * context_relevance
      (context_relevance = 1.0 if directly relevant to chief complaint, 0.6 otherwise)
    - return the highest-scoring question with a 'reason' field, or None if no candidates remain.
    """
    if isinstance(question_bank, dict):
        questions: List[Dict[str, Any]] = question_bank.get("questions", [])
    elif isinstance(question_bank, list):
        questions = question_bank
    else:
        return None

    best_candidate: Optional[Dict[str, Any]] = None
    best_score: float = -1.0
    best_metadata: Dict[str, Any] = {}

    for q in questions:
        field = q.get("field")
        if not field:
            continue

        # Skip if state already has a non-null value for question["field"]
        val = state.get(field)
        if val is None and isinstance(state.get("vitals"), dict):
            val = state["vitals"].get(field)
        if val is None and isinstance(state.get("symptoms"), dict):
            val = state["symptoms"].get(field)
        if val is not None:
            continue

        # Check contextual relevance
        is_applicable, is_direct = is_contextually_relevant(state, q)
        if not is_applicable:
            continue

        context_relevance = 1.0 if is_direct else 0.6
        risk_impact = float(q.get("risk_impact", 1.0))
        uncertainty_reduction = float(q.get("uncertainty_reduction", 1.0))

        score = risk_impact * uncertainty_reduction * context_relevance

        if score > best_score:
            best_score = score
            best_candidate = q
            best_metadata = {
                "field": field,
                "score": score,
                "risk_impact": risk_impact,
                "uncertainty_reduction": uncertainty_reduction,
                "is_direct": is_direct,
            }

    if not best_candidate:
        return None

    # Build human-readable reason string
    field_name = best_metadata.get("field", "vital")
    is_direct = best_metadata.get("is_direct", False)
    relevance_desc = "directly related to chief complaint" if is_direct else "contextually relevant"
    reason = (
        f"Selected because: highest expected risk impact ({best_metadata.get('risk_impact')}) "
        f"and critical missing vital/symptom '{field_name}' ({relevance_desc})."
    )

    result = dict(best_candidate)
    result["selection_reason"] = reason
    result["reason"] = reason
    result["selection_score"] = round(best_score, 4)

    return result
