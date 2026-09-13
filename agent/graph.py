"""LangGraph workflow for TriageFlow AI.

Nodes:
  - assess: Computes deterministic risk score and uncertainty.
  - select_question: Picks the highest-impact question if not yet decided.
  - receive_answer: Ingests new user answer for the current question/cycle.
  - update_state: Updates the patient state with the answered value.
  - detect_contradictions: Checks for contradictions and records audit events.
  - decide: Finalizes routing when a decision condition is met.

Topology:
  START -> assess -> [decide -> END] OR [select_question -> receive_answer -> update_state -> detect_contradictions -> assess]

Exposes `run_one_cycle(state, new_answer=None)` for step-by-step Streamlit execution.
"""
from typing import Any, Dict, List, Optional
from langgraph.graph import StateGraph, START, END

from agent.state import PatientState, RoutingDecision
from tools.risk_scorer import calculate_risk, validate_vital
from tools.contradiction_detector import detect_contradiction, append_audit_event
from agent.decision_engine import calculate_uncertainty, decide_routing
from agent.question_selector import select_next_question


# ---------------------------------------------------------------------------
# Node Implementations
# ---------------------------------------------------------------------------

def assess_node(state: PatientState) -> Dict[str, Any]:
    """Assess patient state: compute risk score, uncertainty, and potential routing."""
    # Calculate deterministic risk
    risk_result = calculate_risk(state)
    risk_score = risk_result["score"]
    risk_level = risk_result["level"]
    risk_factors = risk_result["risk_factors"]
    invalid_fields = risk_result.get("invalid_fields", [])

    # Calculate deterministic uncertainty and missing critical fields
    unc_result = calculate_uncertainty(state)
    uncertainty = unc_result["uncertainty"]
    missing_fields = unc_result["missing_fields"]

    temp_state: Dict[str, Any] = {
        **state,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_factors": risk_factors,
        "invalid_fields": invalid_fields,
        "uncertainty": uncertainty,
        "missing_critical_fields": missing_fields,
    }

    # Evaluate routing rules
    routing_result = decide_routing(temp_state)
    routing_decision = routing_result.get("routing")
    routing_reason = routing_result.get("reason", "")

    updates: Dict[str, Any] = {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_factors": risk_factors,
        "invalid_fields": invalid_fields,
        "uncertainty": uncertainty,
        "missing_critical_fields": missing_fields,
        "routing_decision": routing_decision,
        "routing_reason": routing_reason,
    }

    # Record risk recalculation in audit log
    append_audit_event(
        temp_state,
        "risk_recalculated",
        {"risk_score": risk_score, "uncertainty": uncertainty, "routing": routing_decision},
    )
    updates["audit_log"] = temp_state.get("audit_log", [])

    return updates


def select_question_node(state: PatientState) -> Dict[str, Any]:
    """Select the next best question from question bank or fallback."""
    question_bank = state.get("question_bank") or []
    next_q = select_next_question(state, question_bank)

    asked = list(state.get("asked_questions") or [])
    if next_q and next_q.get("id") and next_q["id"] not in asked:
        asked.append(next_q["id"])

    updates: Dict[str, Any] = {
        "current_question": next_q,
        "asked_questions": asked,
    }

    if next_q:
        temp_state = {**state, **updates}
        append_audit_event(temp_state, "question_asked", {"question": next_q})
        updates["audit_log"] = temp_state.get("audit_log", [])

    return updates


def receive_answer_node(state: PatientState) -> Dict[str, Any]:
    """Capture new answer provided for the current cycle."""
    new_ans = state.get("new_answer")
    return {"new_answer": new_ans}


def update_state_node(state: PatientState) -> Dict[str, Any]:
    """Apply answer to patient state and update turn count."""
    new_ans = state.get("new_answer")
    if new_ans and isinstance(new_ans, dict):
        field = new_ans.get("field")
        value = new_ans.get("value")
        if field and field in (
            "oxygen_saturation",
            "heart_rate",
            "respiratory_rate",
            "systolic_bp",
            "temperature",
            "age",
            "spo2",
            "hr",
            "rr",
            "sbp",
            "temp",
        ):
            is_valid, _ = validate_vital(field, value)
            if not is_valid:
                # Do NOT update invalid field
                return {}

    updates: Dict[str, Any] = {
        "turn_count": state.get("turn_count", 0) + 1,
    }

    if new_ans and isinstance(new_ans, dict):
        field = new_ans.get("field")
        value = new_ans.get("value")
        if field:
            updates[field] = value
            # Also sync into nested dict if applicable
            if field in ["chest_pain", "pain_radiation", "shortness_of_breath", "confusion", "dizziness"]:
                symptoms = dict(state.get("symptoms") or {})
                symptoms[field] = value
                updates["symptoms"] = symptoms
            elif field in ["oxygen_saturation", "heart_rate", "respiratory_rate", "systolic_bp", "temperature", "age"]:
                vitals = dict(state.get("vitals") or {})
                vitals[field] = value
                updates["vitals"] = vitals

    return updates


def detect_contradictions_node(state: PatientState) -> Dict[str, Any]:
    """Detect contradiction against prior answers and update audit log."""
    new_ans = state.get("new_answer")
    contradictions = list(state.get("contradictions") or [])
    contra_details = list(state.get("contradiction_details") or [])
    temp_state = dict(state)

    if new_ans and isinstance(new_ans, dict):
        field = new_ans.get("field")
        value = new_ans.get("value")
        if field:
            conflict = detect_contradiction(state, field, value)
            if conflict:
                contra_str = f"Contradiction on {field}: previously {conflict['old_value']}, now {conflict['new_value']}"
                if contra_str not in contradictions:
                    contradictions.append(contra_str)
                    contra_details.append(conflict)
                    append_audit_event(temp_state, "contradiction_detected", conflict)

    return {
        "contradictions": contradictions,
        "contradiction_details": contra_details,
        "audit_log": temp_state.get("audit_log", []),
    }


def decide_node(state: PatientState) -> Dict[str, Any]:
    """Mark triage complete and record final routing decision."""
    routing = state.get("routing_decision")
    reason = state.get("routing_reason") or "Triage routing completed."
    temp_state = dict(state)
    append_audit_event(temp_state, "routing_finalized", {"routing": routing, "reason": reason})

    return {
        "is_complete": True,
        "routing_decision": routing,
        "routing_reason": reason,
        "audit_log": temp_state.get("audit_log", []),
    }


# ---------------------------------------------------------------------------
# Routing / Condition Logic
# ---------------------------------------------------------------------------

def should_decide_or_continue(state: PatientState) -> str:
    """Assess whether a decisive routing outcome has been reached."""
    if state.get("routing_decision") is not None:
        return "decide"
    return "select_question"


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Construct the LangGraph triage workflow."""
    workflow = StateGraph(PatientState)

    workflow.add_node("assess", assess_node)
    workflow.add_node("select_question", select_question_node)
    workflow.add_node("receive_answer", receive_answer_node)
    workflow.add_node("update_state", update_state_node)
    workflow.add_node("detect_contradictions", detect_contradictions_node)
    workflow.add_node("decide", decide_node)

    # Edges
    workflow.add_edge(START, "assess")
    workflow.add_conditional_edges(
        "assess",
        should_decide_or_continue,
        {
            "decide": "decide",
            "select_question": "select_question",
        },
    )
    workflow.add_edge("select_question", "receive_answer")
    workflow.add_edge("receive_answer", "update_state")
    workflow.add_edge("update_state", "detect_contradictions")
    workflow.add_edge("detect_contradictions", "assess")
    workflow.add_edge("decide", END)

    return workflow


# Compiled graph instance
_compiled_graph = None

def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph().compile()
    return _compiled_graph


# ---------------------------------------------------------------------------
# Single-Cycle Execution for Streamlit
# ---------------------------------------------------------------------------

def run_one_cycle(
    state: PatientState,
    new_answer: Optional[Dict[str, Any]] = None,
    question_bank: Optional[Any] = None,
) -> PatientState:
    """Execute ONE turn/cycle of the triage interaction without infinite looping.

    Steps performed:
    1. If `new_answer` provided:
       - Validate physiological plausibility of vital-sign fields.
       - If invalid, log audit event, keep prior state unchanged, set UI warning,
         re-select the same question via select_next_question, and return.
       - Check for contradiction against current state.
       - Update patient state with the new value.
       - Re-run contradiction detection and audit log.
    2. Assess risk, uncertainty, and missing critical fields.
    3. Evaluate deterministic routing:
       - If routing decision reached (EMERGENCY, PRIMARY_CARE, DISCHARGE, ESCALATE_TO_HUMAN):
         finalize routing, mark `is_complete = True`.
       - Else:
         select next question from question bank.
    """
    updated_state: PatientState = dict(state)  # type: ignore

    if question_bank is not None:
        updated_state["question_bank"] = question_bank

    if "audit_log" not in updated_state or updated_state["audit_log"] is None:
        updated_state["audit_log"] = []
    if "contradictions" not in updated_state or updated_state["contradictions"] is None:
        updated_state["contradictions"] = []
    if "contradiction_details" not in updated_state or updated_state["contradiction_details"] is None:
        updated_state["contradiction_details"] = []
    if "turn_count" not in updated_state or updated_state["turn_count"] is None:
        updated_state["turn_count"] = 0

    # Step 1: Process new answer if provided
    if new_answer and isinstance(new_answer, dict):
        updated_state["new_answer"] = new_answer
        
        field = new_answer.get("field")
        value = new_answer.get("value")
        if field:
            # Check physiological plausibility for vital signs & age
            if field in (
                "oxygen_saturation",
                "heart_rate",
                "respiratory_rate",
                "systolic_bp",
                "temperature",
                "age",
                "spo2",
                "hr",
                "rr",
                "sbp",
                "temp",
            ):
                is_valid, reason = validate_vital(field, value)
                if not is_valid:
                    # Log audit event
                    append_audit_event(
                        updated_state,
                        "invalid_input_rejected",
                        {"field": field, "attempted_value": value, "reason": reason},
                    )
                    # Do NOT update that field in patient state
                    # Set warning message in state
                    warning_msg = f"That value is outside a plausible range for {field} — please re-enter."
                    updated_state["last_validation_error"] = warning_msg

                    # Re-select the SAME question (via select_next_question) instead of advancing
                    qb = question_bank or updated_state.get("question_bank") or []
                    next_q = select_next_question(updated_state, qb)
                    if next_q:
                        updated_state["current_question"] = next_q

                    # Re-assess without advancing turn_count
                    assess_res = assess_node(updated_state)
                    updated_state.update(assess_res)
                    return updated_state

            # Valid answer: clear last validation error
            updated_state["last_validation_error"] = None

            # Detect contradiction before overwriting
            contra = detect_contradiction(updated_state, field, value)
            if contra:
                contra_msg = f"Contradiction on {field}: previously {contra['old_value']}, now {contra['new_value']}"
                if contra_msg not in updated_state["contradictions"]:
                    updated_state["contradictions"].append(contra_msg)
                    updated_state["contradiction_details"].append(contra)
                    append_audit_event(updated_state, "contradiction_detected", contra)

            # Apply value update
            updated_state[field] = value
            if field in ["chest_pain", "pain_radiation", "shortness_of_breath", "confusion", "dizziness"]:
                syms = dict(updated_state.get("symptoms") or {})
                syms[field] = value
                updated_state["symptoms"] = syms
            elif field in ["oxygen_saturation", "heart_rate", "respiratory_rate", "systolic_bp", "temperature", "age"]:
                vits = dict(updated_state.get("vitals") or {})
                vits[field] = value
                updated_state["vitals"] = vits

        updated_state["turn_count"] = updated_state.get("turn_count", 0) + 1

    # Step 2: Assess risk & uncertainty
    assess_res = assess_node(updated_state)
    updated_state.update(assess_res)

    # Step 3: Check routing decision
    routing = updated_state.get("routing_decision")
    if routing is not None:
        decide_res = decide_node(updated_state)
        updated_state.update(decide_res)
        updated_state["is_complete"] = True
    else:
        # If no routing decision reached yet, select next question
        q_res = select_question_node(updated_state)
        updated_state.update(q_res)
        updated_state["is_complete"] = False

    return updated_state
