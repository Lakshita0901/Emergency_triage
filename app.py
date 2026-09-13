"""TriageFlow AI — Simulated Emergency Triage Agent (Streamlit Interface).

Pure deterministic triage UI over Python rules and LangGraph workflow.
NO external APIs, NO LLM routing.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import streamlit as st

from agent.graph import run_one_cycle
from agent.state import PatientState
from tools.patient_data_tool import classify_value, lookup_reference_ranges

# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TriageFlow AI — Emergency Triage",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 98%;
    }
    .metric-card {
        background-color: #1e2530;
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .banner-emergency {
        background: linear-gradient(90deg, #e53e3e, #9b2c2c);
        color: white;
        padding: 16px;
        border-radius: 8px;
        font-size: 22px;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 4px 12px rgba(229, 62, 62, 0.4);
    }
    .banner-escalate {
        background: linear-gradient(90deg, #dd6b20, #9c4221);
        color: white;
        padding: 16px;
        border-radius: 8px;
        font-size: 20px;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 4px 12px rgba(221, 107, 32, 0.4);
    }
    .banner-primary {
        background: linear-gradient(90deg, #d69e2e, #744210);
        color: white;
        padding: 16px;
        border-radius: 8px;
        font-size: 20px;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 4px 12px rgba(214, 158, 46, 0.4);
    }
    .banner-discharge {
        background: linear-gradient(90deg, #38a169, #22543d);
        color: white;
        padding: 16px;
        border-radius: 8px;
        font-size: 20px;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 4px 12px rgba(56, 161, 105, 0.4);
    }
    .banner-pending {
        background: #2d3748;
        color: #e2e8f0;
        padding: 14px;
        border-radius: 8px;
        font-size: 18px;
        text-align: center;
        border: 1px dashed #718096;
    }
    .audit-box {
        max-height: 420px;
        overflow-y: auto;
        font-family: monospace;
        font-size: 11px;
        background-color: #0f141c;
        border: 1px solid #2d3748;
        border-radius: 6px;
        padding: 10px;
    }
    .stButton button {
        border-radius: 6px;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Data Loaders
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"

def _sort_steps(case: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return case steps sorted by the 'order' field."""
    return sorted(case.get("steps", []), key=lambda s: s.get("order", 0))


@st.cache_data
def load_question_bank() -> List[Dict[str, Any]]:
    path = DATA_DIR / "question_bank.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("questions", [])
    return []

@st.cache_data
def load_synthetic_cases() -> List[Dict[str, Any]]:
    path = DATA_DIR / "synthetic_cases.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("cases", [])
    return []

# ---------------------------------------------------------------------------
# Initial State Generator
# ---------------------------------------------------------------------------
def create_initial_patient_state(age: Optional[int] = None, chief_complaint: str = "general") -> PatientState:
    qb = load_question_bank()
    base_state: PatientState = {
        "age": age,
        "chief_complaint": chief_complaint,
        "vitals": {"age": age} if age else {},
        "symptoms": {},
        "oxygen_saturation": None,
        "heart_rate": None,
        "respiratory_rate": None,
        "systolic_bp": None,
        "temperature": None,
        "chest_pain": None,
        "pain_radiation": None,
        "shortness_of_breath": None,
        "confusion": None,
        "dizziness": None,
        "messages": [],
        "asked_questions": [],
        "current_question": None,
        "new_answer": None,
        "risk_score": 0,
        "risk_level": "LOW",
        "risk_factors": [],
        "uncertainty": 1.0,
        "contradictions": [],
        "contradiction_details": [],
        "audit_log": [],
        "missing_critical_fields": [
            "oxygen_saturation",
            "heart_rate",
            "respiratory_rate",
            "systolic_bp",
            "chest_pain",
            "shortness_of_breath",
        ],
        "invalid_fields": [],
        "last_validation_error": None,
        "routing_decision": None,
        "routing_reason": None,
        "rationale": None,
        "turn_count": 0,
        "is_complete": False,
    }
    # Run initial assessment cycle to set initial question and uncertainty
    return run_one_cycle(base_state, new_answer=None, question_bank=qb)

# ---------------------------------------------------------------------------
# Initialize Session State
# ---------------------------------------------------------------------------
if "patient_state" not in st.session_state:
    st.session_state.patient_state = create_initial_patient_state()

if "demo_case" not in st.session_state:
    st.session_state.demo_case = None

if "demo_step_idx" not in st.session_state:
    st.session_state.demo_step_idx = 0

state: PatientState = st.session_state.patient_state
question_bank = load_question_bank()
synthetic_cases = load_synthetic_cases()

# ---------------------------------------------------------------------------
# Top Controls Header
# ---------------------------------------------------------------------------
st.markdown("### 🏥 **TriageFlow AI** `Simulated Emergency Triage (Prototype - NOT Clinical)`")

col_b1, col_b2, col_b3, col_b4, col_b5, col_b_inv, col_b6 = st.columns([1.1, 1.0, 1.2, 1.3, 1.2, 1.6, 0.9])

with col_b1:
    if st.button("➕ Initialize Patient", use_container_width=True):
        st.session_state.patient_state = create_initial_patient_state(age=45, chief_complaint="general")
        st.session_state.demo_case = None
        st.session_state.demo_step_idx = 0
        st.rerun()

with col_b2:
    if st.button("🚀 Start Triage", use_container_width=True):
        qb = load_question_bank()
        st.session_state.patient_state = run_one_cycle(st.session_state.patient_state, new_answer=None, question_bank=qb)
        st.rerun()

def _load_demo(case_id: str, fallback_complaint: str = "general") -> None:
    cases = [c for c in synthetic_cases if c["id"] == case_id]
    if cases:
        demo = cases[0]
        st.session_state.demo_case = demo
        st.session_state.demo_steps = _sort_steps(demo)
        st.session_state.demo_step_idx = 0
        st.session_state.patient_state = create_initial_patient_state(
            age=demo["demographics"]["age"],
            chief_complaint=demo.get("chief_complaint", fallback_complaint)
        )
        st.rerun()

with col_b3:
    if st.button("🔴 High-Risk Demo", use_container_width=True):
        _load_demo("high_risk_chest_pain", "chest_pain")

with col_b4:
    if st.button("⚡ Contradiction Demo", use_container_width=True):
        _load_demo("contradictory_breathing_case", "shortness_of_breath")

with col_b5:
    if st.button("🟢 Low-Risk Demo", use_container_width=True):
        _load_demo("low_risk_muscle_pain", "chest_pain")

with col_b_inv:
    if st.button("⚠️ Simulate Invalid Input", use_container_width=True):
        curr_q = st.session_state.patient_state.get("current_question")
        target_field = curr_q.get("field") if curr_q else "oxygen_saturation"
        invalid_val_map = {
            "oxygen_saturation": 150.0,
            "heart_rate": 500,
            "respiratory_rate": 150,
            "systolic_bp": 450,
            "temperature": 60.0,
            "age": 250,
        }
        bad_value = invalid_val_map.get(target_field, 150.0)
        qb = load_question_bank()
        st.session_state.patient_state = run_one_cycle(
            st.session_state.patient_state,
            new_answer={"field": target_field, "value": bad_value},
            question_bank=qb,
        )
        st.rerun()

with col_b6:
    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.patient_state = create_initial_patient_state()
        st.session_state.demo_case = None
        st.session_state.demo_step_idx = 0
        st.rerun()

# ---------------------------------------------------------------------------
# Demo Scenario Stepper Banner (when a demo is active)
# ---------------------------------------------------------------------------
if st.session_state.demo_case:
    demo = st.session_state.demo_case
    steps = st.session_state.get("demo_steps") or _sort_steps(demo)
    idx = st.session_state.demo_step_idx
    total_steps = len(steps)

    st.markdown("---")
    expected = demo.get("expected_routing", "")
    c_demo_info, c_demo_btn = st.columns([3.5, 1.5])
    with c_demo_info:
        st.info(
            f"**Active Demo** — {demo['title']}  "
            f"(Step {idx}/{total_steps}, Expected: **{expected}**)\n\n"
            f"*{demo['description']}*"
        )
    with c_demo_btn:
        if idx < total_steps:
            next_step = steps[idx]
            note = next_step.get("clinical_note", "")
            btn_label = f"Step {idx + 1}: {next_step['label']}"
            st.caption(f"📋 {note}")
            if st.button(btn_label, type="primary", use_container_width=True):
                st.session_state.patient_state = run_one_cycle(
                    st.session_state.patient_state,
                    new_answer={"field": next_step["field"], "value": next_step["value"]},
                    question_bank=question_bank,
                )
                st.session_state.demo_step_idx += 1
                st.rerun()
        else:
            st.success("✅ Demo sequence complete.")

st.markdown("---")

# ===========================================================================
# 3-Column Main Layout
# ===========================================================================
left_col, center_col, right_col = st.columns([1.1, 1.4, 1.3], gap="large")

# ---------------------------------------------------------------------------
# LEFT COLUMN: Patient Overview, Known Vitals & Symptoms
# ---------------------------------------------------------------------------
with left_col:
    st.markdown("#### 📋 Patient Profile & Data")

    # Demographics
    st.markdown(
        f"""
        <div class="metric-card">
            <b>Age:</b> {state.get('age') if state.get('age') is not None else '<i>Unknown</i>'}<br>
            <b>Chief Complaint:</b> {state.get('chief_complaint') or 'General'}<br>
            <b>Turn Count:</b> {state.get('turn_count', 0)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Known Vitals
    st.markdown("##### 🫀 Known Vitals")
    vitals_data = [
        ("oxygen_saturation", "Oxygen Saturation (SpO2)", state.get("oxygen_saturation"), "%"),
        ("heart_rate", "Heart Rate (HR)", state.get("heart_rate"), "bpm"),
        ("respiratory_rate", "Respiratory Rate (RR)", state.get("respiratory_rate"), "/min"),
        ("systolic_bp", "Systolic BP", state.get("systolic_bp"), "mmHg"),
        ("temperature", "Temperature", state.get("temperature"), "°C"),
    ]
    vitals_table = []
    for field_key, label, val, unit in vitals_data:
        classification = classify_value(field_key, val)
        ref_ranges = lookup_reference_ranges(field_key)
        norm_range = ref_ranges.get("normal_range")
        ref_text = f"{norm_range[0]}–{norm_range[1]} {unit}" if norm_range else ""

        if val is None or classification == "unknown":
            status_label = "⚪ Missing"
        elif classification == "normal":
            status_label = "🟢 Normal"
        elif classification == "abnormal":
            status_label = "🟡 Abnormal"
        elif classification == "critical":
            status_label = "🔴 Critical"
        else:
            status_label = f"⚪ {classification.title()}"

        vitals_table.append({
            "Vital": label,
            "Value": f"{val} {unit}" if val is not None else "— Missing —",
            "Status": status_label,
            "Ref Range": ref_text,
        })
    st.dataframe(vitals_table, hide_index=True, use_container_width=True)

    # Known Symptoms
    st.markdown("##### 🩺 Known Symptoms")
    symptoms_data = [
        ("Chest Pain", state.get("chest_pain")),
        ("Pain Radiation", state.get("pain_radiation")),
        ("Shortness of Breath", state.get("shortness_of_breath")),
        ("Confusion", state.get("confusion")),
        ("Dizziness", state.get("dizziness")),
    ]
    symptoms_table = []
    for label, val in symptoms_data:
        display_val = "Yes" if val is True else ("No" if val is False else "— Unknown —")
        symptoms_table.append({
            "Symptom": label,
            "Present": display_val,
        })
    st.dataframe(symptoms_table, hide_index=True, use_container_width=True)

    # Missing Critical Fields
    st.markdown("##### ⚠️ Missing Critical Fields")
    missing = state.get("missing_critical_fields") or []
    if missing:
        for m in missing:
            st.markdown(f"- 🔸 `{m}`")
    else:
        st.success("All critical fields recorded!")

    # Contradictions List
    st.markdown("##### ⚡ Contradictions")
    contradictions = state.get("contradictions") or []
    if contradictions:
        for c in contradictions:
            st.error(f"⚠️ {c}")
    else:
        st.caption("No contradictions detected.")


# ---------------------------------------------------------------------------
# CENTER COLUMN: Adaptive Questioning & Data Ingestion
# ---------------------------------------------------------------------------
with center_col:
    st.markdown("#### 💬 Active Inquiry & Input")

    val_error = state.get("last_validation_error")
    if val_error:
        st.warning(val_error)

    current_q = state.get("current_question")
    is_complete = state.get("is_complete", False)
    routing_decision = state.get("routing_decision")

    if is_complete or routing_decision is not None:
        st.info(f"**Triage decision finalized.** Current Routing: **{routing_decision}**")
        if state.get("routing_reason"):
            st.caption(f"Reason: {state.get('routing_reason')}")
    elif current_q:
        st.markdown(
            f"""
            <div class="metric-card" style="border-left: 4px solid #3182ce;">
                <span style="font-size:16px; font-weight:600;">{current_q.get('question') or current_q.get('text', '')}</span><br>
                <small style="color: #63b3ed;"><b>Target Field:</b> <code>{current_q.get('field', '')}</code></small>
            </div>
            """,
            unsafe_allow_html=True,
        )

        reason_str = current_q.get("reason") or current_q.get("selection_reason") or "Adaptive priority selection."
        st.markdown(f"💡 *{reason_str}*")

        field = current_q.get("field")
        q_type = current_q.get("type", "float")

        with st.form("answer_form"):
            user_val = None
            if q_type == "bool":
                ans_bool = st.radio("Response:", ["Yes", "No"], horizontal=True)
                user_val = (ans_bool == "Yes")
            elif q_type == "int":
                min_v = int(current_q.get("min", 0))
                max_v = int(current_q.get("max", 300))
                default_v = 75 if "heart" in field else (16 if "resp" in field else 120)
                user_val = st.number_input("Value:", min_value=min_v, max_value=max_v, value=default_v, step=1)
            else:
                min_v = float(current_q.get("min", 0.0))
                max_v = float(current_q.get("max", 100.0))
                default_v = 98.0 if "sat" in field else 37.0
                user_val = st.number_input("Value:", min_value=min_v, max_value=max_v, value=default_v, step=0.1)

            submitted = st.form_submit_button("Submit Answer ➔", type="primary", use_container_width=True)
            if submitted:
                st.session_state.patient_state = run_one_cycle(
                    state,
                    new_answer={"field": field, "value": user_val},
                    question_bank=question_bank,
                )
                st.rerun()
    else:
        st.info("No active question. All available questions answered or routing complete.")

    st.markdown("---")

    # Inject Free-text Information / Custom Field
    st.markdown("##### 💉 Inject Specific Field Value")
    with st.expander("Manual / Free-text Field Injection", expanded=False):
        c_f1, c_f2 = st.columns([1.5, 1.2])
        all_possible_fields = [
            "oxygen_saturation", "heart_rate", "respiratory_rate", "systolic_bp",
            "temperature", "chest_pain", "pain_radiation", "shortness_of_breath",
            "confusion", "dizziness", "age"
        ]
        with c_f1:
            sel_field = st.selectbox("Select Field to update:", all_possible_fields)
        with c_f2:
            if sel_field in ["chest_pain", "pain_radiation", "shortness_of_breath", "confusion", "dizziness"]:
                sel_val = (st.selectbox("Value:", ["True", "False"]) == "True")
            else:
                sel_val = st.number_input("Value:", value=95.0, step=1.0)

        if st.button("Inject Field Value", use_container_width=True):
            st.session_state.patient_state = run_one_cycle(
                state,
                new_answer={"field": sel_field, "value": sel_val},
                question_bank=question_bank,
            )
            st.rerun()

    # Inject Contradiction Button
    st.markdown("##### ⚡ Inject Contradiction")
    st.caption("Flips a known boolean symptom to immediately test conflict resolution & escalation.")
    boolean_fields = ["chest_pain", "shortness_of_breath", "confusion", "dizziness"]
    known_bools = [f for f in boolean_fields if state.get(f) is not None]

    if known_bools:
        target_flip = st.selectbox("Select boolean field to invert:", known_bools)
        current_val = state.get(target_flip)
        flipped_val = not bool(current_val)
        if st.button(f"Flip `{target_flip}` ({current_val} ➔ {flipped_val})", type="secondary", use_container_width=True):
            st.session_state.patient_state = run_one_cycle(
                state,
                new_answer={"field": target_flip, "value": flipped_val},
                question_bank=question_bank,
            )
            st.rerun()
    else:
        st.info("Record at least one boolean symptom first to test contradiction injection.")


# ---------------------------------------------------------------------------
# RIGHT COLUMN: Risk Engine, Routing Banner & Audit Trail
# ---------------------------------------------------------------------------
with right_col:
    st.markdown("#### 🎯 Triage Decision & Risk Analysis")

    # Routing Banner
    decision = state.get("routing_decision")
    if decision == "EMERGENCY":
        st.markdown('<div class="banner-emergency">🚨 EMERGENCY</div>', unsafe_allow_html=True)
    elif decision == "ESCALATE_TO_HUMAN":
        st.markdown('<div class="banner-escalate">👤 ESCALATE TO HUMAN</div>', unsafe_allow_html=True)
    elif decision == "PRIMARY_CARE":
        st.markdown('<div class="banner-primary">🩺 PRIMARY CARE</div>', unsafe_allow_html=True)
    elif decision == "DISCHARGE":
        st.markdown('<div class="banner-discharge">🟢 DISCHARGE</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="banner-pending">⏳ PENDING (Collecting vitals/symptoms...)</div>', unsafe_allow_html=True)

    if state.get("routing_reason"):
        st.caption(f"**Rationale:** {state.get('routing_reason')}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Score Metrics
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Risk Score", f"{state.get('risk_score', 0)}")
    with m2:
        st.metric("Risk Level", f"{state.get('risk_level', 'LOW')}")
    with m3:
        unc = state.get("uncertainty", 1.0)
        st.metric("Uncertainty", f"{unc:.2f}")

    # Active Risk Factors
    st.markdown("##### 📊 Active Risk Factors")
    factors = state.get("risk_factors") or []
    if factors:
        for fac in factors:
            st.markdown(f"- 🔴 {fac}")
    else:
        st.caption("No positive risk factors recorded.")

    st.markdown("---")

    # Audit Trail (Newest first)
    c_at_title, c_at_btn = st.columns([2.0, 1.5])
    with c_at_title:
        st.markdown("##### 📜 Audit Trail")
    with c_at_btn:
        audit_log = state.get("audit_log") or []
        json_log = json.dumps(audit_log, indent=2)
        st.download_button(
            label="📥 Download JSON",
            data=json_log,
            file_name="triageflow_audit_log.json",
            mime="application/json",
            use_container_width=True,
        )

    # Render Audit Box
    if audit_log:
        audit_lines = []
        for ev in reversed(audit_log):
            t = ev.get("timestamp", "")
            ev_type = ev.get("event_type", "event")
            payload = json.dumps(ev.get("payload", {}))
            audit_lines.append(f"[{t}] {ev_type.upper()}: {payload}")
        audit_text = "\n\n".join(audit_lines)
        st.markdown(f'<div class="audit-box"><pre>{audit_text}</pre></div>', unsafe_allow_html=True)
    else:
        st.caption("No audit events recorded yet.")
