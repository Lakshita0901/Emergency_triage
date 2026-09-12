# TriageFlow AI

> **Deterministic, human-in-the-loop emergency triage agent built on LangGraph and Streamlit.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/pytest-127%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## ⚠️ Safety Disclaimer (Verbatim)

> **SAFETY DISCLAIMER:** TriageFlow AI is a simulated emergency triage research prototype designed for hackathon demonstration and academic exploration. It is **NOT** a certified medical device, **NOT** intended for actual clinical use, diagnostic evaluation, or emergency routing of real human patients, and must **NEVER** replace the clinical judgment of qualified healthcare professionals.

---

## 🎯 Architecture Diagram

```text
=========================================================================================
                                  TRIAGEFLOW AI ARCHITECTURE
=========================================================================================

  +-----------------------------------------------------------------------------------+
  |                               Streamlit Frontend                                  |
  |   - Column 1: Patient Profile, Recorded Vitals, Known Symptoms, Contradictions    |
  |   - Column 2: Adaptive Question Form, Free-text Injection, Contradiction Toggle   |
  |   - Column 3: Live Risk Gauge, Routing Banner, Dynamic Factors, Audit Log (JSON)  |
  +-----------------------------------------------------------------------------------+
                                         │
                               run_one_cycle(state, answer)
                                         ▼
  +-----------------------------------------------------------------------------------+
  |                               LangGraph Workflow                                  |
  |                                                                                   |
  |        [START] ────────► [ assess ] ──────(Decisive routing reached?)             |
  |                             ▲   │                                │                |
  |                             │   │ NO (Needs data)                │ YES            |
  |                             │   ▼                                ▼                |
  |                             │ [ select_question ]         [ decide ] ──► [END]    |
  |                             │   │                                                 |
  |                             │   ▼                                                 |
  |                             │ [ receive_answer ]                                  |
  |                             │   │                                                 |
  |                             │   ▼                                                 |
  |                             │ [ update_state ]                                    |
  |                             │   │                                                 |
  |                             │   ▼                                                 |
  |                             │ [ detect_contradictions ]                           |
  |                             └───────┘                                             |
  +-----------------------------------------------------------------------------------+
                                         │
                                         ▼
  +-----------------------------------------------------------------------------------+
  |                           Deterministic Engines (No LLM)                          |
  |                                                                                   |
  |  • tools.risk_scorer:                                                             |
  |      SpO2, HR, RR, SBP, Temp, Symptoms (+2/+3), Age table (Score: 0 - 27)         |
  |  • agent.decision_engine:                                                         |
  |      Uncertainty = (missing_crit / 6) + min(0.15 * contradictions, 0.30)          |
  |  • agent.question_selector:                                                       |
  |      Score = Risk Impact × Uncertainty Reduction × Context Relevance (1.0 vs 0.6) |
  |  • tools.contradiction_detector:                                                  |
  |      Boolean symptom state tracking & ISO-8601 audit event appending              |
  +-----------------------------------------------------------------------------------+
```

---

## 🚦 The 4 Routing Outcomes

TriageFlow AI evaluates patients into exactly four deterministic routing categories:

| Outcome | Risk Level & Condition | Clinical Action & Guidance |
| :--- | :--- | :--- |
| **`EMERGENCY`** | **High Risk ($\ge 7$)** with low uncertainty ($< 0.50$), zero contradictions, and no missing critical fields. | Immediate emergency department resuscitation, acute medical intervention, and emergent physician team activation. |
| **`ESCALATE_TO_HUMAN`** | **High Risk ($\ge 7$)** AND (uncertainty $\ge 0.50$, detected contradictions, OR critical missing fields). | Immediate bedside triage nurse or attending physician evaluation. System halts autonomous routing to prevent unsafe clinical failure. |
| **`PRIMARY_CARE`** | **Moderate Risk ($4 \le \text{risk} \le 6$)**. | Outpatient clinic, urgent care center, or primary care physician visit within 24–48 hours for non-emergent workup. |
| **`DISCHARGE`** | **Low Risk ($\text{risk} < 4$)** with verified low uncertainty ($\text{uncertainty} \le 0.25$). | Safe for discharge with home care self-monitoring instructions and emergency return warnings. |

> **Note on Incomplete Data:** If $\text{risk} < 4$ but $\text{uncertainty} > 0.25$, the routing outcome remains `None` (pending), and the agent iteratively selects the next optimal question.

---

## ⚡ Setup & Quick Start

### 1. Clone & Install Dependencies

```bash
# Clone the repository
git clone https://github.com/your-org/emergency_triage.git
cd emergency_triage

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch Streamlit Application

```bash
streamlit run app.py
```

*Windows fallback (if `streamlit` is not in your system PATH):*
```powershell
py -m streamlit run app.py
```

### 3. Run Test Suite

```bash
# Run all 127 deterministic test cases
pytest -v

# Or using the Python launcher on Windows
py -m pytest tests/ -v
```

---

## 📂 Repository Structure

```text
emergency_triage/
├── app.py                     # 3-Column interactive Streamlit triage UI & scenario runner
├── requirements.txt           # Pinned dependencies (streamlit, langgraph, pydantic, pandas, pytest)
├── README.md                  # System overview, architecture, setup, and safety documentation
├── agent/
│   ├── __init__.py            # Agent package initialization
│   ├── graph.py               # LangGraph state machine & single-cycle step executor
│   ├── state.py               # PatientState TypedDict specification
│   ├── question_selector.py   # Contextual & entropy/risk question selector
│   └── decision_engine.py     # Deterministic uncertainty & 4-outcome routing engine
├── tools/
│   ├── __init__.py            # Tools package initialization
│   ├── risk_scorer.py         # Pure Python threshold vital/symptom risk scorer
│   ├── contradiction_detector.py # Boolean conflict detector & ISO audit event logger
│   └── patient_data_tool.py   # Patient records normalization & merger
├── data/
│   ├── question_bank.json     # 11 structured clinical inquiries with entropy weights
│   ├── synthetic_cases.json   # Step-by-step presets for High-Risk, Contradiction, & Low-Risk
│   └── triage_guidelines.json # Standardized threshold rules and point tables
├── tests/
│   ├── test_cases.py          # All 7 routing branches, boundary conditions, & uncertainty tests
│   ├── test_risk_scorer.py    # Cutoff boundaries for every score_* function & synthetic cases
│   └── test_question_selector.py # Candidate exhaustion, context weights, and fixtures
└── docs/
    ├── architecture.png.txt   # Architecture diagram reference notes
    ├── demo_script.md         # 2.5-minute timestamped live hackathon demonstration script
    └── presentation_outline.md# 4-slide presentation outline
```
