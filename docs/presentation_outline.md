# TriageFlow AI — Presentation Outline (4 Slides)

> **Theme:** Deterministic, Safe, Human-in-the-Loop Emergency Triage Agent  
> **Target Audience:** Hackathon Judges, Clinical Directors, Technical Evaluators

---

## Slide 1: The Problem — ED Bottlenecks & The Hallucination Hazard

- **The Crisis in Emergency Departments:**
  - Emergency rooms globally experience unprecedented overcrowding, delayed triage, and clinician burnout.
  - Initial patient assessments often take 30–90 minutes during peak hours, delaying critical interventions for acute conditions like STEMI and sepsis.
- **The Generative AI Dilemma:**
  - General-purpose LLMs are probabilistic text predictors prone to stochastic hallucinations, inconsistent outputs, and lack of mathematical rigor.
  - Entrusting an unconstrained LLM to independently route emergency patients is unsafe and unacceptable for clinical deployment.
- **Our Vision:**
  - What if we paired the natural conversational flexibility of modern agent architectures with the uncompromising safety of deterministic clinical rule engines?

---

## Slide 2: System Architecture — Deterministic Rules over LangGraph

- **Core Architectural Invariant:**
  - **The LLM never makes the clinical routing decision.** Pure Python code computes all risk scores, uncertainty indexes, and routing states.
- **LangGraph Stateful Orchestration:**
  - State machine with cyclical graph topology: `assess` ➔ `select_question` ➔ `receive_answer` ➔ `update_state` ➔ `detect_contradictions` ➔ `assess`.
  - Non-blocking single-cycle execution pattern (`run_one_cycle`) perfectly synchronized with the user interaction turn in Streamlit.
- **Adaptive Question Selection:**
  - Dynamically computes information gain: $\text{Score} = \text{Risk Impact} \times \text{Uncertainty Reduction} \times \text{Context Relevance}$.
  - Prioritizes chief complaint questions ($1.0$ weight) and high-impact vital signs ($0.6$ general weight), minimizing unnecessary patient questioning.

---

## Slide 3: The Engine — Scorer, Uncertainty Index & 4 Routing Outcomes

- **Deterministic Risk Scoring Table (0 to 27+ Points):**
  - **Vitals:** $\text{SpO}_2 \le 91\%$ (+3), $92\text{--}93\%$ (+2) | $\text{HR} \ge 131$ or $\le 40$ (+3), $111\text{--}130$ (+2) | $\text{RR} \ge 25$ (+3) | $\text{Systolic BP} \le 90$ (+3) | $\text{Temp} \ge 39.1^\circ\text{C}$ or $\le 35.0^\circ\text{C}$ (+3).
  - **Symptoms:** Chest Pain (+2), Radiation (+2), Shortness of Breath (+2), Confusion (+3).
  - **Demographics:** Age $\ge 70$ (+2), Age $55\text{--}69$ (+1).
- **Formal Uncertainty Metric:**
  - $\text{Uncertainty} = \frac{\text{Missing Critical Fields}}{6} + \min(0.15 \times \text{Contradictions}, 0.30)$, bounded in $[0.0, 1.0]$.
- **The 4 Transparent Clinical Routes:**
  1. **`EMERGENCY`**: $\text{Risk} \ge 7$, clean complete data $\rightarrow$ Immediate resuscitation & ED escalation.
  2. **`ESCALATE_TO_HUMAN`**: $\text{Risk} \ge 7$ with uncertainty $\ge 0.50$, missing critical fields, or contradictions $\rightarrow$ Nurse override.
  3. **`PRIMARY_CARE`**: Moderate Risk ($4 \le \text{Risk} \le 6$) $\rightarrow$ Outpatient clinic within 24–48 hours.
  4. **`DISCHARGE`**: Low Risk ($\text{Risk} < 4$) and verified $\text{Uncertainty} \le 0.25 \rightarrow$ Safe discharge with home monitoring.

---

## Slide 4: Validation, Impact & Future Roadmap

- **Comprehensive Automated Testing:**
  - **127 automated pytest tests passing** across 3 dedicated test suites.
  - 100% boundary coverage for every cutoff threshold (just below, at, and just above).
  - Complete verification of all 3 synthetic clinical cases and routing branches.
- **Enterprise-Ready Auditability:**
  - Full ISO-8601 timestamped audit trail tracking every risk recalculation, question asked, contradiction detected, and routing decision.
  - Instant JSON download for medical-legal compliance and peer review.
- **Future Clinical Roadmap:**
  - HL7 / FHIR EHR integration for automated electronic chart ingestion.
  - Multi-vital wearable sensor streaming (BLE SpO2 and continuous ECG telemetry).
  - Prospective validation studies against retrospective ED triage logs (ESI Level 1–5).
