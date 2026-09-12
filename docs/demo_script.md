# TriageFlow AI — Live Demo Script (2.5 Minutes)

> **Format:** Live Hackathon / Technical Showcase Presentation  
> **Total Duration:** 2 minutes, 30 seconds (150 seconds)  
> **Audience:** Judges, Clinical Advisors, and Software Engineers  
> **Presenter Setup:** Streamlit application running on browser (`py -m streamlit run app.py`).

---

### [00:00 – 00:25] 1. Hook & The Core Problem (25 seconds)

**Presenter (Speaking):**
> *"Welcome everyone. Emergency departments worldwide face severe overcrowding and triage bottlenecks. But while generative AI is popular, **allowing a stochastic LLM to hallucinate emergency triage decisions is fundamentally unsafe.**
> 
> Meet **TriageFlow AI**. We built a deterministic, safe, human-in-the-loop triage system using LangGraph and Python. Our core architectural rule is absolute: **The LLM never makes the clinical routing decision.** All risk scoring, contradiction detection, and escalation logic are pure, deterministic Python."*

**Action on Screen:**
- Show the Streamlit header: **TriageFlow AI — Simulated Emergency Triage**.
- Point out the 3-column layout:
  1. *Patient Profile & Data* (Left)
  2. *Active Inquiry & Inputs* (Center)
  3. *Triage Decision & Risk Analysis* (Right)

---

### [00:25 – 01:05] 2. Scenario 1: High-Risk Emergency (40 seconds)

**Presenter (Speaking):**
> *"Let's test our first case: a 68-year-old presenting with acute chest pain. Watch as I click the **'🔴 High-Risk Demo'** button."*

**Action on Screen:**
1. Click **'🔴 High-Risk Demo'**.
2. Click through the steps:
   - **Step 1:** Chest Pain: Present (+2 pts)
   - **Step 2:** Pain Radiation to left arm (+2 pts)
   - **Step 3:** Shortness of Breath (+2 pts)
   - **Step 4:** SpO2 90% — Critical hypoxia (+3 pts)
   - **Step 5:** Heart Rate 118 bpm (+2 pts)
   - **Step 6:** Respiratory Rate 26 bpm (+3 pts)
   - **Step 7:** Systolic BP 88 mmHg — Cardiogenic shock (+3 pts)

**Presenter (Speaking):**
> *"Notice what happened in real-time: with each vital entered, the deterministic scoring engine recalculated the patient's risk profile. The patient hits **18 risk points** with severe hypoxia, tachycardia, tachypnea, and hypotension.
> 
> The decision engine instantly activates the glowing red **EMERGENCY** banner, citing every single contributing clinical factor. Clean data, critical risk — zero hallucination."*

---

### [01:05 – 01:45] 3. Scenario 2: The Safety Net — Contradiction & Escalation (40 seconds)

**Presenter (Speaking):**
> *"Now, what happens if patient statements conflict or data is unreliable? Traditional chatbots might average the inputs or generate confident nonsense. Here is how TriageFlow handles it."*

**Action on Screen:**
1. Click **'⚡ Contradiction Demo'**.
2. Step through the sequence:
   - **Step 1:** Patient reports Shortness of Breath (`True`).
   - **Step 2:** SpO2 is critical at 91% (+3 pts).
   - **Step 6:** Patient confirms Chest Pain (+2 pts).
   - **Step 7:** Patient suddenly denies Shortness of Breath (`False`).

**Presenter (Speaking):**
> *"Watch the screen: in Step 1, the patient affirmed shortness of breath. In Step 7, they contradicted themselves and denied it.
> 
> Instantly, the **Contradiction Detector** flagged a conflict. The decision engine recognized that for high-risk presentations, conflicting clinical inputs cannot be resolved automatically.
> 
> The system switches to **👤 ESCALATE TO HUMAN**. It refuses to guess, halts autonomous routing, and requires bedside nurse review."*

---

### [01:45 – 02:15] 4. Scenario 3: Safe Low-Risk Discharge (30 seconds)

**Presenter (Speaking):**
> *"Finally, let's see how the engine handles non-emergent presentations to prevent unnecessary hospital admissions."*

**Action on Screen:**
1. Click **'🟢 Low-Risk Demo'**.
2. Rapidly step through all normal vitals (SpO2 99%, HR 76, RR 14, SBP 122, Temp 36.8°C, localized muscular pain).

**Presenter (Speaking):**
> *"Here we have a 28-year-old with reproducible musculoskeletal chest wall pain. His vitals are stable, giving only 2 risk points for localized discomfort.
> 
> Once all critical fields are recorded and uncertainty drops below 0.25, the system safely triggers **🟢 DISCHARGE** with self-care advice and safety-net instructions."*

---

### [02:15 – 02:30] 5. Wrap-Up & Auditability (15 seconds)

**Presenter (Speaking):**
> *"Every single cycle, state transition, and question asked is logged into a tamper-evident, ISO-8601 timestamped audit trail, exportable as JSON right here with one click.
> 
> TriageFlow AI proves that AI in emergency healthcare should not replace clinical judgment — it should empower it through transparent, deterministic safety rails. Thank you!"*

**Action on Screen:**
- Click **'📥 Download JSON'** in the Audit Trail section to demonstrate compliance logging.
