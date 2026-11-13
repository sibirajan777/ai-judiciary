# ⚖️ AI Judiciary: Courtroom-Style Multi-Agent Debate CLI

**AI Judiciary** is a Python CLI application that conducts structured, adversarial courtroom debates between specialized AI agents, adjudicated by an impartial AI Judge.

---

## 🏛️ Debate Flow

```
                      +-------------------+
                      |    User Claim     |
                      +---------+---------+
                                |
             +------------------+------------------+
             |                                     |
             v                                     v
   [Advocate Agent (PRO)]                [Opposer Agent (CON)]
   - Exactly 3 arguments                 - Exactly 3 arguments
   - Honest, grounded, no fake stats     - Honest, grounded, no fake stats
   - 120s timeout per call               - 120s timeout per call
             |                                     |
             +------------------+------------------+
                                |  (Sequential Step 1 - Rate-Limit Safe)
                                v
                       [Rebuttal Round]
       - Advocate counters Opposer's strongest point
       - Opposer counters Advocate's strongest point
                                |  (Sequential Step 2 - Rate-Limit Safe)
                                v
               [Dual-Panel Label-Swapped Adjudication]
       - Panel 1: Evaluated with Side A = Pro, Side B = Con
       - Panel 2: Evaluated with Side A = Con, Side B = Pro (Swapped)
                                |  (Sequential Step 3 - Rate-Limit Safe)
                                v
               [Consensus Check & Decision Synthesis]
       - If both panels agree: Final verdict with average scores
       - If panels disagree: "Split decision: too close to call" with both rulings displayed
```

1. **Claim Submission**: The user enters a controversial claim or motion (e.g., *"AI should be allowed in exams"*).
2. **Advocate Agent (PRO)**: Formulates exactly 3 grounded, persuasive arguments strictly in favor.
3. **Opposer Agent (CON)**: Formulates exactly 3 grounded, persuasive arguments strictly against (executed sequentially to preserve free-tier rate limits).
4. **Rebuttal Round**: Both sides inspect the opponent's case and formulate targeted rebuttals against each other's strongest point sequentially.
5. **Dual Judge Verification**: The debate transcript is submitted sequentially to two independent judge evaluations—one with normal labels (Side A = Pro, Side B = Con) and one with swapped labels (Side A = Con, Side B = Pro). If both runs agree on the real side (Pro, Con, or Draw), that is declared the final verdict with averaged scores. If they disagree, the debate is ruled a *"Split decision: too close to call"* with both rulings displayed in full detail.

---

## 🛡️ Anti-Bias & Reliability Guarantees

* **Sequential Free-Tier Safety**: Agent calls run sequentially to prevent burst concurrency limits on Google AI Studio free tier keys.
* **Dual-Panel Position Permutation Testing**: Running dual judge calls with inverted labels (Side A/B swapped) guarantees that a verdict is only awarded when robust against position bias. Disagreements automatically produce an honest "Split decision: too close to call".
* **Clean Configuration**: Thinking config and unsupported parameters are removed to prevent HTTP 400 `INVALID_ARGUMENT` errors. The exact request config is logged before each call.
* **Capped Retries & 10-30s Backoff**: Retries are strictly capped at 3 per call. On HTTP 429 (Rate Limit / Quota Exhaustion), execution pauses with a 10 to 30-second backoff before retrying. HTTP 400 errors are never retried and fail fast with clear diagnostic info.
* **Strict JSON Enforcement**: All agents communicate in raw, schema-validated JSON with code-fence stripping and corrective auto-retries.
* **No Fabricated Facts**: Prompts explicitly prohibit inventing statistics, phantom studies, or fake citations.

---

## 📁 Project Structure

| File | Purpose |
| :--- | :--- |
| `main.py` | CLI entry point, sequential rate-limit-safe orchestrator, and formatted terminal renderer with call timings. |
| `agents.py` | Agent role logic (`AdvocateAgent`, `OpposerAgent`, `RebuttalAgent`, `JudgeAgent`), JSON validation, and blinding mechanism. |
| `prompts.py` | Named constants for all system prompts (`ADVOCATE_SYSTEM_PROMPT`, `OPPOSER_SYSTEM_PROMPT`, `REBUTTAL_SYSTEM_PROMPT`, `JUDGE_SYSTEM_PROMPT`). |
| `llm.py` | Unified LLM caller with request config logging, 120s timeout, capped 3 retries, 10-30s backoff for 429, and immediate 400 failure. |
| `requirements.txt`| Python dependencies (`google-genai`, `anthropic`, `python-dotenv`). |
| `.env.example` | Template for configuring API keys and provider selection. |
| `README.md` | Documentation, setup guide, and example claims. |

---

## 🚀 Setup & Installation

### 1. Prerequisites
Ensure you have Python 3.9+ installed.

### 2. Clone / Navigate to Directory
```bash
cd d:\ai-judiciary
```

### 3. Activate Virtual Environment
```bash
# Windows
.\venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure API Keys
Verify `.env` has your valid API key and model:
```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_MODEL=gemini-3.8-flash
```

---

## 🎮 Running the Application

### Interactive Mode
```bash
python main.py
```

### Direct Argument Mode
```bash
python main.py "AI should be allowed in exams"
```

---

## 🧪 3 Compelling Test Claims

### Test Claim 1: Education & Technology
> **`"AI should be allowed in university examinations."`**
* **Advocate Focus**: Real-world job alignment, shifting from rote memorization to high-order synthesis, equalizing access to tutoring.
* **Opposer Focus**: Erosion of foundational cognitive abilities, unverifiable individual competence, academic dishonesty, and digital divide.

### Test Claim 2: Governance & Automation
> **`"Governments should replace human traffic judges with automated algorithmic sentencing for traffic violations."`**
* **Advocate Focus**: Elimination of human bias, 100% consistent penalty enforcement, dramatic reduction in court backlogs and taxpayer costs.
* **Opposer Focus**: Lack of equitable discretion for edge cases, sensor calibration errors, algorithmic opacity, and violation of the constitutional right to face one's accuser.

### Test Claim 3: Intellectual Property & Creative Arts
> **`"AI-generated artwork should receive copyright protection under the law."`**
* **Advocate Focus**: Incentive structures for human prompt engineers and creative directors, recognition of labor in model curation, economic utility.
* **Opposer Focus**: The human authorship doctrine, risk of monopolizing artistic expressions via mass generation, dilution of human creative culture.

---

## 📊 Sample Terminal Output

```
================================================================================
          AI JUDICIARY : THE COURTROOM OF AUTONOMOUS AGENTS           
================================================================================

  CASE TOPIC : "AI should be allowed in university examinations."
  STATUS     : Proceedings initialized. All agents summoned to court.

  [1/3] ⚖️ Formulating Opening Arguments (Sequential to respect rate limits)...
        ► Advocate Agent (PRO)... DONE (7.42s)
        ► Opposer Agent (CON)... DONE (8.15s)
  [2/3] ⚔️ Commencing Rebuttal Round (Sequential)...
        ► Advocate (PRO) Rebuttal... DONE (5.20s)
        ► Opposer (CON) Rebuttal... DONE (6.10s)
  [3/3] 🏛️ Submitting to Dual Judge Panels (Sequential with swapped labels)...
        ► Judge Panel 1 (Side A = PRO, Side B = CON)... DONE (9.35s)
        ► Judge Panel 2 (Side A = CON, Side B = PRO [Swapped])... DONE (8.90s)

--------------------------------------------------------------------------------
 ⚖️  PRO
--------------------------------------------------------------------------------
  Role: Advocate (Arguing strictly in favor)

  [1] THESIS: Real-World Professional Alignment
  ────────────────────────────────────────────────────────────────────────────
  REASONING:
    Modern workplaces require collaborative problem-solving using AI tools.
    Exams should evaluate candidates under authentic working conditions rather
    than artificial isolation.
  EVIDENCE / GROUNDING:
    Engineering, finance, and software development standards increasingly
    integrate automated copilot systems into standard operating procedure.

...

--------------------------------------------------------------------------------
 🏛️  JUDGE'S VERDICT : UNANIMOUS CONSENSUS
--------------------------------------------------------------------------------
  Note: Evaluated across two independent, blinded judge runs:
    • Panel 1 : Evaluated with Side A = Advocate (PRO), Side B = Opposer (CON)
    • Panel 2 : Evaluated with Side A = Opposer (CON), Side B = Advocate (PRO) [Labels Swapped]
    Both independent runs confirmed the same verdict: OPPOSER (CON) PREVAILS.

  AVERAGE JUDICIAL SCORECARD (Across Swapped Panels):
  ┌──────────────────────────┬────────────────┬────────────────┐
  │ CRITERION (0-10)         │ ADVOCATE (PRO) │ OPPOSER (CON)  │
  ├──────────────────────────┼────────────────┼────────────────┤
  │ Evidence & Truth         │      8/10      │      9/10      │
  │ Logic & Reasoning        │      8/10      │      9/10      │
  │ Relevance & Rebuttal     │      9/10      │      8/10      │
  ├──────────────────────────┼────────────────┼────────────────┤
  │ TOTAL SCORE (out of 30)  │     25/30      │     26/30      │
  └──────────────────────────┴────────────────┴────────────────┘

  BENCH RECOGNITION:
    • Advocate's Strongest Point : "Real-World Professional Alignment"
    • Opposer's Strongest Point  : "Erosion of Core Cognitive Competencies"

  DELIBERATION TIMING:
    • Step 1: Opening Arguments : Total 15.57s (Advocate: 7.42s, Opposer: 8.15s)
    • Step 2: Rebuttals         : Total 11.30s (Advocate: 5.20s, Opposer: 6.10s)
    • Step 3: Dual Judge Panels : Total 18.25s (Panel 1: 9.35s, Panel 2: 8.90s)
    • Total Deliberation Time   : 45.12s

  ══════════════════════════════════════════════════════════════════════════════
                   🏆 FINAL VERDICT: OPPOSER (CON) PREVAILS                     
  ══════════════════════════════════════════════════════════════════════════════

  JUDICIAL OPINION (PANEL 1 - NORMAL LABELS):
    While Side B (Advocate) presented a forward-looking economic rationale,
    Side A (Opposer) successfully demonstrated that certification of baseline
    human competence cannot occur when the cognitive load is outsourced.
    Side A's rebuttal effectively dismantled Side B's vocational premise.
================================================================================
```
