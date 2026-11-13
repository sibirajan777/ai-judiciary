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
                                |  (Parallel Step 1)
                                v
                       [Rebuttal Round]
       - Advocate counters Opposer's strongest point
       - Opposer counters Advocate's strongest point
                                |  (Parallel Step 2)
                                v
               [Dual-Panel Label-Swapped Adjudication]
       - Panel 1: Evaluated with Side A = Pro, Side B = Con
       - Panel 2: Evaluated with Side A = Con, Side B = Pro (Swapped)
       - Both runs executed concurrently in parallel
                                |  (Parallel Step 3)
                                v
               [Consensus Check & Decision Synthesis]
       - If both panels agree: Final verdict with average scores
       - If panels disagree: "Split decision: too close to call" with both rulings displayed
```

1. **Claim Submission**: The user enters a controversial claim or motion (e.g., *"AI should be allowed in exams"*).
2. **Advocate Agent (PRO)**: Presents exactly 3 grounded, persuasive arguments strictly in favor (executed in parallel with Opposer).
3. **Opposer Agent (CON)**: Presents exactly 3 grounded, persuasive arguments strictly against (executed in parallel with Advocate).
4. **Rebuttal Round**: Both sides inspect the opponent's case and formulate targeted rebuttals against each other's strongest point in parallel.
5. **Dual Parallel Judge Verification**: The debate transcript is submitted in parallel to two independent judge evaluations—one with normal labels (Side A = Pro, Side B = Con) and one with swapped labels (Side A = Con, Side B = Pro). If both runs agree on the real side (Pro, Con, or Draw), that is declared the final verdict with averaged scores. If they disagree, the debate is ruled a *"Split decision: too close to call"* with both rulings displayed in full detail.

---

## 🛡️ Anti-Bias & Reliability Guarantees

* **Dual-Panel Position Permutation Testing**: LLMs can suffer from positional and affirmative bias (e.g., favoring the first-listed side). Running dual parallel judge calls with inverted labels (Side A/B swapped) guarantees that a verdict is only awarded when robust against framing effects. Disagreements automatically produce an honest "Split decision: too close to call".
* **Individual 120s Call Timeout**: Timeout is strictly enforced per individual LLM call (120 seconds), not as an arbitrary premature step cutoff.
* **Intelligent Rate Limit (HTTP 429) Handling**: When concurrency thresholds or rate limits are encountered, calls pause with exponential backoff and randomized jitter before retrying up to 3 times per model.
* **Transparent Diagnostics**: Failures surface the real underlying error (HTTP status code, attempt count, model) instead of generic timeout messages.
* **Fine-Grained Execution Timing**: Displays individual call durations (e.g., Advocate vs Opposer) as well as total round elapsed times.
* **Strict JSON Enforcement**: All agents communicate in raw, schema-validated JSON with code-fence stripping and corrective auto-retries.
* **No Fabricated Facts**: Prompts explicitly prohibit inventing statistics, phantom studies, or fake citations.

---

## 📁 Project Structure

| File | Purpose |
| :--- | :--- |
| `main.py` | CLI entry point, concurrent thread pool orchestrator, and formatted terminal renderer with call timings. |
| `agents.py` | Agent role logic (`AdvocateAgent`, `OpposerAgent`, `RebuttalAgent`, `JudgeAgent`), JSON validation, and blinding mechanism. |
| `prompts.py` | Named constants for all system prompts (`ADVOCATE_SYSTEM_PROMPT`, `OPPOSER_SYSTEM_PROMPT`, `REBUTTAL_SYSTEM_PROMPT`, `JUDGE_SYSTEM_PROMPT`). |
| `llm.py` | Unified LLM caller with 120s per-call timeout, HTTP 429 retry backoff, thinking disabled, and multi-model fallback. |
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

### 3. Create a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure API Keys
Copy `.env.example` to `.env`:
```bash
# Windows (PowerShell)
Copy-Item .env.example .env

# macOS / Linux
cp .env.example .env
```

Open `.env` in any text editor and set your preferred provider:

#### For Google Gemini (Default):
```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_MODEL=gemini-3.8-flash
```
*(Get a key from [Google AI Studio](https://aistudio.google.com/))*

#### For Anthropic Claude:
```env
LLM_PROVIDER=claude
ANTHROPIC_API_KEY=your_actual_anthropic_api_key_here
CLAUDE_MODEL=claude-3-5-sonnet-20241022
```
*(Get a key from [Anthropic Console](https://console.anthropic.com/))*

---

## 🎮 Running the Application

### Interactive Mode
Run the script without arguments to be prompted for a claim:
```bash
python main.py
```

### Direct Argument Mode
Pass the claim directly as a command-line argument:
```bash
python main.py "AI should be allowed in exams"
```

---

## 🧪 3 Compelling Test Claims

Try running `ai-judiciary` with these three thought-provoking claims:

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

  [1/3] ⚖️ Formulating Opening Arguments (in parallel)... DONE (Advocate: 8.42s, Opposer: 11.15s | Step: 11.15s)
  [2/3] ⚔️ Commencing Rebuttal Round (both sides in parallel)... DONE (Advocate: 5.20s, Opposer: 6.10s | Step: 6.12s)
  [3/3] 🏛️ Submitting to Dual Judge Panels in parallel (with swapped labels)... DONE (Panel 1: 9.35s, Panel 2: 8.90s | Step: 9.36s)

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
  Note: Evaluated across two independent, blinded parallel judge runs:
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
    • Opening Arguments (Parallel) : Step 11.15s (Advocate: 8.42s, Opposer: 11.15s)
    • Rebuttals (Parallel)         : Step 6.12s (Advocate: 5.20s, Opposer: 6.10s)
    • Dual Judge Panels (Parallel) : Step 9.36s (Panel 1: 9.35s, Panel 2: 8.90s)
    • Total Deliberation Time      : 26.63s

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
