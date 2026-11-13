"""
main.py - Entry point for AI Judiciary CLI application

Orchestrates the 5-step courtroom debate flow:
1. User enters or supplies a claim.
2. Advocate agent formulates exactly 3 arguments in favor (sequential to respect rate limits).
3. Opposer agent formulates exactly 3 arguments against (sequential to respect rate limits).
4. Rebuttal round: each side counters the other's strongest argument (sequential).
5. Judge agent evaluates blinded submissions across dual panels with swapped labels (sequential).
6. Results are printed in a clean, structured terminal format with sections:
   - PRO
   - CON
   - REBUTTALS
   - JUDGE'S VERDICT (with scores table and deliberation timing breakdown)
"""

import sys
import os
import time
import shutil
import textwrap
from typing import Dict, Any, Optional, Tuple, Callable
from agents import AdvocateAgent, OpposerAgent, RebuttalAgent, JudgeAgent


def get_terminal_width(default: int = 80) -> int:
    """Returns current terminal width, bounded sensibly for readability."""
    try:
        width = shutil.get_terminal_size((default, 24)).columns
        return min(max(width, 70), 100)
    except Exception:
        return default


def print_banner(width: int):
    """Prints the courtroom banner."""
    bar = "=" * width
    print("\n" + bar)
    print(" ⚖️   AI JUDICIARY : THE COURTROOM OF AUTONOMOUS AGENTS   ⚖️ ".center(width))
    print(" A Structured Adversarial Debate Judged by Impartial Artificial Intelligence ".center(width))
    print(bar + "\n")


def print_header(title: str, width: int, icon: str = "§"):
    """Prints a prominent section header."""
    bar = "-" * width
    print(f"\n{bar}")
    print(f" {icon}  {title.upper()}")
    print(f"{bar}\n")


def wrap_text(
    text: str,
    indent: str = "    ",
    width: int = 80,
    initial_indent: Optional[str] = None,
) -> str:
    """
    Wraps text cleanly at word boundaries using Python's textwrap.
    Preserves multi-paragraph breaks while ensuring each line breaks
    strictly at word boundaries within the allocated terminal width.
    """
    first_indent = initial_indent if initial_indent is not None else indent
    wrapper = textwrap.TextWrapper(
        width=width,
        initial_indent=first_indent,
        subsequent_indent=indent,
        break_long_words=False,
        break_on_hyphens=True,
    )
    paragraphs = str(text).split("\n")
    wrapped_paras = []
    for idx, p in enumerate(paragraphs):
        p_clean = p.strip()
        if p_clean:
            if idx > 0 and initial_indent is not None:
                wrapper.initial_indent = indent
            wrapped_paras.append(wrapper.fill(p_clean))
        else:
            wrapped_paras.append("")
    return "\n".join(wrapped_paras)


def format_arguments_section(section_title: str, role_title: str, args_list: list, width: int):
    """Renders PRO or CON argument section with clear formatting wrapped at word boundaries."""
    print_header(section_title, width, icon="⚖️")
    print(f"  Role: {role_title}\n")

    for i, arg in enumerate(args_list, 1):
        point = arg.get("point", "").strip()
        reason = arg.get("reason", "").strip()
        evidence = arg.get("evidence", "").strip()

        # Wrap thesis cleanly at word boundaries with aligned indentation
        thesis_prefix = f"  [{i}] THESIS: "
        thesis_indent = " " * len(thesis_prefix)
        print(wrap_text(point, indent=thesis_indent, width=width, initial_indent=thesis_prefix))
        print("  " + "─" * (width - 4))
        print("  REASONING:")
        print(wrap_text(reason, indent="    ", width=width))
        print("  EVIDENCE / GROUNDING:")
        print(wrap_text(evidence, indent="    ", width=width))
        print()


def format_rebuttals_section(pro_rebuttal: dict, con_rebuttal: dict, width: int):
    """Renders the rebuttal round section with word-boundary wrapping."""
    print_header("REBUTTALS", width, icon="⚔️")

    # Advocate counter
    print("  ► ADVOCATE (PRO) REBUTTAL:")
    pro_target = pro_rebuttal.get("target_point", "").strip()
    target_prefix = '    Targeted Opponent Point : "'
    target_indent = "                              "
    print(wrap_text(f'{pro_target}"', indent=target_indent, width=width, initial_indent=target_prefix))
    print("    Counter-Argument        :")
    print(wrap_text(pro_rebuttal.get("rebuttal", "").strip(), indent="      ", width=width))
    print()

    # Opposer counter
    print("  ► OPPOSER (CON) REBUTTAL:")
    con_target = con_rebuttal.get("target_point", "").strip()
    print(wrap_text(f'{con_target}"', indent=target_indent, width=width, initial_indent=target_prefix))
    print("    Counter-Argument        :")
    print(wrap_text(con_rebuttal.get("rebuttal", "").strip(), indent="      ", width=width))
    print()


def fmt_score(val) -> str:
    """Formats integer or float score values cleanly."""
    try:
        fval = float(val)
        if fval.is_integer():
            return str(int(fval))
        return f"{fval:.1f}"
    except Exception:
        return str(val)


def format_scores_table(pro_scores: dict, con_scores: dict, width: int, title: str = "JUDICIAL SCORECARD:"):
    """Draws a clean, centered Unicode/ASCII table for criteria scores."""
    pro_ev = pro_scores.get("evidence", 0)
    pro_lo = pro_scores.get("logic", 0)
    pro_re = pro_scores.get("relevance", 0)
    pro_total = pro_ev + pro_lo + pro_re

    con_ev = con_scores.get("evidence", 0)
    con_lo = con_scores.get("logic", 0)
    con_re = con_scores.get("relevance", 0)
    con_total = con_ev + con_lo + con_re

    col1_w = 26
    col2_w = 16
    col3_w = 16

    top_border = f"  ┌{'─' * col1_w}┬{'─' * col2_w}┬{'─' * col3_w}┐"
    hdr_line   = f"  │ {'CRITERION (0-10)':<{col1_w - 2}} │ {'ADVOCATE (PRO)':^{col2_w - 2}} │ {'OPPOSER (CON)':^{col3_w - 2}} │"
    mid_border = f"  ├{'─' * col1_w}┼{'─' * col2_w}┼{'─' * col3_w}┤"
    row_ev     = f"  │ {'Evidence & Truth':<{col1_w - 2}} │ {f'{fmt_score(pro_ev)}/10':^{col2_w - 2}} │ {f'{fmt_score(con_ev)}/10':^{col3_w - 2}} │"
    row_lo     = f"  │ {'Logic & Reasoning':<{col1_w - 2}} │ {f'{fmt_score(pro_lo)}/10':^{col2_w - 2}} │ {f'{fmt_score(con_lo)}/10':^{col3_w - 2}} │"
    row_re     = f"  │ {'Relevance & Rebuttal':<{col1_w - 2}} │ {f'{fmt_score(pro_re)}/10':^{col2_w - 2}} │ {f'{fmt_score(con_re)}/10':^{col3_w - 2}} │"
    sep_border = f"  ├{'─' * col1_w}┼{'─' * col2_w}┼{'─' * col3_w}┤"
    row_tot    = f"  │ {'TOTAL SCORE (out of 30)':<{col1_w - 2}} │ {f'{fmt_score(pro_total)}/30':^{col2_w - 2}} │ {f'{fmt_score(con_total)}/30':^{col3_w - 2}} │"
    bot_border = f"  └{'─' * col1_w}┴{'─' * col2_w}┴{'─' * col3_w}┘"

    print(f"  {title}")
    print(top_border)
    print(hdr_line)
    print(mid_border)
    print(row_ev)
    print(row_lo)
    print(row_re)
    print(sep_border)
    print(row_tot)
    print(bot_border)
    print()


def format_verdict_section(
    run1: Dict[str, Any],
    run2: Dict[str, Any],
    width: int,
    timings: Optional[Dict[str, float]] = None,
):
    """
    Renders the final verdict based on the dual judge runs.
    - If both runs pick the same real side (Pro or Con or Draw):
      Prints that as the final verdict with the average scores.
    - If the two runs disagree:
      Prints 'Split decision: too close to call' and displays both rulings.
    """
    v1 = run1["mapped"]["verdict"].lower()  # "pro", "con", or "draw"
    v2 = run2["mapped"]["verdict"].lower()  # "pro", "con", or "draw"

    def label_for_verdict(v_code: str) -> str:
        if v_code == "pro":
            return "ADVOCATE (PRO) PREVAILS"
        elif v_code == "con":
            return "OPPOSER (CON) PREVAILS"
        else:
            return "DEBATE DECLARED A DRAW"

    if v1 == v2:
        # Both runs agreed on the same real side!
        print_header("JUDGE'S VERDICT : UNANIMOUS CONSENSUS", width, icon="🏛️")
        print("  Note: Evaluated across two independent, blinded judge runs:")
        print("    • Panel 1 : Evaluated with Side A = Advocate (PRO), Side B = Opposer (CON)")
        print("    • Panel 2 : Evaluated with Side A = Opposer (CON), Side B = Advocate (PRO) [Labels Swapped]")
        print(f"    Both independent runs confirmed the same verdict: {label_for_verdict(v1)}.\n")

        # Compute average scores
        r1_pro = run1["mapped"]["pro_scores"]
        r2_pro = run2["mapped"]["pro_scores"]
        r1_con = run1["mapped"]["con_scores"]
        r2_con = run2["mapped"]["con_scores"]

        avg_pro_scores = {
            k: (r1_pro.get(k, 0) + r2_pro.get(k, 0)) / 2.0
            for k in ("evidence", "logic", "relevance")
        }
        avg_con_scores = {
            k: (r1_con.get(k, 0) + r2_con.get(k, 0)) / 2.0
            for k in ("evidence", "logic", "relevance")
        }

        # Average Scorecard Table
        format_scores_table(
            avg_pro_scores,
            avg_con_scores,
            width,
            title="AVERAGE JUDICIAL SCORECARD (Across Swapped Panels):",
        )

        # Bench recognition
        print("  BENCH RECOGNITION:")
        adv_prefix = "    • Advocate's Strongest Point : \""
        adv_indent = "                                   "
        print(wrap_text(f'{run1["mapped"]["strongest_pro"].strip()}"', indent=adv_indent, width=width, initial_indent=adv_prefix))

        opp_prefix = "    • Opposer's Strongest Point  : \""
        opp_indent = "                                   "
        print(wrap_text(f'{run1["mapped"]["strongest_con"].strip()}"', indent=opp_indent, width=width, initial_indent=opp_prefix))
        print()

        # Timing breakdown if available
        if timings:
            print("  DELIBERATION TIMING:")
            print(f"    • Step 1: Opening Arguments : Total {timings.get('step1_total', 0.0):.2f}s (Advocate: {timings.get('adv_open', 0.0):.2f}s, Opposer: {timings.get('opp_open', 0.0):.2f}s)")
            print(f"    • Step 2: Rebuttals         : Total {timings.get('step2_total', 0.0):.2f}s (Advocate: {timings.get('adv_reb', 0.0):.2f}s, Opposer: {timings.get('opp_reb', 0.0):.2f}s)")
            print(f"    • Step 3: Dual Judge Panels : Total {timings.get('step3_total', 0.0):.2f}s (Panel 1: {timings.get('judge_panel1', 0.0):.2f}s, Panel 2: {timings.get('judge_panel2', 0.0):.2f}s)")
            print(f"    • Total Deliberation Time   : {timings.get('total', 0.0):.2f}s\n")

        # Consensus Verdict Badge
        box_w = width - 4
        print("  " + "═" * box_w)
        verdict_str = f"🏆 FINAL VERDICT: {label_for_verdict(v1)}"
        print(f"  {verdict_str.center(box_w)}")
        print("  " + "═" * box_w + "\n")

        # Judicial opinions
        print("  JUDICIAL OPINION (PANEL 1 - NORMAL LABELS):")
        print(wrap_text(run1["mapped"]["reasoning"], indent="    ", width=width))
        print()
        if run1["mapped"]["reasoning"].strip() != run2["mapped"]["reasoning"].strip():
            print("  JUDICIAL OPINION (PANEL 2 - SWAPPED LABELS):")
            print(wrap_text(run2["mapped"]["reasoning"], indent="    ", width=width))
            print()
        print("=" * width + "\n")

    else:
        # Disagreement between the two runs!
        print_header("JUDGE'S VERDICT : SPLIT DECISION", width, icon="⚖️")
        print("  Note: Evaluated across two independent, blinded judge runs with swapped labels:")
        print(f"    • Panel 1 (Side A = PRO, Side B = CON) : Ruled '{label_for_verdict(v1)}'")
        print(f"    • Panel 2 (Side A = CON, Side B = PRO) : Ruled '{label_for_verdict(v2)}'")
        print("  When the presentation order was flipped, the two runs reached conflicting conclusions.\n")

        # Verdict Badge explicitly matching user requirement:
        box_w = width - 4
        print("  " + "═" * box_w)
        split_title = "Split decision: too close to call"
        print(f"  {split_title.center(box_w)}")
        print("  " + "═" * box_w + "\n")

        # Timing breakdown if available
        if timings:
            print("  DELIBERATION TIMING:")
            print(f"    • Step 1: Opening Arguments : Total {timings.get('step1_total', 0.0):.2f}s (Advocate: {timings.get('adv_open', 0.0):.2f}s, Opposer: {timings.get('opp_open', 0.0):.2f}s)")
            print(f"    • Step 2: Rebuttals         : Total {timings.get('step2_total', 0.0):.2f}s (Advocate: {timings.get('adv_reb', 0.0):.2f}s, Opposer: {timings.get('opp_reb', 0.0):.2f}s)")
            print(f"    • Step 3: Dual Judge Panels : Total {timings.get('step3_total', 0.0):.2f}s (Panel 1: {timings.get('judge_panel1', 0.0):.2f}s, Panel 2: {timings.get('judge_panel2', 0.0):.2f}s)")
            print(f"    • Total Deliberation Time   : {timings.get('total', 0.0):.2f}s\n")

        # Show Both Rulings
        print("  " + "─" * (width - 4))
        print(f"  🏛️ RULING 1 (Panel 1: Side A = Advocate, Side B = Opposer) -> {label_for_verdict(v1)}")
        print("  " + "─" * (width - 4))
        format_scores_table(
            run1["mapped"]["pro_scores"],
            run1["mapped"]["con_scores"],
            width,
            title="PANEL 1 SCORECARD:",
        )
        print("  Panel 1 Recognized Strongest Points:")
        print(f"    • Advocate : \"{run1['mapped']['strongest_pro'].strip()}\"")
        print(f"    • Opposer  : \"{run1['mapped']['strongest_con'].strip()}\"\n")
        print("  Panel 1 Judicial Opinion:")
        print(wrap_text(run1["mapped"]["reasoning"], indent="    ", width=width))
        print("\n")

        print("  " + "─" * (width - 4))
        print(f"  🏛️ RULING 2 (Panel 2: Side A = Opposer, Side B = Advocate [Swapped]) -> {label_for_verdict(v2)}")
        print("  " + "─" * (width - 4))
        format_scores_table(
            run2["mapped"]["pro_scores"],
            run2["mapped"]["con_scores"],
            width,
            title="PANEL 2 SCORECARD:",
        )
        print("  Panel 2 Recognized Strongest Points:")
        print(f"    • Advocate : \"{run2['mapped']['strongest_pro'].strip()}\"")
        print(f"    • Opposer  : \"{run2['mapped']['strongest_con'].strip()}\"\n")
        print("  Panel 2 Judicial Opinion:")
        print(wrap_text(run2["mapped"]["reasoning"], indent="    ", width=width))
        print("\n" + "=" * width + "\n")


def run_timed_task(task_fn: Callable, *args) -> Tuple[Any, float]:
    """Executes a function and returns (result, elapsed_seconds)."""
    start_time = time.perf_counter()
    result = task_fn(*args)
    duration = time.perf_counter() - start_time
    return result, duration


def run_debate(claim: str):
    """Executes the courtroom debate pipeline sequentially to protect free-tier rate limits."""
    width = get_terminal_width()
    print_banner(width)

    topic_prefix = '  CASE TOPIC : "'
    topic_indent = "               "
    print(wrap_text(f'{claim}"', indent=topic_indent, width=width, initial_indent=topic_prefix))
    print("  STATUS     : Proceedings initialized. All agents summoned to court.\n")

    total_start = time.perf_counter()

    # Step 1: Advocate and Opposer opening arguments (Sequential to respect rate limits)
    print("  [1/3] ⚖️ Formulating Opening Arguments (Sequential to respect rate limits)...")

    # 1a. Advocate opening arguments
    print("        ► Advocate Agent (PRO)... ", end="", flush=True)
    try:
        pro_case, t_adv = run_timed_task(AdvocateAgent.argue, claim)
        print(f"DONE ({t_adv:.2f}s)")
    except Exception as exc:
        print("FAILED\n")
        print("  " + "=" * 68)
        print("  [!] JUDICIAL PROCEEDINGS HALTED: ADVOCATE AGENT CALL FAILURE")
        print("  " + "=" * 68)
        print(f"  • Underlying Error : {exc}\n")
        sys.exit(1)

    # 1b. Opposer opening arguments
    print("        ► Opposer Agent (CON)... ", end="", flush=True)
    try:
        con_case, t_opp = run_timed_task(OpposerAgent.argue, claim)
        print(f"DONE ({t_opp:.2f}s)")
    except Exception as exc:
        print("FAILED\n")
        print("  " + "=" * 68)
        print("  [!] JUDICIAL PROCEEDINGS HALTED: OPPOSER AGENT CALL FAILURE")
        print("  " + "=" * 68)
        print(f"  • Underlying Error : {exc}\n")
        sys.exit(1)

    t_open = t_adv + t_opp
    pro_arguments = pro_case.get("arguments", [])
    con_arguments = con_case.get("arguments", [])

    # Step 2: Rebuttal Round (Sequential)
    print("  [2/3] ⚔️ Commencing Rebuttal Round (Sequential)...")

    # 2a. Advocate rebuttal
    print("        ► Advocate (PRO) Rebuttal... ", end="", flush=True)
    try:
        pro_rebuttal, t_adv_reb = run_timed_task(
            RebuttalAgent.rebut,
            "Advocate",
            claim,
            "in favor",
            con_arguments,
        )
        print(f"DONE ({t_adv_reb:.2f}s)")
    except Exception as exc:
        print("FAILED\n")
        print("  " + "=" * 68)
        print("  [!] JUDICIAL PROCEEDINGS HALTED: ADVOCATE REBUTTAL FAILURE")
        print("  " + "=" * 68)
        print(f"  • Underlying Error : {exc}\n")
        sys.exit(1)

    # 2b. Opposer rebuttal
    print("        ► Opposer (CON) Rebuttal... ", end="", flush=True)
    try:
        con_rebuttal, t_opp_reb = run_timed_task(
            RebuttalAgent.rebut,
            "Opposer",
            claim,
            "against",
            pro_arguments,
        )
        print(f"DONE ({t_opp_reb:.2f}s)")
    except Exception as exc:
        print("FAILED\n")
        print("  " + "=" * 68)
        print("  [!] JUDICIAL PROCEEDINGS HALTED: OPPOSER REBUTTAL FAILURE")
        print("  " + "=" * 68)
        print(f"  • Underlying Error : {exc}\n")
        sys.exit(1)

    t_reb = t_adv_reb + t_opp_reb

    # Step 3: Presiding Judge evaluation (Dual sequential panels with swapped labels)
    print("  [3/3] 🏛️ Submitting to Dual Judge Panels (Sequential with swapped labels)...")

    # 3a. Panel 1 (Normal labels)
    print("        ► Judge Panel 1 (Side A = PRO, Side B = CON)... ", end="", flush=True)
    try:
        run1, t_j1 = run_timed_task(
            JudgeAgent.evaluate,
            claim,
            pro_arguments,
            con_arguments,
            pro_rebuttal,
            con_rebuttal,
            True,
        )
        print(f"DONE ({t_j1:.2f}s)")
    except Exception as exc:
        print("FAILED\n")
        print("  " + "=" * 68)
        print("  [!] JUDICIAL PROCEEDINGS HALTED: JUDGE PANEL 1 FAILURE")
        print("  " + "=" * 68)
        print(f"  • Underlying Error : {exc}\n")
        sys.exit(1)

    # 3b. Panel 2 (Swapped labels)
    print("        ► Judge Panel 2 (Side A = CON, Side B = PRO [Swapped])... ", end="", flush=True)
    try:
        run2, t_j2 = run_timed_task(
            JudgeAgent.evaluate,
            claim,
            pro_arguments,
            con_arguments,
            pro_rebuttal,
            con_rebuttal,
            False,
        )
        print(f"DONE ({t_j2:.2f}s)\n")
    except Exception as exc:
        print("FAILED\n")
        print("  " + "=" * 68)
        print("  [!] JUDICIAL PROCEEDINGS HALTED: JUDGE PANEL 2 FAILURE")
        print("  " + "=" * 68)
        print(f"  • Underlying Error : {exc}\n")
        sys.exit(1)

    t_judge = t_j1 + t_j2

    total_time = time.perf_counter() - total_start
    timings = {
        "step1_total": t_open,
        "adv_open": t_adv,
        "opp_open": t_opp,
        "step2_total": t_reb,
        "adv_reb": t_adv_reb,
        "opp_reb": t_opp_reb,
        "step3_total": t_judge,
        "judge_panel1": t_j1,
        "judge_panel2": t_j2,
        "total": total_time,
    }

    # Step 4: Render results
    format_arguments_section(
        section_title="PRO",
        role_title="Advocate (Arguing strictly in favor)",
        args_list=pro_arguments,
        width=width,
    )

    format_arguments_section(
        section_title="CON",
        role_title="Opposer (Arguing strictly against)",
        args_list=con_arguments,
        width=width,
    )

    format_rebuttals_section(
        pro_rebuttal=pro_rebuttal,
        con_rebuttal=con_rebuttal,
        width=width,
    )

    format_verdict_section(
        run1=run1,
        run2=run2,
        width=width,
        timings=timings,
    )


def main():
    """Main CLI entrypoint."""
    # Check for claim passed as CLI argument
    if len(sys.argv) > 1:
        # Join arguments in case user passed unquoted string
        claim = " ".join(sys.argv[1:]).strip()
    else:
        # Interactive prompt
        print_banner(get_terminal_width())
        try:
            claim = input("Enter debate claim (e.g., 'AI should be allowed in exams'): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSession aborted by user.")
            sys.exit(0)

    if not claim:
        print("Error: Claim cannot be empty.")
        print("Usage: python main.py [\"claim text\"]")
        sys.exit(1)

    # Strip redundant outer quotes if entered
    claim = claim.strip().strip("'\"").strip()

    run_debate(claim)


if __name__ == "__main__":
    main()
