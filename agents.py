"""
agents.py - AI Judiciary Agent Definitions and Orchestration

This module implements the core agent roles in the courtroom-style debate:
1. AdvocateAgent: Argues strictly in favor of the claim (exactly 3 arguments).
2. OpposerAgent: Argues strictly against the claim (exactly 3 arguments).
3. RebuttalAgent: Handles the rebuttal round where each side counters the opponent's strongest argument.
4. JudgeAgent: Impartial judge that evaluates blinded submissions ("Side A" vs "Side B")
   without knowing which is Pro or Con, delivers scores (0-10), and rules a verdict.

Includes safe JSON parsing with code-fence stripping and up to 2 retries on malformed output.
"""

import re
import json
import random
import logging
from typing import Dict, Any, Callable, Optional, Tuple
from llm import call_llm
from prompts import (
    ADVOCATE_SYSTEM_PROMPT,
    OPPOSER_SYSTEM_PROMPT,
    REBUTTAL_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


# ============================================================================
# SAFE JSON PARSING & RETRY MECHANISM
# ============================================================================

def strip_code_fences_and_extract(text: str) -> str:
    """
    Cleans raw LLM response text by stripping markdown code fences (```json ... ```)
    and isolating the outermost JSON object or array delimited by { } or [ ].
    Handles multiple code blocks, nested structures, and edge cases.
    """
    if not text:
        return ""

    cleaned = text.strip()

    # Remove all markdown code block markers (handles multiple blocks)
    # Pattern matches ```json, ```javascript, ``` or just ```
    cleaned = re.sub(r"^```\w*\s*", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)
    # Also remove any remaining ``` markers in the middle
    cleaned = re.sub(r"```", "", cleaned)
    cleaned = cleaned.strip()

    # Find the outermost JSON structure bounds (object or array)
    obj_start = cleaned.find("{")
    obj_end = cleaned.rfind("}")
    arr_start = cleaned.find("[")
    arr_end = cleaned.rfind("]")

    # Determine which structure comes first and is complete
    candidates = []
    if obj_start != -1 and obj_end != -1 and obj_end >= obj_start:
        candidates.append((obj_start, obj_end + 1, "object"))
    if arr_start != -1 and arr_end != -1 and arr_end >= arr_start:
        candidates.append((arr_start, arr_end + 1, "array"))

    if candidates:
        # Pick the one that starts earliest
        candidates.sort(key=lambda x: x[0])
        start, end, _ = candidates[0]
        cleaned = cleaned[start:end]

    return cleaned


def parse_json_safely(raw_text: str) -> Dict[str, Any]:
    """
    Extracts and parses JSON from raw LLM output.
    Raises ValueError with descriptive message if parsing fails.
    Uses efficient parsing strategies without O(n²) substring search.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("LLM returned an empty response.")

    cleaned = strip_code_fences_and_extract(raw_text)
    if not cleaned:
        raise ValueError("No JSON object or array found in LLM response after cleaning.")

    # Strategy 1: Direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.debug(f"Direct JSON parse failed: {exc.msg}, trying recovery strategies")

    # Strategy 2: Fix common JSON issues
    fixed = _attempt_json_repair(cleaned)
    if fixed:
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

    # Strategy 3: Try json5 for more lenient parsing (handles trailing commas, comments, etc.)
    try:
        import json5
        return json5.loads(cleaned)
    except ImportError:
        pass
    except Exception:
        pass

    # Strategy 4: Fallback - try to find valid JSON using brace matching (O(n) not O(n²))
    # Only attempt this for reasonably sized responses
    if len(cleaned) < 50000:
        for brace_type in ('{}', '[]'):
            open_brace, close_brace = brace_type[0], brace_type[1]
            start = cleaned.find(open_brace)
            if start == -1:
                continue
            # Use a stack-based approach to find matching closing brace
            stack = 0
            for i, ch in enumerate(cleaned[start:], start):
                if ch == open_brace:
                    stack += 1
                elif ch == close_brace:
                    stack -= 1
                    if stack == 0:
                        # Found matching pair
                        substring = cleaned[start:i+1]
                        try:
                            parsed = json.loads(substring)
                            if isinstance(parsed, (dict, list)):
                                logger.debug("Recovered JSON using brace matching")
                                return parsed if isinstance(parsed, dict) else {"data": parsed}
                        except json.JSONDecodeError:
                            pass
                        break

    raise ValueError(
        f"JSON syntax error after all recovery attempts. "
        f"Cleaned output preview: {cleaned[:200]}..."
    )


def _attempt_json_repair(text: str) -> Optional[str]:
    """Attempt to fix common JSON formatting issues."""
    # Fix trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    # Fix missing quotes around keys (simple cases)
    text = re.sub(r"([{,]\s*)(\w+)(\s*:)", r'\1"\2"\3', text)
    # Fix single quotes to double quotes for string values (not inside already quoted strings)
    # This handles cases like: {'key': 'value'} -> {"key": "value"}
    # We only do this for simple cases to avoid breaking already valid JSON
    # Match single-quoted strings that are not escaped
    text = re.sub(r"'([^'\\]*(?:\\.[^'\\]*)*)'", r'"\1"', text)
    return text


def call_agent_with_retry(
    system_prompt: str,
    user_message: str,
    schema_description: str,
    validator_fn: Optional[Callable[[Dict[str, Any]], Tuple[bool, str]]] = None,
    max_retries: int = 2,
) -> Dict[str, Any]:
    """
    Calls the LLM and parses JSON output safely.
    If parsing or schema validation fails, retries up to `max_retries` times
    with corrective feedback, then raises RuntimeError if all attempts fail.
    Network and API authentication errors are raised immediately without false retries.
    """
    current_prompt = user_message
    last_error = ""
    last_raw_response = ""

    for attempt in range(max_retries + 1):
        # 1. Call the LLM (API/network errors bubble up directly with clear diagnosis)
        raw_response = call_llm(system_prompt, current_prompt)
        last_raw_response = raw_response

        # 2. Parse and validate JSON
        try:
            data = parse_json_safely(raw_response)
            if validator_fn:
                valid, error_msg = validator_fn(data)
                if not valid:
                    raise ValueError(error_msg)
            return data
        except Exception as parse_exc:
            last_error = str(parse_exc)
            if attempt < max_retries:
                current_prompt = (
                    f"{user_message}\n\n"
                    f"[CORRECTIVE NOTICE - ATTEMPT {attempt + 2}/{max_retries + 1}]\n"
                    f"Your previous response was rejected due to JSON formatting error: {last_error}\n"
                    f"You must output STRICT JSON ONLY matching this specification:\n{schema_description}\n"
                    "Do NOT include markdown formatting (no ```json code blocks), no explanation, only raw JSON."
                )

    raise RuntimeError(
        f"Failed to obtain valid JSON from agent after {max_retries + 1} attempts.\n"
        f"Last parsing error: {last_error}\n"
        f"Last raw output snippet: {last_raw_response[:300]}"
    )


def call_agent_with_retry_vision(
    system_prompt: str,
    user_message: str,
    image_base64: str,
    image_mime_type: str,
    schema_description: str,
    validator_fn: Optional[Callable[[Dict[str, Any]], Tuple[bool, str]]] = None,
    max_retries: int = 2,
    max_tokens: int = None,
) -> Dict[str, Any]:
    """
    Calls the vision LLM and parses JSON output safely.
    If parsing or schema validation fails, retries up to `max_retries` times
    with corrective feedback, then raises RuntimeError if all attempts fail.
    
    Handles truncation (finish_reason=length) by retrying once with increased token budget.
    """
    from llm import call_llm_vision, NVIDIA_VISION_MAX_TOKENS

    current_prompt = user_message
    last_error = ""
    last_raw_response = ""
    current_max_tokens = max_tokens
    truncation_retry_done = False

    for attempt in range(max_retries + 1):
        # 1. Call the vision LLM
        try:
            raw_response = call_llm_vision(system_prompt, current_prompt, image_base64, image_mime_type, max_tokens=current_max_tokens)
        except RuntimeError as e:
            # Handle truncation by retrying with increased token budget (once)
            if "truncated (finish_reason=length)" in str(e) and not truncation_retry_done and current_max_tokens < NVIDIA_VISION_MAX_TOKENS:
                # Increase token budget by 50% up to global max
                current_max_tokens = min(int(current_max_tokens * 1.5), NVIDIA_VISION_MAX_TOKENS)
                logger.warning(f"Response truncated, retrying with increased token budget: {current_max_tokens}")
                truncation_retry_done = True
                # Don't count this as a retry attempt - retry immediately with new token budget
                continue
            # Re-raise other errors
            raise
        
        last_raw_response = raw_response

        # 2. Parse and validate JSON
        try:
            data = parse_json_safely(raw_response)
            if validator_fn:
                valid, error_msg = validator_fn(data)
                if not valid:
                    raise ValueError(error_msg)
            return data
        except Exception as parse_exc:
            last_error = str(parse_exc)
            if attempt < max_retries:
                current_prompt = (
                    f"{user_message}\n\n"
                    f"[CORRECTIVE NOTICE - ATTEMPT {attempt + 2}/{max_retries + 1}]\n"
                    f"Your previous response was rejected due to JSON formatting error: {last_error}\n"
                    f"You must output STRICT JSON ONLY matching this specification:\n{schema_description}\n"
                    "Do NOT include markdown formatting (no ```json code blocks), no explanation, only raw JSON."
                )
            else:
                raise RuntimeError(
                    f"Failed to obtain valid JSON from vision agent after {max_retries + 1} attempts.\n"
                    f"Last parsing error: {last_error}\n"
                    f"Last raw output snippet: {last_raw_response[:300]}"
                )


# ============================================================================
# SCHEMA VALIDATORS
# ============================================================================

def _validate_arguments_schema(data: Dict[str, Any]) -> Tuple[bool, str]:
    """Validates that output has an 'arguments' list with exactly 3 items."""
    if not isinstance(data, dict) or "arguments" not in data:
        return False, "Missing top-level key 'arguments'."
    args = data["arguments"]
    if not isinstance(args, list):
        return False, "'arguments' must be a list."
    if len(args) != 3:
        return False, f"Expected exactly 3 arguments, but received {len(args)}."
    for idx, item in enumerate(args):
        if not isinstance(item, dict):
            return False, f"Argument {idx + 1} must be a JSON object."
        for field in ("point", "reason", "evidence"):
            if field not in item or not str(item[field]).strip():
                return False, f"Argument {idx + 1} is missing or has empty '{field}'."
    return True, ""


def _validate_rebuttal_schema(data: Dict[str, Any]) -> Tuple[bool, str]:
    """Validates that output has 'target_point' and 'rebuttal'."""
    if not isinstance(data, dict):
        return False, "Output must be a JSON object."
    if "target_point" not in data or not str(data["target_point"]).strip():
        return False, "Missing or empty 'target_point'."
    if "rebuttal" not in data or not str(data["rebuttal"]).strip():
        return False, "Missing or empty 'rebuttal'."
    return True, ""


def _validate_judge_schema(data: Dict[str, Any]) -> Tuple[bool, str]:
    """Validates the judge's scoring, verdict, and reasoning structure."""
    if not isinstance(data, dict):
        return False, "Judge output must be a JSON object."

    scores = data.get("scores")
    if not isinstance(scores, dict):
        return False, "Missing 'scores' object in judge output."

    for side in ("side_a", "side_b"):
        if side not in scores or not isinstance(scores[side], dict):
            return False, f"Missing scores for '{side}'."
        for crit in ("evidence", "logic", "relevance"):
            if crit not in scores[side]:
                return False, f"Missing criterion '{crit}' for '{side}'."
            try:
                val = int(round(float(scores[side][crit])))
                if not (0 <= val <= 10):
                    return False, f"Score for {side}.{crit} ({val}) is outside [0, 10]."
                scores[side][crit] = val
            except (ValueError, TypeError):
                return False, f"Score for {side}.{crit} must be an integer between 0 and 10."

    strongest = data.get("strongest_point_each")
    if not isinstance(strongest, dict):
        return False, "Missing 'strongest_point_each' dictionary in judge output."
    for side in ("side_a", "side_b"):
        if side not in strongest or not str(strongest[side]).strip():
            return False, f"Missing or empty strongest point for '{side}'."

    verdict = str(data.get("verdict", "")).strip().lower()
    if verdict not in ("side_a", "side_b", "draw"):
        return False, f"Verdict must be 'side_a', 'side_b', or 'draw'; received '{verdict}'."
    data["verdict"] = verdict

    reasoning = data.get("reasoning")
    if not reasoning or not str(reasoning).strip():
        return False, "Missing or empty 'reasoning' from judge."

    return True, ""


# ============================================================================
# AGENT IMPLEMENTATIONS
# ============================================================================

class AdvocateAgent:
    """
    Role: Counsel for the Affirmative (Advocate).
    Argues strictly and persuasively in favor of the claim with exactly 3 arguments.
    Never invents statistics and grounds reasoning in documented principles.
    """

    @staticmethod
    def argue(claim: str) -> Dict[str, Any]:
        user_message = (
            f"Debate Claim: \"{claim}\"\n\n"
            "Formulate your formal opening case. Present EXACTLY 3 arguments strictly IN FAVOR of this claim. "
            "Output your arguments in the specified JSON format."
        )
        schema_desc = '{"arguments": [{"point": "string", "reason": "string", "evidence": "string"}]} (exactly 3 items)'
        return call_agent_with_retry(
            system_prompt=ADVOCATE_SYSTEM_PROMPT,
            user_message=user_message,
            schema_description=schema_desc,
            validator_fn=_validate_arguments_schema,
        )


class OpposerAgent:
    """
    Role: Counsel for the Opposition (Opposer).
    Argues strictly and persuasively against the claim with exactly 3 arguments.
    Focuses on vulnerabilities, unintended consequences, and counter-evidence.
    """

    @staticmethod
    def argue(claim: str) -> Dict[str, Any]:
        user_message = (
            f"Debate Claim: \"{claim}\"\n\n"
            "Formulate your formal opening case. Present EXACTLY 3 arguments strictly AGAINST this claim. "
            "Output your arguments in the specified JSON format."
        )
        schema_desc = '{"arguments": [{"point": "string", "reason": "string", "evidence": "string"}]} (exactly 3 items)'
        return call_agent_with_retry(
            system_prompt=OPPOSER_SYSTEM_PROMPT,
            user_message=user_message,
            schema_description=schema_desc,
            validator_fn=_validate_arguments_schema,
        )


class RebuttalAgent:
    """
    Role: Rebuttal Counsel.
    Each side reviews the opponent's 3 opening arguments, identifies the single
    strongest point, and delivers a targeted rebuttal against it.
    """

    @staticmethod
    def rebut(role_name: str, claim: str, own_side: str, opponent_args: list) -> Dict[str, Any]:
        # Format opponent's arguments cleanly for review
        formatted_opp_args = "\n".join(
            [
                f"{i + 1}. Point: {arg.get('point')}\n   Reason: {arg.get('reason')}\n   Evidence: {arg.get('evidence')}"
                for i, arg in enumerate(opponent_args)
            ]
        )

        user_message = (
            f"Debate Claim: \"{claim}\"\n"
            f"Your Role: {role_name} (arguing {own_side})\n\n"
            "Here are the arguments presented by your opponent:\n"
            f"{formatted_opp_args}\n\n"
            "Identify the opponent's single strongest argument and deliver a sharp, decisive rebuttal against it. "
            "Output strict JSON with 'target_point' and 'rebuttal'."
        )
        schema_desc = '{"target_point": "string", "rebuttal": "string"}'
        return call_agent_with_retry(
            system_prompt=REBUTTAL_SYSTEM_PROMPT,
            user_message=user_message,
            schema_description=schema_desc,
            validator_fn=_validate_rebuttal_schema,
        )


class JudgeAgent:
    """
    Role: Impartial Presiding Judge.
    Evaluates the debate in a strictly blinded fashion:
    - Never sees 'Pro' or 'Con'.
    - Sides are randomly assigned to 'Side A' and 'Side B' to prevent order or position bias.
    - Evaluates solely on court record (Evidence, Logic, Relevance from 0 to 10).
    - Results are mapped back to Pro/Con for user presentation.
    """

    @staticmethod
    def _sanitize_text_for_judge(text: str, is_side_a_pro: bool) -> str:
        """
        Ensures that the Judge never sees explicit 'pro', 'con', 'advocate', or 'opposer'
        labels, replacing role designations with neutral 'Side A' and 'Side B' labels.
        """
        pro_label = "Side A" if is_side_a_pro else "Side B"
        con_label = "Side B" if is_side_a_pro else "Side A"

        text = re.sub(r"\bpro\b", pro_label, text, flags=re.IGNORECASE)
        text = re.sub(r"\bcon\b", con_label, text, flags=re.IGNORECASE)
        text = re.sub(r"\badvocate\b", f"{pro_label} representative", text, flags=re.IGNORECASE)
        text = re.sub(r"\bopposer\b", f"{con_label} representative", text, flags=re.IGNORECASE)
        text = re.sub(r"\bthe affirmative\b", pro_label, text, flags=re.IGNORECASE)
        text = re.sub(r"\bthe opposition\b", con_label, text, flags=re.IGNORECASE)
        return text

    @classmethod
    def evaluate(
        cls,
        claim: str,
        pro_arguments: list,
        con_arguments: list,
        pro_rebuttal: dict,
        con_rebuttal: dict,
        pro_is_side_a: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Blinds the sides as Side A and Side B, submits the case to the Judge,
        and unblinds the outcome back to Pro / Con.
        If pro_is_side_a is None, assigns 50/50 randomly.
        """
        if pro_is_side_a is None:
            pro_is_side_a = random.choice([True, False])

        if pro_is_side_a:
            side_a_args = pro_arguments
            side_b_args = con_arguments
            side_a_rebuttal = pro_rebuttal
            side_b_rebuttal = con_rebuttal
        else:
            side_a_args = con_arguments
            side_b_args = pro_arguments
            side_a_rebuttal = con_rebuttal
            side_b_rebuttal = pro_rebuttal

        def fmt_args(args_list: list) -> str:
            return "\n".join(
                [
                    f"Argument {i + 1}:\n"
                    f"  Thesis: {a.get('point', '')}\n"
                    f"  Reasoning: {a.get('reason', '')}\n"
                    f"  Evidence: {a.get('evidence', '')}"
                    for i, a in enumerate(args_list)
                ]
            )

        court_record = (
            f"DEBATE TOPIC:\n\"{claim}\"\n\n"
            f"=== SUBMISSION FROM SIDE A ===\n"
            f"{fmt_args(side_a_args)}\n\n"
            f"Side A Rebuttal against Side B:\n"
            f"  Target: {side_a_rebuttal.get('target_point', '')}\n"
            f"  Rebuttal: {side_a_rebuttal.get('rebuttal', '')}\n\n"
            f"=== SUBMISSION FROM SIDE B ===\n"
            f"{fmt_args(side_b_args)}\n\n"
            f"Side B Rebuttal against Side A:\n"
            f"  Target: {side_b_rebuttal.get('target_point', '')}\n"
            f"  Rebuttal: {side_b_rebuttal.get('rebuttal', '')}\n\n"
            "Review both sides with complete impartiality. Deliver your scores (0-10), "
            "strongest points, verdict ('side_a', 'side_b', or 'draw'), and judicial reasoning in strict JSON format."
        )

        court_record = cls._sanitize_text_for_judge(court_record, pro_is_side_a)

        schema_desc = (
            '{\n'
            '  "scores": {\n'
            '    "side_a": {"evidence": 0, "logic": 0, "relevance": 0},\n'
            '    "side_b": {"evidence": 0, "logic": 0, "relevance": 0}\n'
            '  },\n'
            '  "strongest_point_each": {"side_a": "string", "side_b": "string"},\n'
            '  "verdict": "side_a | side_b | draw",\n'
            '  "reasoning": "string"\n'
            '}'
        )

        judge_raw = call_agent_with_retry(
            system_prompt=JUDGE_SYSTEM_PROMPT,
            user_message=court_record,
            schema_description=schema_desc,
            validator_fn=_validate_judge_schema,
        )

        # Unblind results back to Pro / Con
        scores = judge_raw["scores"]
        strongest = judge_raw["strongest_point_each"]
        verdict = judge_raw["verdict"]

        if pro_is_side_a:
            pro_scores = scores["side_a"]
            con_scores = scores["side_b"]
            strongest_pro = strongest["side_a"]
            strongest_con = strongest["side_b"]
            if verdict == "side_a":
                mapped_verdict = "pro"
            elif verdict == "side_b":
                mapped_verdict = "con"
            else:
                mapped_verdict = "draw"
        else:
            pro_scores = scores["side_b"]
            con_scores = scores["side_a"]
            strongest_pro = strongest["side_b"]
            strongest_con = strongest["side_a"]
            if verdict == "side_a":
                mapped_verdict = "con"
            elif verdict == "side_b":
                mapped_verdict = "pro"
            else:
                mapped_verdict = "draw"

        return {
            "pro_is_side_a": pro_is_side_a,
            "raw_judge_output": judge_raw,
            "mapped": {
                "verdict": mapped_verdict,
                "pro_scores": pro_scores,
                "con_scores": con_scores,
                "strongest_pro": strongest_pro,
                "strongest_con": strongest_con,
                "reasoning": judge_raw.get("reasoning", ""),
            },
        }


# =============================================================================
# VISION AGENT IMPLEMENTATIONS (for image-based debate Q&A)
# =============================================================================

class VisionAdvocateAgent:
    """
    Role: Vision Advocate.
    Analyzes an image and argues FOR a specific answer to a question about the image.
    """

    @staticmethod
    def argue(question: str, image_base64: str, image_mime_type: str = "image/png") -> Dict[str, Any]:
        from prompts import VISION_ADVOCATE_SYSTEM_PROMPT
        from llm import VISION_ADVOCATE_MAX_TOKENS

        user_message = (
            f"Question about image: \"{question}\"\n\n"
            "Analyze the image and formulate your formal opening case. "
            "Present EXACTLY 3 arguments IN FAVOR of a specific answer to this question. "
            "Each argument MUST reference specific visual elements from the image. "
            "Output your arguments in the specified JSON format."
        )
        schema_desc = '{"arguments": [{"point": "string", "reason": "string", "evidence": "string"}]} (exactly 3 items)'
        return call_agent_with_retry_vision(
            system_prompt=VISION_ADVOCATE_SYSTEM_PROMPT,
            user_message=user_message,
            image_base64=image_base64,
            image_mime_type=image_mime_type,
            schema_description=schema_desc,
            validator_fn=_validate_arguments_schema,
            max_tokens=VISION_ADVOCATE_MAX_TOKENS,
        )


class VisionOpposerAgent:
    """
    Role: Vision Opposer.
    Analyzes an image and argues AGAINST a specific answer to a question about the image.
    """

    @staticmethod
    def argue(question: str, image_base64: str, image_mime_type: str = "image/png") -> Dict[str, Any]:
        from prompts import VISION_OPPOSER_SYSTEM_PROMPT
        from llm import VISION_OPPOSER_MAX_TOKENS

        user_message = (
            f"Question about image: \"{question}\"\n\n"
            "Analyze the image and formulate your formal opening case. "
            "Present EXACTLY 3 arguments AGAINST the answer to this question (or supporting the opposite view). "
            "Each argument MUST reference specific visual elements from the image. "
            "Output your arguments in the specified JSON format."
        )
        schema_desc = '{"arguments": [{"point": "string", "reason": "string", "evidence": "string"}]} (exactly 3 items)'
        return call_agent_with_retry_vision(
            system_prompt=VISION_OPPOSER_SYSTEM_PROMPT,
            user_message=user_message,
            image_base64=image_base64,
            image_mime_type=image_mime_type,
            schema_description=schema_desc,
            validator_fn=_validate_arguments_schema,
            max_tokens=VISION_OPPOSER_MAX_TOKENS,
        )


class VisionRebuttalAgent:
    """
    Role: Vision Rebuttal Counsel.
    Each side reviews the opponent's visual arguments and counters the strongest one.
    """

    @staticmethod
    def rebut(role_name: str, question: str, own_side: str, opponent_args: list, image_base64: str, image_mime_type: str = "image/png") -> Dict[str, Any]:
        from prompts import VISION_REBUTTAL_SYSTEM_PROMPT
        from llm import VISION_REBUTTAL_MAX_TOKENS

        # Format opponent's arguments cleanly for review
        formatted_opp_args = "\n".join(
            [
                f"{i + 1}. Point: {arg.get('point')}\n   Reason: {arg.get('reason')}\n   Evidence: {arg.get('evidence')}"
                for i, arg in enumerate(opponent_args)
            ]
        )

        user_message = (
            f"Question about image: \"{question}\"\n"
            f"Your Role: {role_name} (arguing {own_side})\n\n"
            "Here are the visual arguments presented by your opponent:\n"
            f"{formatted_opp_args}\n\n"
            "Identify the opponent's single strongest visual argument and deliver a sharp, decisive rebuttal against it. "
            "Reference specific visual elements from the image. "
            "Output strict JSON with 'target_point' and 'rebuttal'."
        )
        schema_desc = '{"target_point": "string", "rebuttal": "string"}'
        return call_agent_with_retry_vision(
            system_prompt=VISION_REBUTTAL_SYSTEM_PROMPT,
            user_message=user_message,
            image_base64=image_base64,
            image_mime_type=image_mime_type,
            schema_description=schema_desc,
            validator_fn=_validate_rebuttal_schema,
            max_tokens=VISION_REBUTTAL_MAX_TOKENS,
        )


class VisionJudgeAgent:
    """
    Role: Vision Impartial Presiding Judge.
    Evaluates the debate with full access to the image for visual fact-checking.
    """

    @staticmethod
    def _sanitize_text_for_judge(text: str, is_side_a_pro: bool) -> str:
        """Ensures the Judge never sees explicit 'pro', 'con', 'advocate', or 'opposer' labels."""
        pro_label = "Side A" if is_side_a_pro else "Side B"
        con_label = "Side B" if is_side_a_pro else "Side A"

        text = re.sub(r"\bpro\b", pro_label, text, flags=re.IGNORECASE)
        text = re.sub(r"\bcon\b", con_label, text, flags=re.IGNORECASE)
        text = re.sub(r"\badvocate\b", f"{pro_label} representative", text, flags=re.IGNORECASE)
        text = re.sub(r"\bopposer\b", f"{con_label} representative", text, flags=re.IGNORECASE)
        text = re.sub(r"\bthe affirmative\b", pro_label, text, flags=re.IGNORECASE)
        text = re.sub(r"\bthe opposition\b", con_label, text, flags=re.IGNORECASE)
        return text

    @classmethod
    def evaluate(
        cls,
        question: str,
        pro_arguments: list,
        con_arguments: list,
        pro_rebuttal: dict,
        con_rebuttal: dict,
        image_base64: str,
        image_mime_type: str = "image/png",
        pro_is_side_a: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates the debate with image context.
        Blinds sides as Side A/B, includes image for visual fact-checking.
        """
        from llm import call_llm_vision
        from prompts import VISION_JUDGE_SYSTEM_PROMPT

        if pro_is_side_a is None:
            pro_is_side_a = random.choice([True, False])

        if pro_is_side_a:
            side_a_args = pro_arguments
            side_b_args = con_arguments
            side_a_rebuttal = pro_rebuttal
            side_b_rebuttal = con_rebuttal
        else:
            side_a_args = con_arguments
            side_b_args = pro_arguments
            side_a_rebuttal = con_rebuttal
            side_b_rebuttal = pro_rebuttal

        def fmt_args(args_list: list) -> str:
            return "\n".join(
                [
                    f"Argument {i + 1}:\n"
                    f"  Thesis: {a.get('point', '')}\n"
                    f"  Reasoning: {a.get('reason', '')}\n"
                    f"  Evidence: {a.get('evidence', '')}"
                    for i, a in enumerate(args_list)
                ]
            )

        court_record = (
            f"QUESTION ABOUT IMAGE:\n\"{question}\"\n\n"
            f"=== SUBMISSION FROM SIDE A ===\n"
            f"{fmt_args(side_a_args)}\n\n"
            f"Side A Rebuttal against Side B:\n"
            f"  Target: {side_a_rebuttal.get('target_point', '')}\n"
            f"  Rebuttal: {side_a_rebuttal.get('rebuttal', '')}\n\n"
            f"=== SUBMISSION FROM SIDE B ===\n"
            f"{fmt_args(side_b_args)}\n\n"
            f"Side B Rebuttal against Side A:\n"
            f"  Target: {side_b_rebuttal.get('target_point', '')}\n"
            f"  Rebuttal: {side_b_rebuttal.get('rebuttal', '')}\n\n"
            "Review both sides with complete impartiality. You have access to the image for visual fact-checking. "
            "Deliver your scores (0-10), strongest points, verdict ('side_a', 'side_b', or 'draw'), "
            "and judicial reasoning in strict JSON format."
        )

        court_record = cls._sanitize_text_for_judge(court_record, pro_is_side_a)

        schema_desc = (
            '{\n'
            '  "scores": {\n'
            '    "side_a": {"evidence": 0, "logic": 0, "relevance": 0},\n'
            '    "side_b": {"evidence": 0, "logic": 0, "relevance": 0}\n'
            '  },\n'
            '  "strongest_point_each": {"side_a": "string", "side_b": "string"},\n'
            '  "verdict": "side_a | side_b | draw",\n'
            '  "reasoning": "string"\n'
            '}'
        )

        from llm import VISION_JUDGE_MAX_TOKENS

        judge_raw = call_agent_with_retry_vision(
            system_prompt=VISION_JUDGE_SYSTEM_PROMPT,
            user_message=court_record,
            image_base64=image_base64,
            image_mime_type=image_mime_type,
            schema_description=schema_desc,
            validator_fn=_validate_judge_schema,
            max_tokens=VISION_JUDGE_MAX_TOKENS,
        )

        # Unblind results back to Pro / Con
        scores = judge_raw["scores"]
        strongest = judge_raw["strongest_point_each"]
        verdict = judge_raw["verdict"]

        if pro_is_side_a:
            pro_scores = scores["side_a"]
            con_scores = scores["side_b"]
            strongest_pro = strongest["side_a"]
            strongest_con = strongest["side_b"]
            if verdict == "side_a":
                mapped_verdict = "pro"
            elif verdict == "side_b":
                mapped_verdict = "con"
            else:
                mapped_verdict = "draw"
        else:
            pro_scores = scores["side_b"]
            con_scores = scores["side_a"]
            strongest_pro = strongest["side_b"]
            strongest_con = strongest["side_a"]
            if verdict == "side_a":
                mapped_verdict = "con"
            elif verdict == "side_b":
                mapped_verdict = "pro"
            else:
                mapped_verdict = "draw"

        return {
            "pro_is_side_a": pro_is_side_a,
            "raw_judge_output": judge_raw,
            "mapped": {
                "verdict": mapped_verdict,
                "pro_scores": pro_scores,
                "con_scores": con_scores,
                "strongest_pro": strongest_pro,
                "strongest_con": strongest_con,
                "reasoning": judge_raw.get("reasoning", ""),
            },
        }
