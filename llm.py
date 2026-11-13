"""
llm.py - Unified LLM Interface for ai-judiciary

Exposes a single function `call_llm(system_prompt, user_message)` that delegates
to either Google Gemini (via the new google-genai SDK) or Anthropic Claude (via anthropic SDK)
based on the `LLM_PROVIDER` environment variable.
API keys are loaded securely from `.env` and are never hardcoded.
"""

import os
import re
import json
import time
import random
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional


def _load_env_file():
    """
    Safely loads key-value pairs from .env in the project directory.
    Uses python-dotenv if installed, otherwise uses a built-in parser
    to ensure zero-dependency environment loading.
    """
    # 1. Try python-dotenv if available
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(override=True)
        return
    except ImportError:
        pass

    # 2. Built-in fallback parser for .env file
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.is_file():
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key:
                        os.environ[key] = val
        except Exception:
            pass


# Ensure .env is loaded upon import
_load_env_file()

# Global cache for verified working model
_WORKING_GEMINI_MODEL = None


def _extract_status_code(exc: Optional[Exception]) -> Optional[int]:
    """Extracts HTTP status code from exception if available."""
    if exc is None:
        return None
    # 1. urllib.error.HTTPError or exceptions with .code attribute
    if hasattr(exc, "code") and isinstance(exc.code, int):
        return exc.code
    # 2. SDK exceptions (e.g. google.genai.errors.APIError, anthropic.APIStatusError)
    for attr in ("status_code", "http_status"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
    # 3. Regex in exception string
    match = re.search(r"\bHTTP\s*(?:status\s*)?(?:code\s*)?[:=]?\s*([45]\d\d)\b", str(exc), re.IGNORECASE)
    if match:
        try:
            return int(match.group(1))
        except Exception:
            pass
    # 4. Standalone error codes
    err_str = str(exc)
    for code in (429, 503, 500, 502, 504, 400, 401, 403, 404):
        if str(code) in err_str:
            return code
    return None


def _is_rate_limit(exc: Optional[Exception], err_str: str) -> bool:
    """Returns True if the error indicates a 429 rate limit or quota exhaustion."""
    code = _extract_status_code(exc)
    if code == 429:
        return True
    err_lower = err_str.lower()
    return any(
        marker in err_lower
        for marker in (
            "429",
            "resource_exhausted",
            "resourceexhausted",
            "rate_limit",
            "rate limit",
            "ratelimit",
            "quota",
            "too many requests",
            "throughput",
        )
    )


def _is_timeout(exc: Optional[Exception], err_str: str) -> bool:
    """Returns True if the error was caused by a socket or HTTP timeout."""
    if exc is not None and isinstance(exc, (TimeoutError,)):
        return True
    err_lower = err_str.lower()
    return any(marker in err_lower for marker in ("timed out", "timeout", "time out", "deadline"))


def _is_service_unavailable(exc: Optional[Exception], err_str: str) -> bool:
    """Returns True if the server is temporarily experiencing high load (503)."""
    code = _extract_status_code(exc)
    if code == 503:
        return True
    err_lower = err_str.lower()
    return any(
        marker in err_lower
        for marker in (
            "503",
            "unavailable",
            "high demand",
            "spikes in demand",
            "temporarily",
        )
    )


def _call_gemini(system_prompt: str, user_message: str) -> str:
    """
    Calls Google Gemini using the new google-genai SDK or standard library REST API fallback.
    Enforces a 120-second timeout per individual call, disables thinking, retries on 429 rate limits,
    and falls back across 3.x candidate models.
    """
    global _WORKING_GEMINI_MODEL

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set in environment or .env file.\n"
            "Please create a .env file based on .env.example and supply your API key."
        )

    api_key = api_key.strip().strip("'\"")

    # Preferred model resolution
    configured_model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip().strip("'\"")
    if configured_model in (
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "",
    ):
        configured_model = "gemini-3.8-flash"

    preferred_model = _WORKING_GEMINI_MODEL or configured_model

    # Candidate models to try in order of priority (current 3.x generation)
    candidate_models = [preferred_model]
    for fallback in (
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.5-flash",
    ):
        if fallback not in candidate_models:
            candidate_models.append(fallback)

    last_error_detail = None
    last_exc = None
    attempt_count = 0

    # --------------------------------------------------------------------------
    # 1. Try with the new google-genai SDK (120s timeout per individual call)
    # --------------------------------------------------------------------------
    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore

        # 120-second timeout per individual call
        http_opts = None
        try:
            http_opts = types.HttpOptions(timeout=120.0)
        except Exception:
            try:
                http_opts = types.HttpOptions(timeout=120000)
            except Exception:
                pass

        client = genai.Client(api_key=api_key, http_options=http_opts) if http_opts else genai.Client(api_key=api_key)

        # Explicitly disable thinking for low latency
        thinking_cfg = None
        try:
            thinking_cfg = types.ThinkingConfig(thinking_budget=0)
        except Exception:
            pass

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.3,
            max_output_tokens=1500,
            response_mime_type="application/json",
            thinking_config=thinking_cfg,
        )

        for mod in candidate_models:
            max_rl_retries = 3
            for rl_attempt in range(max_rl_retries + 1):
                attempt_count += 1
                try:
                    response = client.models.generate_content(
                        model=mod,
                        contents=user_message,
                        config=config,
                    )
                    if response and response.text:
                        _WORKING_GEMINI_MODEL = mod
                        return response.text
                except Exception as exc:
                    err_str = str(exc)
                    last_error_detail = err_str
                    last_exc = exc
                    status_code = _extract_status_code(exc)

                    # If thinking_config rejected, retry without it
                    if "thinking" in err_str.lower() and config.thinking_config is not None:
                        config.thinking_config = None
                        continue

                    # If model not found (404), move to next candidate model immediately
                    if status_code == 404 or "404" in err_str or "not_found" in err_str.lower():
                        break

                    # If individual call timed out after 120s
                    if _is_timeout(exc, err_str):
                        last_error_detail = f"Individual call timed out after 120 seconds on model '{mod}' (attempt {attempt_count})"
                        break

                    # If 429 Rate Limit / Resource Exhausted: wait a few seconds and retry
                    if status_code == 429 or _is_rate_limit(exc, err_str):
                        if rl_attempt < max_rl_retries:
                            wait_sec = (3.0 * (rl_attempt + 1)) + random.uniform(0.5, 2.0)
                            print(
                                f"\n  [Rate limit (HTTP 429) on {mod} (attempt {attempt_count}). "
                                f"Pausing {wait_sec:.1f}s before retry {rl_attempt + 1}/{max_rl_retries}...]",
                                flush=True,
                            )
                            time.sleep(wait_sec)
                            continue
                        else:
                            # Retries exhausted on this model, switch to next candidate model
                            break

                    # If 503 high demand: short backoff and try next
                    if status_code == 503 or _is_service_unavailable(exc, err_str):
                        if rl_attempt == 0:
                            time.sleep(2.0)
                            continue
                        else:
                            break

                    # For permanent client errors (400, 401, 403), raise immediately with full details
                    status_label = f"HTTP {status_code}" if status_code else type(exc).__name__
                    raise RuntimeError(
                        f"Gemini API call failed ({status_label}) on attempt {attempt_count}: {exc}"
                    ) from exc

    except ImportError:
        # SDK not installed; proceed to standard library REST fallback below
        pass

    # --------------------------------------------------------------------------
    # 2. Direct REST API fallback using standard library (120s timeout per call)
    # --------------------------------------------------------------------------
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }
    payload = {
        "system_instruction": {
            "parts": [{"text": system_prompt}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_message}]
            }
        ],
        "generationConfig": {
            "temperature": 0.3,
            "responseMimeType": "application/json",
            "maxOutputTokens": 1500,
            "thinkingConfig": {
                "thinkingBudget": 0
            }
        }
    }
    data_bytes = json.dumps(payload).encode("utf-8")

    for mod in candidate_models:
        for api_version in ("v1beta", "v1"):
            url = f"https://generativelanguage.googleapis.com/{api_version}/models/{mod}:generateContent?key={api_key}"
            req = urllib.request.Request(url, data=data_bytes, headers=headers, method="POST")

            max_rl_retries = 3
            for rl_attempt in range(max_rl_retries + 1):
                attempt_count += 1
                try:
                    with urllib.request.urlopen(req, timeout=120) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                        candidates = data.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            if parts:
                                _WORKING_GEMINI_MODEL = mod
                                return parts[0].get("text", "")
                except urllib.error.HTTPError as http_err:
                    err_body = http_err.read().decode("utf-8", errors="replace")
                    last_error_detail = f"HTTP {http_err.code}: {err_body}"
                    last_exc = http_err

                    # If 429 Rate limit: wait and retry
                    if http_err.code == 429:
                        if rl_attempt < max_rl_retries:
                            wait_sec = (3.0 * (rl_attempt + 1)) + random.uniform(0.5, 2.0)
                            print(
                                f"\n  [REST API Rate limit (HTTP 429) on {mod} (attempt {attempt_count}). "
                                f"Pausing {wait_sec:.1f}s before retry {rl_attempt + 1}/{max_rl_retries}...]",
                                flush=True,
                            )
                            time.sleep(wait_sec)
                            continue
                        else:
                            break

                    if http_err.code in (404, 503):
                        break

                    raise RuntimeError(
                        f"Gemini REST API error (HTTP {http_err.code}, attempt {attempt_count}): {err_body}"
                    ) from http_err

                except (TimeoutError, urllib.error.URLError) as url_err:
                    err_str = str(url_err)
                    last_exc = url_err
                    if _is_timeout(url_err, err_str):
                        last_error_detail = f"Individual call timed out after 120 seconds on model '{mod}' (attempt {attempt_count})"
                    else:
                        last_error_detail = f"Network error on attempt {attempt_count}: {err_str}"
                    break

                except Exception as exc:
                    last_error_detail = str(exc)
                    last_exc = exc
                    break

    status_code = _extract_status_code(last_exc) if last_exc else None
    status_str = f"HTTP {status_code}" if status_code else "HTTP status N/A"
    raise RuntimeError(
        f"Unable to generate response from Gemini API ({status_str}, total attempts: {attempt_count}).\n"
        f"Attempted models: {', '.join(candidate_models)}\n"
        f"Last error detail: {last_error_detail}"
    )


def _call_claude(system_prompt: str, user_message: str) -> str:
    """
    Calls Anthropic Claude using the official anthropic SDK or REST API fallback.
    Enforces a 120-second timeout per individual call, retries on 429 rate limits.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY is not set in environment or .env file.\n"
            "Please create a .env file based on .env.example and supply your API key."
        )

    api_key = api_key.strip().strip("'\"")
    model = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022").strip().strip("'\"")

    attempt_count = 0
    max_retries = 3

    # 1. Try official anthropic SDK (120s timeout per call)
    try:
        import anthropic  # type: ignore
        client = anthropic.Anthropic(api_key=api_key, timeout=120.0)

        for rl_attempt in range(max_retries + 1):
            attempt_count += 1
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=1500,
                    temperature=0.3,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}],
                )
                if response and response.content:
                    return response.content[0].text
            except Exception as exc:
                err_str = str(exc)
                status_code = _extract_status_code(exc)

                if status_code == 429 or _is_rate_limit(exc, err_str):
                    if rl_attempt < max_retries:
                        wait_sec = (3.0 * (rl_attempt + 1)) + random.uniform(0.5, 2.0)
                        print(
                            f"\n  [Claude Rate limit (HTTP 429) (attempt {attempt_count}). "
                            f"Pausing {wait_sec:.1f}s before retry {rl_attempt + 1}/{max_retries}...]",
                            flush=True,
                        )
                        time.sleep(wait_sec)
                        continue

                # If primary model failed, try haiku fallback
                if "model" in err_str.lower() and model != "claude-3-5-haiku-20241022":
                    try:
                        resp_fallback = client.messages.create(
                            model="claude-3-5-haiku-20241022",
                            max_tokens=1500,
                            temperature=0.3,
                            system=system_prompt,
                            messages=[{"role": "user", "content": user_message}],
                        )
                        if resp_fallback and resp_fallback.content:
                            return resp_fallback.content[0].text
                    except Exception:
                        pass

                status_label = f"HTTP {status_code}" if status_code else type(exc).__name__
                raise RuntimeError(
                    f"Claude SDK call failed ({status_label}) on attempt {attempt_count}: {exc}"
                ) from exc

    except ImportError:
        pass

    # 2. Direct REST API fallback using standard library (120s timeout per call)
    endpoint = "https://api.anthropic.com/v1/messages"
    payload = {
        "model": model,
        "max_tokens": 1500,
        "temperature": 0.3,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_message}],
    }

    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST"
    )

    for rl_attempt in range(max_retries + 1):
        attempt_count += 1
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content = data.get("content", [])
                if content and "text" in content[0]:
                    return content[0]["text"]
                raise RuntimeError(f"Unexpected response structure from Anthropic Claude: {data}")
        except urllib.error.HTTPError as http_err:
            err_body = http_err.read().decode("utf-8", errors="replace")
            if http_err.code == 429 and rl_attempt < max_retries:
                wait_sec = (3.0 * (rl_attempt + 1)) + random.uniform(0.5, 2.0)
                print(
                    f"\n  [Claude REST Rate limit (HTTP 429) (attempt {attempt_count}). "
                    f"Pausing {wait_sec:.1f}s before retry {rl_attempt + 1}/{max_retries}...]",
                    flush=True,
                )
                time.sleep(wait_sec)
                continue
            raise RuntimeError(
                f"Claude API request failed (HTTP {http_err.code}, attempt {attempt_count}): {err_body}"
            ) from http_err
        except Exception as exc:
            if _is_timeout(exc, str(exc)):
                raise RuntimeError(
                    f"Claude API call timed out after 120s on attempt {attempt_count}."
                ) from exc
            raise RuntimeError(
                f"Failed to communicate with Claude on attempt {attempt_count}: {exc}"
            ) from exc

    raise RuntimeError(f"Claude API call exhausted retries after {attempt_count} attempts.")


def call_llm(system_prompt: str, user_message: str) -> str:
    """
    Unified LLM caller. Selects provider based on LLM_PROVIDER in .env.

    Parameters:
        system_prompt (str): Instructions and role definition for the agent.
        user_message (str): The prompt / context message sent to the agent.

    Returns:
        str: Raw text output from the language model.
    """
    # Refresh environment to detect newly added keys/settings
    _load_env_file()
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()

    if provider in ("gemini", "google"):
        return _call_gemini(system_prompt, user_message)
    elif provider in ("claude", "anthropic"):
        return _call_claude(system_prompt, user_message)
    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER '{provider}'. "
            "Please set LLM_PROVIDER to either 'gemini' or 'claude' in your .env file."
        )
