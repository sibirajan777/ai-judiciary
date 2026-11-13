"""
llm.py - Unified LLM Interface for ai-judiciary

Exposes a single function `call_llm(system_prompt, user_message)` that delegates
to Google Gemini, Anthropic Claude, or NVIDIA NIM / OpenRouter based on `LLM_PROVIDER`.

Key Reliability & SSL Policies:
1. Standard Library SSL for Gemini: Uses standard library defaults for SSL (ssl.create_default_context(),
   no custom SSL context) to eliminate Windows TLS handshake timeouts caused by third-party transports.
2. NVIDIA NIM / OpenRouter Integration: Uses the official `openai` Python SDK with base_url="https://integrate.api.nvidia.com/v1"
   (or auto-detects OpenRouter "sk-or-v1-" keys to route to "https://openrouter.ai/api/v1") with standard library HTTPS fallback.
3. Split Timeout: 15-second connect timeout and 90-second read timeout (120 seconds total deadline).
4. Retry Policy: Retries up to 3 times on SSL or timeout errors.
5. HTTP 429 Rate Limit: Backs off for 10 to 30 seconds before retrying.
6. HTTP 400 (INVALID_ARGUMENT / Bad Request): Never retried (unless recovering from unsupported response_format);
   fails immediately with the exact config and error message.
7. Clean Config: Only standard supported parameters; JSON output requested where supported.
"""

import os
import re
import json
import time
import random
import socket
import ssl
import http.client
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def _load_env_file():
    """
    Safely loads key-value pairs from .env in the project directory.
    Uses python-dotenv if installed, otherwise uses a built-in parser.
    Preserves explicitly set environment variables (override=False).
    """
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(override=False)
        return
    except ImportError:
        pass

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
                    if key and key not in os.environ:
                        os.environ[key] = val
        except Exception:
            pass


# Ensure .env is loaded upon import
_load_env_file()

# Configurable timeouts via environment variables
def _get_float_env(key: str, default: float) -> float:
    """Safely parse float from environment variable with fallback."""
    try:
        return float(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        logger.warning(f"Invalid value for {key}, using default: {default}")
        return default


def _get_int_env(key: str, default: int) -> int:
    """Safely parse int from environment variable with fallback."""
    try:
        return int(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        logger.warning(f"Invalid value for {key}, using default: {default}")
        return default


CONNECT_TIMEOUT = _get_float_env("LLM_CONNECT_TIMEOUT", 15.0)
READ_TIMEOUT = _get_float_env("LLM_READ_TIMEOUT", 90.0)
MAX_RETRIES = _get_int_env("LLM_MAX_RETRIES", 3)
RATE_LIMIT_BASE_BACKOFF = _get_float_env("LLM_RATE_LIMIT_BACKOFF", 10.0)
MAX_CONCURRENCY = _get_int_env("LLM_MAX_CONCURRENCY", 2)

# Vision model configuration - NVIDIA NIM vision models
NVIDIA_VISION_MODEL = os.getenv("NVIDIA_VISION_MODEL", "").strip().strip("'\"")
if not NVIDIA_VISION_MODEL:
    NVIDIA_VISION_MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"

# Vision output token limits - configurable via environment
NVIDIA_VISION_MAX_TOKENS = _get_int_env("NVIDIA_VISION_MAX_TOKENS", 4096)

# Per-agent token budgets (can be overridden via env)
VISION_ADVOCATE_MAX_TOKENS = _get_int_env("VISION_ADVOCATE_MAX_TOKENS", 2048)
VISION_OPPOSER_MAX_TOKENS = _get_int_env("VISION_OPPOSER_MAX_TOKENS", 2048)
VISION_REBUTTAL_MAX_TOKENS = _get_int_env("VISION_REBUTTAL_MAX_TOKENS", 2048)
VISION_JUDGE_MAX_TOKENS = _get_int_env("VISION_JUDGE_MAX_TOKENS", 4096)

MAX_IMAGE_SIZE_MB = _get_int_env("MAX_IMAGE_SIZE_MB", 20)
ALLOWED_IMAGE_TYPES = os.getenv("ALLOWED_IMAGE_TYPES", "image/png,image/jpeg,image/webp").split(",")

logger.info(f"LLM timeouts configured: connect={CONNECT_TIMEOUT}s, read={READ_TIMEOUT}s, max_retries={MAX_RETRIES}")
logger.info(f"Vision model: {NVIDIA_VISION_MODEL}, max image size: {MAX_IMAGE_SIZE_MB}MB")
logger.info(f"Vision token budgets - Advocate: {VISION_ADVOCATE_MAX_TOKENS}, Opposer: {VISION_OPPOSER_MAX_TOKENS}, Rebuttal: {VISION_REBUTTAL_MAX_TOKENS}, Judge: {VISION_JUDGE_MAX_TOKENS}")


class SplitTimeoutHTTPSConnection(http.client.HTTPSConnection):
    """
    Standard library HTTPS connection enforcing:
    - Configurable connect timeout (TCP connect + SSL TLS handshake)
    - Standard library default SSL context (ssl.create_default_context(), no custom context)
    - Configurable read timeout (waiting for server generation response)
    """

    def __init__(
        self,
        host: str,
        port: Optional[int] = None,
        connect_timeout: float = CONNECT_TIMEOUT,
        read_timeout: float = READ_TIMEOUT,
        **kwargs,
    ):
        super().__init__(host, port, timeout=connect_timeout, **kwargs)
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout

    def connect(self):
        # 1. Connect TCP socket using configurable connect timeout
        self.sock = socket.create_connection(
            (self.host, self.port),
            timeout=self.connect_timeout,
            source_address=self.source_address,
        )
        # 2. SSL handshake using standard library default SSL context (no custom context)
        context = ssl.create_default_context()
        self.sock = context.wrap_socket(self.sock, server_hostname=self.host)
        # 3. Switch socket timeout to configurable read timeout for response streaming
        self.sock.settimeout(self.read_timeout)


def _call_gemini(system_prompt: str, user_message: str) -> str:
    """
    Calls Google Gemini using the standard library HTTPS connection:
    - Uses standard library defaults for SSL (no custom SSL context)
    - Configurable connect timeout (TCP connect + TLS handshake)
    - Configurable read timeout (response generation)
    - Retries up to MAX_RETRIES times on SSL or timeout errors
    - Retries on HTTP 429 with configurable backoff
    - Fails immediately on HTTP 400 (never retried)
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set in environment or .env file.\n"
            "Please create a .env file based on .env.example and supply your API key."
        )

    api_key = api_key.strip().strip("'\"")

    # Resolve target model from .env
    model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip().strip("'\"")
    if not model or model in ("gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.5-flash", "gemini-1.5-flash"):
        model = "gemini-3.5-flash-lite"

    request_config: Dict[str, Any] = {
        "model": model,
        "temperature": 0.3,
        "max_output_tokens": 1500,
        "response_mime_type": "application/json",
    }

    # Print exact request config sent to model
    print(f"\n  [Gemini Request Config] -> {json.dumps(request_config, indent=2)}", flush=True)

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
            "temperature": request_config["temperature"],
            "responseMimeType": request_config["response_mime_type"],
            "maxOutputTokens": request_config["max_output_tokens"],
        }
    }
    data_bytes = json.dumps(payload).encode("utf-8")

    host = "generativelanguage.googleapis.com"
    path = f"/v1beta/models/{model}:generateContent?key={api_key}"

    max_retries = MAX_RETRIES

    for attempt in range(max_retries + 1):
        conn = None
        try:
            conn = SplitTimeoutHTTPSConnection(host, connect_timeout=CONNECT_TIMEOUT, read_timeout=READ_TIMEOUT)
            conn.request("POST", path, body=data_bytes, headers=headers)
            resp = conn.getresponse()
            status_code = resp.status
            resp_body = resp.read().decode("utf-8")

            # 1. Success (200 OK)
            if status_code == 200:
                data = json.loads(resp_body)
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return parts[0].get("text", "")
                raise RuntimeError(
                    f"Gemini API returned 200 OK but response structure had no candidates: {resp_body[:200]}"
                )

            # 2. HTTP 400 INVALID_ARGUMENT: NEVER RETRY, fail immediately with message
            if status_code == 400:
                raise RuntimeError(
                    f"Gemini API returned HTTP 400 INVALID_ARGUMENT on model '{model}'.\n"
                    f"Request config sent:\n{json.dumps(request_config, indent=2)}\n"
                    f"Underlying error message:\n{resp_body}"
                )

            # 3. HTTP 404 NOT FOUND: fail immediately
            if status_code == 404:
                raise RuntimeError(
                    f"Gemini model '{model}' not found (HTTP 404). Please verify GEMINI_MODEL in your .env file.\n"
                    f"Underlying error: {resp_body}"
                )

            # 4. HTTP 429 RATE LIMIT: Back off with configurable base
            if status_code == 429:
                if attempt < max_retries:
                    wait_sec = min(RATE_LIMIT_BASE_BACKOFF * (attempt + 1), 30.0) + random.uniform(0.5, 2.0)
                    print(
                        f"\n  [Rate limit (HTTP 429) on {model} (attempt {attempt + 1}/{max_retries + 1}). "
                        f"Backing off {wait_sec:.1f}s before retry {attempt + 1}/{max_retries}...]",
                        flush=True,
                    )
                    time.sleep(wait_sec)
                    continue
                else:
                    raise RuntimeError(
                        f"Gemini API rate limit (HTTP 429) on model '{model}' persisted after all {max_retries} retries.\n"
                        f"Underlying error: {resp_body}"
                    )

            # 5. HTTP 503 SERVICE UNAVAILABLE: Back off 10s
            if status_code == 503:
                if attempt < max_retries:
                    wait_sec = 10.0
                    print(
                        f"\n  [Service temporarily unavailable (503). Retrying in {wait_sec:.0f}s ({attempt + 1}/{max_retries})...]",
                        flush=True,
                    )
                    time.sleep(wait_sec)
                    continue

            # Other HTTP error: fail immediately
            raise RuntimeError(
                f"Gemini API error (HTTP {status_code}) on attempt {attempt + 1}: {resp_body}"
            )

        except (
            ssl.SSLError,
            socket.timeout,
            TimeoutError,
            http.client.RemoteDisconnected,
            http.client.HTTPException,
            ConnectionError,
            OSError,
        ) as net_err:
            err_str = str(net_err)
            is_ssl = isinstance(net_err, ssl.SSLError) or "ssl" in err_str.lower() or "handshake" in err_str.lower()
            is_to = isinstance(net_err, (socket.timeout, TimeoutError)) or "timeout" in err_str.lower() or "timed out" in err_str.lower()

            if is_ssl or is_to:
                if attempt < max_retries:
                    wait_sec = 2.0 * (attempt + 1)
                    category = "SSL error" if is_ssl else "Timeout error"
                    print(
                        f"\n  [{category} on attempt {attempt + 1}/{max_retries + 1}: {net_err}. "
                        f"Retrying in {wait_sec:.0f}s ({attempt + 1}/{max_retries})...]",
                        flush=True,
                    )
                    time.sleep(wait_sec)
                    continue
                else:
                    category = "SSL handshake/connection" if is_ssl else "Connect/read timeout"
                    raise RuntimeError(
                        f"Gemini API call failed after {max_retries} retries due to {category} error.\n"
                        f"Underlying error ({type(net_err).__name__}): {net_err}"
                    ) from net_err

            # Re-raise any other unexpected exception
            raise

        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass

    raise RuntimeError(f"Gemini API call exhausted {max_retries} retries on model '{model}'.")


def _call_claude(system_prompt: str, user_message: str) -> str:
    """
    Calls Anthropic Claude using standard library HTTPS connection with configurable timeouts.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY is not set in environment or .env file.\n"
            "Please create a .env file based on .env.example and supply your API key."
        )

    api_key = api_key.strip().strip("'\"")
    model = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022").strip().strip("'\"")

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 1500,
        "temperature": 0.3,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_message}],
    }
    data_bytes = json.dumps(payload).encode("utf-8")

    host = "api.anthropic.com"
    path = "/v1/messages"
    max_retries = MAX_RETRIES

    for attempt in range(max_retries + 1):
        conn = None
        try:
            conn = SplitTimeoutHTTPSConnection(host, connect_timeout=CONNECT_TIMEOUT, read_timeout=READ_TIMEOUT)
            conn.request("POST", path, body=data_bytes, headers=headers)
            resp = conn.getresponse()
            status_code = resp.status
            resp_body = resp.read().decode("utf-8")

            if status_code == 200:
                data = json.loads(resp_body)
                content = data.get("content", [])
                if content and "text" in content[0]:
                    return content[0]["text"]
                raise RuntimeError(f"Unexpected response structure from Claude: {resp_body[:200]}")

            if status_code == 400:
                raise RuntimeError(f"Claude API returned HTTP 400: {resp_body}")

            if status_code == 429:
                if attempt < max_retries:
                    wait_sec = min(RATE_LIMIT_BASE_BACKOFF * (attempt + 1), 30.0) + random.uniform(0.5, 2.0)
                    print(
                        f"\n  [Claude Rate limit (HTTP 429) (attempt {attempt + 1}/{max_retries + 1}). "
                        f"Backing off {wait_sec:.1f}s before retry {attempt + 1}/{max_retries}...]",
                        flush=True,
                    )
                    time.sleep(wait_sec)
                    continue
                else:
                    raise RuntimeError(f"Claude API rate limit (HTTP 429) exhausted retries: {resp_body}")

            raise RuntimeError(f"Claude API request failed (HTTP {status_code}): {resp_body}")

        except (
            ssl.SSLError,
            socket.timeout,
            TimeoutError,
            http.client.RemoteDisconnected,
            http.client.HTTPException,
            ConnectionError,
            OSError,
        ) as net_err:
            if attempt < max_retries:
                wait_sec = 2.0 * (attempt + 1)
                print(f"\n  [Claude SSL/Timeout error: {net_err}. Retrying in {wait_sec:.0f}s...]", flush=True)
                time.sleep(wait_sec)
                continue
            raise RuntimeError(f"Claude API call failed after {max_retries} retries: {net_err}") from net_err

        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass

    raise RuntimeError(f"Claude API call exhausted retries after {max_retries} attempts.")


# Global client instances for connection reuse
_nvidia_client = None
_nvidia_async_client = None


def _get_nvidia_client(api_key: str, base_url: str, is_openrouter: bool) -> "OpenAI":
    """Get or create a shared OpenAI client for NVIDIA NIM/OpenRouter."""
    global _nvidia_client
    from openai import OpenAI
    
    default_headers = {}
    if is_openrouter:
        default_headers = {
            "HTTP-Referer": "https://github.com/sibirajan777/ai-judiciary",
            "X-Title": "AI Judiciary",
        }
    
    if _nvidia_client is None or _nvidia_client.base_url != base_url or _nvidia_client.api_key != api_key:
        _nvidia_client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=READ_TIMEOUT + CONNECT_TIMEOUT,
            default_headers=default_headers if default_headers else None,
        )
    return _nvidia_client


def _call_nvidia(system_prompt: str, user_message: str) -> str:
    """
    Calls NVIDIA NIM API or OpenRouter using the official `openai` Python SDK
    or standard library REST fallback.
    - If key starts with 'nvapi-', routes to https://integrate.api.nvidia.com/v1
    - If key starts with 'sk-or-', automatically routes to https://openrouter.ai/api/v1
    - Base URL can also be overridden via NVIDIA_BASE_URL.
    - Requests JSON output (response_format={'type': 'json_object'}) where supported.
    - 120-second timeout per individual call.
    - Retries capped at 3 per call.
    - HTTP 429: Backs off 10 to 30 seconds before retrying.
    - HTTP 400: Never retried (unless falling back from unsupported response_format).
    - HTTP 401, 403, 404: Never retried (permanent errors).
    """
    api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "NVIDIA_API_KEY is not set in environment or .env file.\n"
            "Please add NVIDIA_API_KEY to your .env file with your key from https://build.nvidia.com/ (or OpenRouter)."
        )

    api_key = api_key.strip().strip("'\"")

    # Smart endpoint detection: if user supplied an OpenRouter key (sk-or-v1-...), route to OpenRouter
    is_openrouter = api_key.startswith("sk-or-")
    default_base_url = "https://openrouter.ai/api/v1" if is_openrouter else "https://integrate.api.nvidia.com/v1"
    base_url = os.getenv("NVIDIA_BASE_URL", default_base_url).strip()

    # Model resolution
    raw_model = os.getenv("NVIDIA_MODEL", "").strip().strip("'\"")
    if not raw_model:
        model = "meta-llama/llama-3.3-70b-instruct" if is_openrouter else "nvidia/llama-3.1-nemotron-70b-instruct"
    else:
        model = raw_model
        if not is_openrouter and "/" not in model:
            model = f"nvidia/{model}"

    max_retries = MAX_RETRIES
    provider_name = "OpenRouter" if is_openrouter else "NVIDIA NIM"

    request_config: Dict[str, Any] = {
        "provider": provider_name,
        "base_url": base_url,
        "model": model,
        "temperature": 0.3,
        "max_tokens": 1500,
        "response_format": "json_object",
    }

    # Log config without sensitive data
    safe_config = {k: v for k, v in request_config.items() if k != "api_key"}
    logger.info(f"[{provider_name} Request Config] -> {json.dumps(safe_config)}")

    # 1. Try with official openai package (with connection reuse)
    try:
        from openai import (  # type: ignore
            OpenAI,
            RateLimitError,
            AuthenticationError,
            NotFoundError,
            BadRequestError,
            APITimeoutError,
            APIConnectionError,
        )

        client = _get_nvidia_client(api_key, base_url, is_openrouter)

        supports_json_format = True
        # Known models that don't support response_format=json_object
        no_json_format_models = [
            "nvidia/nemotron-3-ultra",
            "nemotron-3-ultra",
        ]
        if any(m in model.lower() for m in no_json_format_models):
            supports_json_format = False
        candidate_models = [model]
        if is_openrouter and model != "meta-llama/llama-3.3-70b-instruct":
            candidate_models.append("meta-llama/llama-3.3-70b-instruct")

        for current_model in candidate_models:
            for attempt in range(max_retries + 1):
                try:
                    kwargs: Dict[str, Any] = {
                        "model": current_model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message},
                        ],
                        "temperature": 0.3,
                        "max_tokens": 1500,
                    }
                    if supports_json_format:
                        kwargs["response_format"] = {"type": "json_object"}

                    response = client.chat.completions.create(**kwargs)
                    if response and response.choices:
                        message = response.choices[0].message
                        content = message.content
                        finish_reason = response.choices[0].finish_reason
                        if not content:
                            if finish_reason == "length":
                                raise RuntimeError(
                                    f"{provider_name} response truncated (finish_reason=length). "
                                    f"Increase max_tokens or reduce prompt size."
                                )
                            if finish_reason == "content_filter":
                                raise RuntimeError(
                                    f"{provider_name} response filtered by content policy."
                                )
                            raise RuntimeError(
                                f"{provider_name} returned empty message.content. "
                                f"finish_reason={finish_reason}. "
                                f"This model may not support the requested format or the prompt may need adjustment."
                            )
                        return content
                    raise RuntimeError(f"{provider_name} returned empty response: {response}")

                except BadRequestError as bad_req:
                    err_msg = str(bad_req)
                    # If model rejects response_format, retry immediately without it
                    if supports_json_format and any(k in err_msg.lower() for k in ("response_format", "json_object", "format")):
                        logger.warning(f"Model '{current_model}' does not support response_format='json_object'. Retrying without it...")
                        supports_json_format = False
                        continue
                    # Never retry other 400 Bad Request errors
                    raise RuntimeError(
                        f"{provider_name} returned HTTP 400 Bad Request on model '{current_model}'.\n"
                        f"Request config sent:\n{json.dumps(request_config, indent=2)}\n"
                        f"Underlying error message:\n{bad_req}"
                    ) from bad_req

                except AuthenticationError as auth_err:
                    auth_note = ""
                    if not is_openrouter and api_key.startswith("sk-or-"):
                        auth_note = " (Note: Your API key appears to be from OpenRouter 'sk-or-v1-'. For NVIDIA NIM use an 'nvapi-' key from build.nvidia.com)."
                    raise RuntimeError(
                        f"{provider_name} Authentication failed (HTTP 401 Unauthorized).\n"
                        f"Please verify NVIDIA_API_KEY in your .env file.{auth_note}\n"
                        f"Underlying error message: {auth_err}"
                    ) from auth_err

                except NotFoundError as not_found_err:
                    # If 404 on OpenRouter and we have a fallback model, try it
                    if is_openrouter and current_model != candidate_models[-1]:
                        logger.warning(f"Model '{current_model}' unavailable on OpenRouter. Falling back to '{candidate_models[-1]}'...")
                        break
                    raise RuntimeError(
                        f"{provider_name} model '{current_model}' not found (HTTP 404).\n"
                        f"Please check NVIDIA_MODEL in your .env file.\n"
                        f"Underlying error: {not_found_err}"
                    ) from not_found_err

                except RateLimitError as rl_err:
                    if attempt < max_retries:
                        wait_sec = min(RATE_LIMIT_BASE_BACKOFF * (attempt + 1), 30.0) + random.uniform(0.5, 2.0)
                        logger.warning(f"Rate limit (HTTP 429) on {current_model} (attempt {attempt + 1}/{max_retries + 1}). Backing off {wait_sec:.1f}s...")
                        time.sleep(wait_sec)
                        continue
                    else:
                        raise RuntimeError(
                            f"{provider_name} rate limit (HTTP 429) persisted after {max_retries} retries on model '{current_model}': {rl_err}"
                        ) from rl_err

                except (APITimeoutError, TimeoutError, socket.timeout) as to_err:
                    if attempt < max_retries:
                        wait_sec = 2.0 * (attempt + 1)
                        logger.warning(f"[{provider_name} call timed out (attempt {attempt + 1}/{max_retries + 1}). Retrying in {wait_sec:.0f}s...]")
                        time.sleep(wait_sec)
                        continue
                    raise RuntimeError(f"{provider_name} call timed out on model '{current_model}' ({to_err})") from to_err

                except (APIConnectionError, ssl.SSLError, ConnectionError) as conn_err:
                    if attempt < max_retries:
                        wait_sec = 2.0 * (attempt + 1)
                        logger.warning(f"[{provider_name} connection error (attempt {attempt + 1}/{max_retries + 1}): {conn_err}. Retrying in {wait_sec:.0f}s...]")
                        time.sleep(wait_sec)
                        continue
                    raise RuntimeError(f"{provider_name} connection failed after {max_retries} retries: {conn_err}") from conn_err

                except Exception as exc:
                    err_str = str(exc)
                    if "429" in err_str:
                        if attempt < max_retries:
                            wait_sec = min(RATE_LIMIT_BASE_BACKOFF * (attempt + 1), 30.0) + random.uniform(0.5, 2.0)
                            logger.warning(f"[{provider_name} Rate limit (attempt {attempt + 1}/{max_retries + 1}). Backing off {wait_sec:.1f}s...]")
                            time.sleep(wait_sec)
                            continue
                    raise RuntimeError(f"{provider_name} call failed on model '{current_model}': {exc}") from exc

        raise RuntimeError(f"{provider_name} call exhausted {max_retries} retries on model '{model}'.")

    except ImportError:
        pass
        pass

    # 2. REST API Fallback using standard library SplitTimeoutHTTPSConnection
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if is_openrouter:
        headers["HTTP-Referer"] = "https://github.com/sibirajan777/ai-judiciary"
        headers["X-Title"] = "AI Judiciary"

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.3,
        "max_tokens": 1500,
        "response_format": {"type": "json_object"},
    }
    # Parse host and path from base_url
    clean_url = base_url.replace("https://", "").replace("http://", "")
    host = clean_url.split("/")[0]
    subpath = "/" + "/".join(clean_url.split("/")[1:]).strip("/") + "/chat/completions"
    supports_json = True

    for attempt in range(max_retries + 1):
        conn = None
        try:
            data_bytes = json.dumps(payload).encode("utf-8")
            conn = SplitTimeoutHTTPSConnection(host, connect_timeout=CONNECT_TIMEOUT, read_timeout=READ_TIMEOUT)
            conn.request("POST", subpath, body=data_bytes, headers=headers)
            resp = conn.getresponse()
            status_code = resp.status
            resp_body = resp.read().decode("utf-8")

            if status_code == 200:
                data = json.loads(resp_body)
                choices = data.get("choices", [])
                if choices:
                    message = choices[0].get("message", {})
                    content = message.get("content", "")
                    finish_reason = choices[0].get("finish_reason", "")
                    if not content:
                        if finish_reason == "length":
                            raise RuntimeError(
                                f"{provider_name} REST response truncated (finish_reason=length). "
                                f"Increase max_tokens or reduce prompt size."
                            )
                        if finish_reason == "content_filter":
                            raise RuntimeError(
                                f"{provider_name} REST response filtered by content policy."
                            )
                        raise RuntimeError(
                            f"{provider_name} REST returned empty message.content. "
                            f"finish_reason={finish_reason}. "
                            f"This model may not support the requested format."
                        )
                    return content
                raise RuntimeError(f"Unexpected {provider_name} response structure: {resp_body[:200]}")

            if status_code == 400:
                if supports_json and any(k in resp_body.lower() for k in ("response_format", "json_object", "format")):
                    logger.warning(f"Model '{model}' does not support response_format='json_object'. Retrying without it...")
                    supports_json = False
                    payload.pop("response_format", None)
                    continue
                raise RuntimeError(
                    f"{provider_name} REST API returned HTTP 400 Bad Request on model '{model}'.\n"
                    f"Underlying error: {resp_body}"
                )

            if status_code == 401:
                raise RuntimeError(
                    f"{provider_name} REST API Authentication failed (HTTP 401 Unauthorized).\n"
                    f"Please verify NVIDIA_API_KEY in your .env file.\n"
                    f"Underlying error: {resp_body}"
                )

            if status_code == 404:
                raise RuntimeError(
                    f"{provider_name} model '{model}' not found (HTTP 404).\n"
                    f"Underlying error: {resp_body}"
                )

            if status_code == 429:
                if attempt < max_retries:
                    wait_sec = min(RATE_LIMIT_BASE_BACKOFF * (attempt + 1), 30.0) + random.uniform(0.5, 2.0)
                    print(f"\n  [{provider_name} REST Rate limit (HTTP 429). Backing off {wait_sec:.1f}s before retry...]", flush=True)
                    time.sleep(wait_sec)
                    continue
                raise RuntimeError(f"{provider_name} REST API rate limit (HTTP 429) exhausted retries: {resp_body}")

            raise RuntimeError(f"{provider_name} REST API request failed (HTTP {status_code}): {resp_body}")

        except (
            ssl.SSLError,
            socket.timeout,
            TimeoutError,
            http.client.RemoteDisconnected,
            http.client.HTTPException,
            ConnectionError,
            OSError,
        ) as net_err:
            if attempt < max_retries:
                wait_sec = 2.0 * (attempt + 1)
                print(f"\n  [{provider_name} REST SSL/Timeout error: {net_err}. Retrying in {wait_sec:.0f}s...]", flush=True)
                time.sleep(wait_sec)
                continue
            raise RuntimeError(f"{provider_name} REST API call failed after {max_retries} retries: {net_err}") from net_err

        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass

    raise RuntimeError(f"{provider_name} call exhausted {max_retries} retries on model '{model}'.")


# =============================================================================
# VISION SUPPORT (NVIDIA NIM / OpenRouter vision models)
# =============================================================================

_nvidia_vision_client = None


def _get_nvidia_vision_client(api_key: str, base_url: str, is_openrouter: bool) -> "OpenAI":
    """Get or create a shared OpenAI client for NVIDIA NIM/OpenRouter vision."""
    global _nvidia_vision_client
    from openai import OpenAI
    
    default_headers = {}
    if is_openrouter:
        default_headers = {
            "HTTP-Referer": "https://github.com/sibirajan777/ai-judiciary",
            "X-Title": "AI Judiciary",
        }
    
    if _nvidia_vision_client is None or _nvidia_vision_client.base_url != base_url or _nvidia_vision_client.api_key != api_key:
        _nvidia_vision_client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=READ_TIMEOUT + CONNECT_TIMEOUT,
            default_headers=default_headers if default_headers else None,
        )
    return _nvidia_vision_client


def _call_nvidia_vision(system_prompt: str, user_message: str, image_base64: str, image_mime_type: str = "image/png", max_tokens: int = None) -> str:
    """
    Calls NVIDIA NIM / OpenRouter vision model with image + text input.
    Uses OpenAI-compatible vision message format.
    - If key starts with 'nvapi-', routes to https://integrate.api.nvidia.com/v1
    - If key starts with 'sk-or-', automatically routes to https://openrouter.ai/api/v1
    - Base URL can also be overridden via NVIDIA_BASE_URL.
    - Requests JSON output (response_format={'type': 'json_object'}) where supported.
    - 120-second timeout per individual call.
    - Retries capped at 3 per call.
    - HTTP 429: Backs off 10 to 30 seconds before retrying.
    - HTTP 400: Never retried (unless falling back from unsupported response_format).
    - HTTP 401, 403, 404: Never retried (permanent errors).
    - max_tokens: Optional per-call token budget. Uses NVIDIA_VISION_MAX_TOKENS if not specified.
    """
    api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "NVIDIA_API_KEY is not set in environment or .env file.\n"
            "Please add NVIDIA_API_KEY to your .env file with your key from https://build.nvidia.com/ (or OpenRouter)."
        )

    api_key = api_key.strip().strip("'\"")

    # Smart endpoint detection
    is_openrouter = api_key.startswith("sk-or-")
    default_base_url = "https://openrouter.ai/api/v1" if is_openrouter else "https://integrate.api.nvidia.com/v1"
    base_url = os.getenv("NVIDIA_BASE_URL", default_base_url).strip()

    # Resolve max_tokens: use provided, then env default, then hardcoded fallback
    if max_tokens is None:
        max_tokens = NVIDIA_VISION_MAX_TOKENS

    # Use vision-specific model
    model = NVIDIA_VISION_MODEL
    if not is_openrouter and "/" not in model:
        model = f"nvidia/{model}"

    max_retries = MAX_RETRIES
    provider_name = "OpenRouter" if is_openrouter else "NVIDIA NIM"

    request_config: Dict[str, Any] = {
        "provider": provider_name,
        "base_url": base_url,
        "model": model,
        "temperature": 0.3,
        "max_tokens": max_tokens,
        "response_format": "json_object",
    }

    # Log config without sensitive data
    safe_config = {k: v for k, v in request_config.items() if k != "api_key"}
    logger.info(f"[{provider_name} Vision Request Config] -> {json.dumps(safe_config)}")

    # 1. Try with official openai package (with connection reuse)
    try:
        from openai import (  # type: ignore
            OpenAI,
            RateLimitError,
            AuthenticationError,
            NotFoundError,
            BadRequestError,
            APITimeoutError,
            APIConnectionError,
        )

        client = _get_nvidia_vision_client(api_key, base_url, is_openrouter)

        supports_json_format = True
        # Known models that don't support response_format=json_object
        no_json_format_models = [
            "nvidia/nemotron-3-ultra",
            "nemotron-3-ultra",
        ]
        if any(m in model.lower() for m in no_json_format_models):
            supports_json_format = False

        for attempt in range(max_retries + 1):
            try:
                kwargs: Dict[str, Any] = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_message},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:{image_mime_type};base64,{image_base64}"}
                                }
                            ]
                        },
                    ],
                    "temperature": 0.3,
                    "max_tokens": max_tokens,
                }
                if supports_json_format:
                    kwargs["response_format"] = {"type": "json_object"}

                response = client.chat.completions.create(**kwargs)
                if response and response.choices:
                    message = response.choices[0].message
                    content = message.content
                    finish_reason = response.choices[0].finish_reason
                    if not content:
                        if finish_reason == "length":
                            raise RuntimeError(
                                f"{provider_name} vision response truncated (finish_reason=length). "
                                f"Increase max_tokens or reduce prompt size."
                            )
                        if finish_reason == "content_filter":
                            raise RuntimeError(
                                f"{provider_name} vision response filtered by content policy."
                            )
                        raise RuntimeError(
                            f"{provider_name} vision returned empty message.content. "
                            f"finish_reason={finish_reason}. "
                            f"This model may not support vision or the requested format."
                        )
                    return content
                raise RuntimeError(f"{provider_name} returned empty response: {response}")

            except BadRequestError as bad_req:
                err_msg = str(bad_req)
                if supports_json_format and any(k in err_msg.lower() for k in ("response_format", "json_object", "format")):
                    logger.warning(f"Model '{model}' does not support response_format='json_object'. Retrying without it...")
                    supports_json_format = False
                    continue
                raise RuntimeError(
                    f"{provider_name} returned HTTP 400 Bad Request on model '{model}'.\n"
                    f"Request config sent:\n{json.dumps(request_config, indent=2)}\n"
                    f"Underlying error message:\n{bad_req}"
                ) from bad_req

            except AuthenticationError as auth_err:
                auth_note = ""
                if not is_openrouter and api_key.startswith("sk-or-"):
                    auth_note = " (Note: Your API key appears to be from OpenRouter 'sk-or-v1-'. For NVIDIA NIM use an 'nvapi-' key from build.nvidia.com)."
                raise RuntimeError(
                    f"{provider_name} Authentication failed (HTTP 401 Unauthorized).\n"
                    f"Please verify NVIDIA_API_KEY in your .env file.{auth_note}\n"
                    f"Underlying error message: {auth_err}"
                ) from auth_err

            except NotFoundError as not_found_err:
                raise RuntimeError(
                    f"{provider_name} model '{model}' not found (HTTP 404).\n"
                    f"Please check NVIDIA_VISION_MODEL in your .env file.\n"
                    f"Underlying error: {not_found_err}"
                ) from not_found_err

            except RateLimitError as rl_err:
                if attempt < max_retries:
                    wait_sec = min(RATE_LIMIT_BASE_BACKOFF * (attempt + 1), 30.0) + random.uniform(0.5, 2.0)
                    logger.warning(f"Rate limit (HTTP 429) on {model} (attempt {attempt + 1}/{max_retries + 1}). Backing off {wait_sec:.1f}s...")
                    time.sleep(wait_sec)
                    continue
                else:
                    raise RuntimeError(
                        f"{provider_name} rate limit (HTTP 429) persisted after {max_retries} retries on model '{model}': {rl_err}"
                    ) from rl_err

            except (APITimeoutError, TimeoutError, socket.timeout) as to_err:
                if attempt < max_retries:
                    wait_sec = 2.0 * (attempt + 1)
                    logger.warning(f"[{provider_name} call timed out (attempt {attempt + 1}/{max_retries + 1}). Retrying in {wait_sec:.0f}s...]")
                    time.sleep(wait_sec)
                    continue
                raise RuntimeError(f"{provider_name} call timed out on model '{model}' ({to_err})") from to_err

            except (APIConnectionError, ssl.SSLError, ConnectionError) as conn_err:
                if attempt < max_retries:
                    wait_sec = 2.0 * (attempt + 1)
                    logger.warning(f"[{provider_name} connection error (attempt {attempt + 1}/{max_retries + 1}): {conn_err}. Retrying in {wait_sec:.0f}s...]")
                    time.sleep(wait_sec)
                    continue
                raise RuntimeError(f"{provider_name} connection failed after {max_retries} retries: {conn_err}") from conn_err

            except Exception as exc:
                err_str = str(exc)
                if "429" in err_str:
                    if attempt < max_retries:
                        wait_sec = min(RATE_LIMIT_BASE_BACKOFF * (attempt + 1), 30.0) + random.uniform(0.5, 2.0)
                        logger.warning(f"[{provider_name} Rate limit (attempt {attempt + 1}/{max_retries + 1}). Backing off {wait_sec:.1f}s...]")
                        time.sleep(wait_sec)
                        continue
                raise RuntimeError(f"{provider_name} call failed on model '{model}': {exc}") from exc

        raise RuntimeError(f"{provider_name} call exhausted {max_retries} retries on model '{model}'.")

    except ImportError:
        pass

    # 2. REST API Fallback using standard library
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if is_openrouter:
        headers["HTTP-Referer"] = "https://github.com/sibirajan777/ai-judiciary"
        headers["X-Title"] = "AI Judiciary"

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message},
                    {"type": "image_url", "image_url": {"url": f"data:{image_mime_type};base64,{image_base64}"}}
                ]
            },
        ],
        "temperature": 0.3,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }

    clean_url = base_url.replace("https://", "").replace("http://", "")
    host = clean_url.split("/")[0]
    subpath = "/" + "/".join(clean_url.split("/")[1:]).strip("/") + "/chat/completions"
    supports_json = True

    for attempt in range(max_retries + 1):
        conn = None
        try:
            data_bytes = json.dumps(payload).encode("utf-8")
            conn = SplitTimeoutHTTPSConnection(host, connect_timeout=CONNECT_TIMEOUT, read_timeout=READ_TIMEOUT)
            conn.request("POST", subpath, body=data_bytes, headers=headers)
            resp = conn.getresponse()
            status_code = resp.status
            resp_body = resp.read().decode("utf-8")

            if status_code == 200:
                data = json.loads(resp_body)
                choices = data.get("choices", [])
                if choices:
                    message = choices[0].get("message", {})
                    content = message.get("content", "")
                    finish_reason = choices[0].get("finish_reason", "")
                    if not content:
                        if finish_reason == "length":
                            raise RuntimeError(
                                f"{provider_name} vision REST response truncated (finish_reason=length). "
                                f"Increase max_tokens or reduce prompt size."
                            )
                        if finish_reason == "content_filter":
                            raise RuntimeError(
                                f"{provider_name} vision REST response filtered by content policy."
                            )
                        raise RuntimeError(
                            f"{provider_name} vision REST returned empty message.content. "
                            f"finish_reason={finish_reason}. "
                            f"This model may not support vision or the requested format."
                        )
                    return content
                raise RuntimeError(f"Unexpected {provider_name} response structure: {resp_body[:200]}")

            if status_code == 400:
                if supports_json and any(k in resp_body.lower() for k in ("response_format", "json_object", "format")):
                    logger.warning(f"Model '{model}' does not support response_format='json_object'. Retrying without it...")
                    supports_json = False
                    payload.pop("response_format", None)
                    continue
                raise RuntimeError(
                    f"{provider_name} REST API returned HTTP 400 Bad Request on model '{model}'.\n"
                    f"Underlying error: {resp_body}"
                )

            if status_code == 401:
                raise RuntimeError(
                    f"{provider_name} REST API Authentication failed (HTTP 401 Unauthorized).\n"
                    f"Please verify NVIDIA_API_KEY in your .env file.\n"
                    f"Underlying error: {resp_body}"
                )

            if status_code == 404:
                raise RuntimeError(
                    f"{provider_name} model '{model}' not found (HTTP 404).\n"
                    f"Underlying error: {resp_body}"
                )

            if status_code == 429:
                if attempt < max_retries:
                    wait_sec = min(RATE_LIMIT_BASE_BACKOFF * (attempt + 1), 30.0) + random.uniform(0.5, 2.0)
                    print(f"\n  [{provider_name} REST Rate limit (HTTP 429). Backing off {wait_sec:.1f}s before retry...]", flush=True)
                    time.sleep(wait_sec)
                    continue
                raise RuntimeError(f"{provider_name} REST API rate limit (HTTP 429) exhausted retries: {resp_body}")

            raise RuntimeError(f"{provider_name} REST API request failed (HTTP {status_code}): {resp_body}")

        except (
            ssl.SSLError,
            socket.timeout,
            TimeoutError,
            http.client.RemoteDisconnected,
            http.client.HTTPException,
            ConnectionError,
            OSError,
        ) as net_err:
            if attempt < max_retries:
                wait_sec = 2.0 * (attempt + 1)
                print(f"\n  [{provider_name} REST SSL/Timeout error: {net_err}. Retrying in {wait_sec:.0f}s...]", flush=True)
                time.sleep(wait_sec)
                continue
            raise RuntimeError(f"{provider_name} REST API call failed after {max_retries} retries: {net_err}") from net_err

        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass

    raise RuntimeError(f"{provider_name} call exhausted {max_retries} retries on model '{model}'.")


async def _call_nvidia_vision_async(system_prompt: str, user_message: str, image_base64: str, image_mime_type: str = "image/png", max_tokens: int = None) -> str:
    """Async version of NVIDIA vision caller using aiohttp."""
    try:
        import aiohttp
    except ImportError:
        raise RuntimeError("aiohttp required for async calls. Install with: pip install aiohttp")

    api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("NVIDIA_API_KEY not configured.")

    api_key = api_key.strip().strip("'\"")
    is_openrouter = api_key.startswith("sk-or-")
    default_base_url = "https://openrouter.ai/api/v1" if is_openrouter else "https://integrate.api.nvidia.com/v1"
    base_url = os.getenv("NVIDIA_BASE_URL", default_base_url).strip()

    model = NVIDIA_VISION_MODEL
    if not is_openrouter and "/" not in model:
        model = f"nvidia/{model}"

    # Resolve max_tokens
    if max_tokens is None:
        max_tokens = NVIDIA_VISION_MAX_TOKENS

    provider_name = "OpenRouter" if is_openrouter else "NVIDIA NIM"
    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if is_openrouter:
        headers["HTTP-Referer"] = "https://github.com/sibirajan777/ai-judiciary"
        headers["X-Title"] = "AI Judiciary"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message},
                    {"type": "image_url", "image_url": {"url": f"data:{image_mime_type};base64,{image_base64}"}}
                ]
            },
        ],
        "temperature": 0.3,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }

    timeout = aiohttp.ClientTimeout(total=CONNECT_TIMEOUT + READ_TIMEOUT, connect=CONNECT_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for attempt in range(MAX_RETRIES + 1):
            try:
                async with session.post(url, headers=headers, json=payload) as resp:
                    status_code = resp.status
                    resp_body = await resp.text()

                    if status_code == 200:
                        data = json.loads(resp_body)
                        choices = data.get("choices", [])
                        if choices:
                            message = choices[0].get("message", {})
                            content = message.get("content", "")
                            finish_reason = choices[0].get("finish_reason", "")
                            if not content:
                                if finish_reason == "length":
                                    raise RuntimeError(
                                        f"{provider_name} async vision response truncated (finish_reason=length). "
                                        f"Increase max_tokens or reduce prompt size."
                                    )
                                if finish_reason == "content_filter":
                                    raise RuntimeError(
                                        f"{provider_name} async vision response filtered by content policy."
                                    )
                                raise RuntimeError(
                                    f"{provider_name} async vision returned empty message.content. "
                                    f"finish_reason={finish_reason}. "
                                    f"This model may not support vision or the requested format."
                                )
                            return content
                        raise RuntimeError(f"Unexpected {provider_name} response: {resp_body[:200]}")

                    if status_code == 400:
                        if "response_format" in resp_body.lower() or "json_object" in resp_body.lower():
                            if attempt < MAX_RETRIES:
                                payload.pop("response_format", None)
                                print(f"\n  [Notice: {provider_name} model doesn't support response_format, retrying...]", flush=True)
                                continue
                        raise RuntimeError(f"{provider_name} REST API HTTP 400: {resp_body}")
                    if status_code == 401:
                        raise RuntimeError(f"{provider_name} Authentication failed (HTTP 401): {resp_body}")
                    if status_code == 404:
                        raise RuntimeError(f"{provider_name} model '{model}' not found (HTTP 404): {resp_body}")
                    if status_code == 429:
                        if attempt < MAX_RETRIES:
                            wait_sec = min(RATE_LIMIT_BASE_BACKOFF * (attempt + 1), 30.0) + random.uniform(0.5, 2.0)
                            print(f"\n  [{provider_name} Rate limit (429), backing off {wait_sec:.1f}s...]", flush=True)
                            await asyncio.sleep(wait_sec)
                            continue
                        raise RuntimeError(f"{provider_name} rate limit exhausted: {resp_body}")

                    raise RuntimeError(f"{provider_name} REST API error (HTTP {status_code}): {resp_body}")

            except (aiohttp.ClientError, asyncio.TimeoutError) as net_err:
                if attempt < MAX_RETRIES:
                    wait_sec = 2.0 * (attempt + 1)
                    print(f"\n  [{provider_name} async error: {net_err}, retrying in {wait_sec:.0f}s...]", flush=True)
                    await asyncio.sleep(wait_sec)
                    continue
                raise RuntimeError(f"{provider_name} async call failed after {MAX_RETRIES} retries: {net_err}") from net_err

    raise RuntimeError(f"{provider_name} async call exhausted {MAX_RETRIES} retries.")


# =============================================================================
# ASYNC SUPPORT (for parallel execution of independent LLM calls)
# =============================================================================

async def _call_gemini_async(system_prompt: str, user_message: str) -> str:
    """Async version of Gemini caller using aiohttp."""
    try:
        import aiohttp
    except ImportError:
        raise RuntimeError("aiohttp required for async calls. Install with: pip install aiohttp")

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not configured.")

    api_key = api_key.strip().strip("'\"")
    model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip().strip("'\"")
    if not model or model in ("gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.5-flash", "gemini-1.5-flash"):
        model = "gemini-3.5-flash-lite"

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_message}]}],
        "generationConfig": {"temperature": 0.3, "responseMimeType": "application/json", "maxOutputTokens": 1500},
    }

    timeout = aiohttp.ClientTimeout(total=CONNECT_TIMEOUT + READ_TIMEOUT, connect=CONNECT_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for attempt in range(MAX_RETRIES + 1):
            try:
                async with session.post(url, headers=headers, json=payload) as resp:
                    status_code = resp.status
                    resp_body = await resp.text()

                    if status_code == 200:
                        data = json.loads(resp_body)
                        candidates = data.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            if parts:
                                return parts[0].get("text", "")
                        raise RuntimeError(f"Gemini API returned 200 but no candidates: {resp_body[:200]}")

                    if status_code == 400:
                        raise RuntimeError(f"Gemini API HTTP 400: {resp_body}")
                    if status_code == 404:
                        raise RuntimeError(f"Gemini model '{model}' not found (HTTP 404): {resp_body}")
                    if status_code == 429:
                        if attempt < MAX_RETRIES:
                            wait_sec = min(RATE_LIMIT_BASE_BACKOFF * (attempt + 1), 30.0) + random.uniform(0.5, 2.0)
                            print(f"\n  [Gemini Rate limit (429), backing off {wait_sec:.1f}s...]", flush=True)
                            await asyncio.sleep(wait_sec)
                            continue
                        raise RuntimeError(f"Gemini rate limit exhausted: {resp_body}")
                    if status_code == 503:
                        if attempt < MAX_RETRIES:
                            await asyncio.sleep(10.0)
                            continue

                    raise RuntimeError(f"Gemini API error (HTTP {status_code}): {resp_body}")

            except (aiohttp.ClientError, asyncio.TimeoutError) as net_err:
                if attempt < MAX_RETRIES:
                    wait_sec = 2.0 * (attempt + 1)
                    print(f"\n  [Gemini async error: {net_err}, retrying in {wait_sec:.0f}s...]", flush=True)
                    await asyncio.sleep(wait_sec)
                    continue
                raise RuntimeError(f"Gemini async call failed after {MAX_RETRIES} retries: {net_err}") from net_err

    raise RuntimeError(f"Gemini async call exhausted {MAX_RETRIES} retries.")


async def _call_claude_async(system_prompt: str, user_message: str) -> str:
    """Async version of Claude caller using aiohttp."""
    try:
        import aiohttp
    except ImportError:
        raise RuntimeError("aiohttp required for async calls. Install with: pip install aiohttp")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not configured.")

    api_key = api_key.strip().strip("'\"")
    model = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022").strip().strip("'\"")

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 1500,
        "temperature": 0.3,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_message}],
    }

    timeout = aiohttp.ClientTimeout(total=CONNECT_TIMEOUT + READ_TIMEOUT, connect=CONNECT_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for attempt in range(MAX_RETRIES + 1):
            try:
                async with session.post(url, headers=headers, json=payload) as resp:
                    status_code = resp.status
                    resp_body = await resp.text()

                    if status_code == 200:
                        data = json.loads(resp_body)
                        content = data.get("content", [])
                        if content and "text" in content[0]:
                            return content[0]["text"]
                        raise RuntimeError(f"Unexpected Claude response: {resp_body[:200]}")

                    if status_code == 400:
                        raise RuntimeError(f"Claude API HTTP 400: {resp_body}")
                    if status_code == 429:
                        if attempt < MAX_RETRIES:
                            wait_sec = min(RATE_LIMIT_BASE_BACKOFF * (attempt + 1), 30.0) + random.uniform(0.5, 2.0)
                            print(f"\n  [Claude Rate limit (429), backing off {wait_sec:.1f}s...]", flush=True)
                            await asyncio.sleep(wait_sec)
                            continue
                        raise RuntimeError(f"Claude rate limit exhausted: {resp_body}")

                    raise RuntimeError(f"Claude API error (HTTP {status_code}): {resp_body}")

            except (aiohttp.ClientError, asyncio.TimeoutError) as net_err:
                if attempt < MAX_RETRIES:
                    wait_sec = 2.0 * (attempt + 1)
                    print(f"\n  [Claude async error: {net_err}, retrying in {wait_sec:.0f}s...]", flush=True)
                    await asyncio.sleep(wait_sec)
                    continue
                raise RuntimeError(f"Claude async call failed after {MAX_RETRIES} retries: {net_err}") from net_err

    raise RuntimeError(f"Claude async call exhausted {MAX_RETRIES} retries.")


async def _call_nvidia_async(system_prompt: str, user_message: str) -> str:
    """Async version of NVIDIA/OpenRouter caller using aiohttp."""
    try:
        import aiohttp
    except ImportError:
        raise RuntimeError("aiohttp required for async calls. Install with: pip install aiohttp")

    api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("NVIDIA_API_KEY not configured.")

    api_key = api_key.strip().strip("'\"")
    is_openrouter = api_key.startswith("sk-or-")
    default_base_url = "https://openrouter.ai/api/v1" if is_openrouter else "https://integrate.api.nvidia.com/v1"
    base_url = os.getenv("NVIDIA_BASE_URL", default_base_url).strip()

    raw_model = os.getenv("NVIDIA_MODEL", "").strip().strip("'\"")
    if not raw_model:
        model = "meta-llama/llama-3.3-70b-instruct" if is_openrouter else "nvidia/llama-3.1-nemotron-70b-instruct"
    else:
        model = raw_model
        if not is_openrouter and "/" not in model:
            model = f"nvidia/{model}"

    provider_name = "OpenRouter" if is_openrouter else "NVIDIA NIM"
    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if is_openrouter:
        headers["HTTP-Referer"] = "https://github.com/sibirajan777/ai-judiciary"
        headers["X-Title"] = "AI Judiciary"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.3,
        "max_tokens": 1500,
        "response_format": {"type": "json_object"},
    }

    timeout = aiohttp.ClientTimeout(total=CONNECT_TIMEOUT + READ_TIMEOUT, connect=CONNECT_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for attempt in range(MAX_RETRIES + 1):
            try:
                async with session.post(url, headers=headers, json=payload) as resp:
                    status_code = resp.status
                    resp_body = await resp.text()

                    if status_code == 200:
                        data = json.loads(resp_body)
                        choices = data.get("choices", [])
                        if choices:
                            content = choices[0].get("message", {}).get("content", "")
                            if content:
                                return content
                        raise RuntimeError(f"Unexpected {provider_name} response: {resp_body[:200]}")

                    if status_code == 400:
                        if "response_format" in resp_body.lower() or "json_object" in resp_body.lower():
                            if attempt < MAX_RETRIES:
                                payload.pop("response_format", None)
                                print(f"\n  [Notice: {provider_name} model doesn't support response_format, retrying...]", flush=True)
                                continue
                        raise RuntimeError(f"{provider_name} REST API HTTP 400: {resp_body}")
                    if status_code == 401:
                        raise RuntimeError(f"{provider_name} Authentication failed (HTTP 401): {resp_body}")
                    if status_code == 404:
                        raise RuntimeError(f"{provider_name} model '{model}' not found (HTTP 404): {resp_body}")
                    if status_code == 429:
                        if attempt < MAX_RETRIES:
                            wait_sec = min(RATE_LIMIT_BASE_BACKOFF * (attempt + 1), 30.0) + random.uniform(0.5, 2.0)
                            print(f"\n  [{provider_name} Rate limit (429), backing off {wait_sec:.1f}s...]", flush=True)
                            await asyncio.sleep(wait_sec)
                            continue
                        raise RuntimeError(f"{provider_name} rate limit exhausted: {resp_body}")

                    raise RuntimeError(f"{provider_name} REST API error (HTTP {status_code}): {resp_body}")

            except (aiohttp.ClientError, asyncio.TimeoutError) as net_err:
                if attempt < MAX_RETRIES:
                    wait_sec = 2.0 * (attempt + 1)
                    print(f"\n  [{provider_name} async error: {net_err}, retrying in {wait_sec:.0f}s...]", flush=True)
                    await asyncio.sleep(wait_sec)
                    continue
                raise RuntimeError(f"{provider_name} async call failed after {MAX_RETRIES} retries: {net_err}") from net_err

    raise RuntimeError(f"{provider_name} async call exhausted {MAX_RETRIES} retries.")


async def call_llm_async(system_prompt: str, user_message: str) -> str:
    """
    Async unified LLM caller. Selects provider based on LLM_PROVIDER in .env.
    Use with asyncio.gather() for parallel execution of independent calls.
    
    Example:
        results = await asyncio.gather(
            call_llm_async(prompt1, msg1),
            call_llm_async(prompt2, msg2),
        )
    """
    _load_env_file()
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()

    if provider in ("gemini", "google"):
        return await _call_gemini_async(system_prompt, user_message)
    elif provider in ("claude", "anthropic"):
        return await _call_claude_async(system_prompt, user_message)
    elif provider in ("nvidia", "nim", "openrouter"):
        return await _call_nvidia_async(system_prompt, user_message)
    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER '{provider}'. "
            "Please set LLM_PROVIDER to 'gemini', 'claude', or 'nvidia' in your .env file."
        )


# Import asyncio at module level for async functions
import asyncio


def call_llm(system_prompt: str, user_message: str) -> str:
    """
    Unified LLM caller. Selects provider based on LLM_PROVIDER in .env.
    Supported providers: 'gemini', 'claude', 'nvidia', 'openrouter'.

    Parameters:
        system_prompt (str): Instructions and role definition for the agent.
        user_message (str): The prompt / context message sent to the agent.

    Returns:
        str: Raw text output from the language model.
    """
    _load_env_file()
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()

    if provider in ("gemini", "google"):
        return _call_gemini(system_prompt, user_message)
    elif provider in ("claude", "anthropic"):
        return _call_claude(system_prompt, user_message)
    elif provider in ("nvidia", "nim", "openrouter"):
        return _call_nvidia(system_prompt, user_message)
    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER '{provider}'. "
            "Please set LLM_PROVIDER to 'gemini', 'claude', or 'nvidia' in your .env file."
        )


def call_llm_vision(system_prompt: str, user_message: str, image_base64: str, image_mime_type: str = "image/png", max_tokens: int = None) -> str:
    """
    Unified LLM vision caller. Currently only supports NVIDIA NIM / OpenRouter vision models.
    
    Parameters:
        system_prompt (str): Instructions and role definition for the agent.
        user_message (str): The prompt / context message sent to the agent.
        image_base64 (str): Base64-encoded image data.
        image_mime_type (str): MIME type of the image (image/png, image/jpeg, image/webp).
        max_tokens (int, optional): Maximum tokens for the response. Uses NVIDIA_VISION_MAX_TOKENS if not specified.

    Returns:
        str: Raw text output from the language model.
    """
    _load_env_file()
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()

    if provider in ("nvidia", "nim", "openrouter"):
        return _call_nvidia_vision(system_prompt, user_message, image_base64, image_mime_type, max_tokens)
    else:
        raise ValueError(
            f"Vision not supported for LLM_PROVIDER '{provider}'. "
            "Please set LLM_PROVIDER to 'nvidia' or 'openrouter' in your .env file."
        )


async def call_llm_vision_async(system_prompt: str, user_message: str, image_base64: str, image_mime_type: str = "image/png", max_tokens: int = None) -> str:
    """
    Async unified LLM vision caller. Currently only supports NVIDIA NIM / OpenRouter vision models.
    Use with asyncio.gather() for parallel execution of independent calls.
    
    Parameters:
        system_prompt (str): Instructions and role definition for the agent.
        user_message (str): The prompt / context message sent to the agent.
        image_base64 (str): Base64-encoded image data.
        image_mime_type (str): MIME type of the image (image/png, image/jpeg, image/webp).
        max_tokens (int, optional): Maximum tokens for the response. Uses NVIDIA_VISION_MAX_TOKENS if not specified.

    Returns:
        str: Raw text output from the language model.
    """
    _load_env_file()
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()

    if provider in ("nvidia", "nim", "openrouter"):
        return await _call_nvidia_vision_async(system_prompt, user_message, image_base64, image_mime_type, max_tokens)
    else:
        raise ValueError(
            f"Vision not supported for LLM_PROVIDER '{provider}'. "
            "Please set LLM_PROVIDER to 'nvidia' or 'openrouter' in your .env file."
        )
