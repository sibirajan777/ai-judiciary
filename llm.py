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
from pathlib import Path
from typing import Optional, Dict, Any


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


class SplitTimeoutHTTPSConnection(http.client.HTTPSConnection):
    """
    Standard library HTTPS connection enforcing:
    - 15-second connect timeout (TCP connect + SSL TLS handshake)
    - Standard library default SSL context (ssl.create_default_context(), no custom context)
    - 90-second read timeout (waiting for server generation response)
    """

    def __init__(
        self,
        host: str,
        port: Optional[int] = None,
        connect_timeout: float = 15.0,
        read_timeout: float = 90.0,
        **kwargs,
    ):
        super().__init__(host, port, timeout=connect_timeout, **kwargs)
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout

    def connect(self):
        # 1. Connect TCP socket using 15-second connect timeout
        self.sock = socket.create_connection(
            (self.host, self.port),
            timeout=self.connect_timeout,
            source_address=self.source_address,
        )
        # 2. SSL handshake using standard library default SSL context (no custom context)
        context = ssl.create_default_context()
        self.sock = context.wrap_socket(self.sock, server_hostname=self.host)
        # 3. Switch socket timeout to 90-second read timeout for response streaming
        self.sock.settimeout(self.read_timeout)


def _call_gemini(system_prompt: str, user_message: str) -> str:
    """
    Calls Google Gemini using the standard library HTTPS connection:
    - Uses standard library defaults for SSL (no custom SSL context)
    - 15-second connect timeout (TCP connect + TLS handshake)
    - 90-second read timeout (response generation)
    - Retries up to 3 times on SSL or timeout errors
    - Retries on HTTP 429 with 10-30s backoff
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

    max_retries = 3

    for attempt in range(max_retries + 1):
        conn = None
        try:
            conn = SplitTimeoutHTTPSConnection(host, connect_timeout=15.0, read_timeout=90.0)
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

            # 4. HTTP 429 RATE LIMIT: Back off 10 to 30s
            if status_code == 429:
                if attempt < max_retries:
                    wait_sec = min(10.0 * (attempt + 1), 30.0) + random.uniform(0.5, 2.0)
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
    Calls Anthropic Claude using standard library HTTPS connection with 15s connect timeout and 90s read timeout.
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
    max_retries = 3

    for attempt in range(max_retries + 1):
        conn = None
        try:
            conn = SplitTimeoutHTTPSConnection(host, connect_timeout=15.0, read_timeout=90.0)
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
                    wait_sec = min(10.0 * (attempt + 1), 30.0) + random.uniform(0.5, 2.0)
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

    max_retries = 3
    provider_name = "OpenRouter" if is_openrouter else "NVIDIA NIM"

    request_config: Dict[str, Any] = {
        "provider": provider_name,
        "base_url": base_url,
        "model": model,
        "temperature": 0.3,
        "max_tokens": 1500,
        "response_format": "json_object",
    }

    print(f"\n  [{provider_name} Request Config] -> {json.dumps(request_config, indent=2)}", flush=True)

    # 1. Try with official openai package
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

        default_headers = {}
        if is_openrouter:
            default_headers = {
                "HTTP-Referer": "https://github.com/sibirajan777/ai-judiciary",
                "X-Title": "AI Judiciary",
            }

        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=120.0,
            default_headers=default_headers if default_headers else None,
        )

        supports_json_format = True
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
                        content = response.choices[0].message.content
                        if content:
                            return content
                    raise RuntimeError(f"{provider_name} returned empty response: {response}")

                except BadRequestError as bad_req:
                    err_msg = str(bad_req)
                    # If model rejects response_format, retry immediately without it
                    if supports_json_format and any(k in err_msg.lower() for k in ("response_format", "json_object", "format")):
                        print(
                            f"\n  [Notice: Model '{current_model}' does not support response_format='json_object'. Retrying without it...]",
                            flush=True,
                        )
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
                        print(f"\n  [Notice: Model '{current_model}' unavailable on OpenRouter. Falling back to '{candidate_models[-1]}'...]", flush=True)
                        break
                    raise RuntimeError(
                        f"{provider_name} model '{current_model}' not found (HTTP 404).\n"
                        f"Please check NVIDIA_MODEL in your .env file.\n"
                        f"Underlying error: {not_found_err}"
                    ) from not_found_err

                except RateLimitError as rl_err:
                    if attempt < max_retries:
                        wait_sec = min(10.0 * (attempt + 1), 30.0) + random.uniform(0.5, 2.0)
                        print(
                            f"\n  [Rate limit (HTTP 429) on {current_model} (attempt {attempt + 1}/{max_retries + 1}). "
                            f"Backing off {wait_sec:.1f}s before retry {attempt + 1}/{max_retries}...]",
                            flush=True,
                        )
                        time.sleep(wait_sec)
                        continue
                    else:
                        raise RuntimeError(
                            f"{provider_name} rate limit (HTTP 429) persisted after {max_retries} retries on model '{current_model}': {rl_err}"
                        ) from rl_err

                except (APITimeoutError, TimeoutError, socket.timeout) as to_err:
                    if attempt < max_retries:
                        wait_sec = 2.0 * (attempt + 1)
                        print(f"\n  [{provider_name} call timed out (attempt {attempt + 1}/{max_retries + 1}). Retrying in {wait_sec:.0f}s...]", flush=True)
                        time.sleep(wait_sec)
                        continue
                    raise RuntimeError(f"{provider_name} call timed out after 120s on model '{current_model}' ({to_err})") from to_err

                except (APIConnectionError, ssl.SSLError, ConnectionError) as conn_err:
                    if attempt < max_retries:
                        wait_sec = 2.0 * (attempt + 1)
                        print(f"\n  [{provider_name} connection error (attempt {attempt + 1}/{max_retries + 1}): {conn_err}. Retrying in {wait_sec:.0f}s...]", flush=True)
                        time.sleep(wait_sec)
                        continue
                    raise RuntimeError(f"{provider_name} connection failed after {max_retries} retries: {conn_err}") from conn_err

                except Exception as exc:
                    err_str = str(exc)
                    if "429" in err_str:
                        if attempt < max_retries:
                            wait_sec = min(10.0 * (attempt + 1), 30.0) + random.uniform(0.5, 2.0)
                            print(f"\n  [{provider_name} Rate limit (attempt {attempt + 1}/{max_retries + 1}). Backing off {wait_sec:.1f}s...]", flush=True)
                            time.sleep(wait_sec)
                            continue
                    raise RuntimeError(f"{provider_name} call failed on model '{current_model}': {exc}") from exc

        raise RuntimeError(f"{provider_name} call exhausted {max_retries} retries on model '{model}'.")

    except ImportError:
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
            conn = SplitTimeoutHTTPSConnection(host, connect_timeout=15.0, read_timeout=90.0)
            conn.request("POST", subpath, body=data_bytes, headers=headers)
            resp = conn.getresponse()
            status_code = resp.status
            resp_body = resp.read().decode("utf-8")

            if status_code == 200:
                data = json.loads(resp_body)
                choices = data.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "")
                    if content:
                        return content
                raise RuntimeError(f"Unexpected {provider_name} response structure: {resp_body[:200]}")

            if status_code == 400:
                if supports_json and any(k in resp_body.lower() for k in ("response_format", "json_object", "format")):
                    print(
                        f"\n  [Notice: Model '{model}' does not support response_format='json_object'. Retrying without it...]",
                        flush=True,
                    )
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
                    wait_sec = min(10.0 * (attempt + 1), 30.0) + random.uniform(0.5, 2.0)
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
