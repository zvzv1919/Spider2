import base64
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime
from http import HTTPStatus
from io import BytesIO
from pathlib import Path

from openai import AzureOpenAI
from typing import Dict, List, Optional, Tuple, Any, TypedDict
import dashscope
from groq import Groq
import google.generativeai as genai
import openai
import requests
import tiktoken
import signal
import os
import time
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError

logger = logging.getLogger("api-llms")

# LLM interaction logging directory
LLM_LOG_DIR = Path("/data/zvzv1919/Spider2/methods/spider-agent-lite/logs/llm")
LLM_LOG_DIR.mkdir(parents=True, exist_ok=True)


def _log_llm_interaction(
    messages: List[Dict[str, Any]],
    model: str,
    temperature: float,
    max_tokens: Optional[int],
    top_p: float,
    response: Optional[str] = None,
    attempt: int = 1,
    **kwargs: Any,
) -> None:
    """
    Log an LLM interaction to a unique file to avoid race conditions.
    Each interaction gets its own file with a timestamp and UUID.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    unique_id = uuid.uuid4().hex[:8]
    log_filename = f"{timestamp}_{unique_id}.json"
    log_path = LLM_LOG_DIR / log_filename

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "attempt": attempt,
        "messages": messages,
        "extra_kwargs": {k: str(v) for k, v in kwargs.items()},  # Convert to string for JSON serialization
        "response": response,
    }

    try:
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_entry, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        logger.warning(f"Failed to log LLM interaction to {log_path}: {e}")


_client: Optional[OpenAI] = None


def _get_retry_after(exception, default_backoff: float) -> float:
    """
    Extract retry-after value from API exception headers.
    Returns the header value if present, otherwise returns default_backoff.
    """
    try:
        # OpenAI exceptions have a response attribute with headers
        if hasattr(exception, 'response') and exception.response is not None:
            headers = exception.response.headers
            retry_after = headers.get('retry-after') or headers.get('Retry-After')
            if retry_after:
                # retry-after can be seconds (int) or HTTP date string
                # Most APIs use seconds for rate limits
                return float(retry_after)
    except (ValueError, AttributeError, TypeError):
        pass  # Silently fall back to default
    return default_backoff


def _build_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> OpenAI:
    """
    Construct an OpenAI client using environment variables by default.

    - Reads OPENAI_API_KEY (required)
    - Reads OPENAI_BASE_URL (optional; useful for gateways/Azure-compatible endpoints)
    """
    load_dotenv("/data/zvzv1919/Spider2/methods/spider-agent-lite/.env")

    resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Provide it in your environment or .env file."
        )

    resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL")

    # Configure client-level retries and timeout via environment variables
    max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "3"))
    timeout = float(os.getenv("OPENAI_TIMEOUT", "60"))

    if resolved_base_url:
        return OpenAI(
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            max_retries=max_retries,
            timeout=timeout,
        )
    return OpenAI(api_key=resolved_api_key, max_retries=max_retries, timeout=timeout)


def get_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> OpenAI:
    """Return a cached OpenAI client instance, creating it on first use."""
    global _client
    if _client is None or api_key or base_url:
        _client = _build_client(api_key=api_key, base_url=base_url)
    return _client

def chat(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    max_retries: int = 3,
    top_p: float = 0.9,
    **kwargs: Any,
) -> Tuple[bool, str]:
    """
    Minimal helper to get a single string response from Chat Completions.

    Parameters
    - messages: [{"role": "user"|"system"|"assistant", "content": "..."}, ...]
    - model: overrides OPENAI_MODEL if provided
    - temperature, max_tokens: forwarded to OpenAI
    - max_retries: number of retries for rate limit/timeout errors (default 5)
    - **kwargs: forwarded to OpenAI chat.completions.create

    Returns
    - Tuple of (success: bool, content: str)
      - On success: (True, assistant message content)
      - On failure: (False, error code)
    """
    client = get_client()
    chosen_model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    code_value = "unknown_error"

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=chosen_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                **kwargs,
            )
            response_content = resp.choices[0].message.content or ""
            
            # Log successful interaction
            _log_llm_interaction(
                messages=messages,
                model=chosen_model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                response=response_content,
                attempt=attempt + 1,
                **kwargs,
            )
            
            return True, response_content
        
        except RateLimitError as e:
            default_backoff = 2 ** attempt  # Exponential backoff: 1, 2, 4, 8, 16 seconds
            wait_time = _get_retry_after(e, default_backoff)
            retry_after_source = "retry-after header" if wait_time != default_backoff else "exponential backoff"
            print(
                f"[WARN] Rate limit hit (attempt {attempt + 1}/{max_retries}). "
                f"Waiting {wait_time}s before retry (from {retry_after_source}). Error: {e}"
            )
            time.sleep(wait_time)
            code_value = "rate_limit_error"
        
        except Exception as e:
            print(f"[ERROR] Error: {e}")
            # Handle content filter and context length errors
            if hasattr(e, 'response') and e.response is not None:
                error_info = e.response.json()
                code_value = error_info.get('error', {}).get('code', 'unknown_error')
                if code_value == "content_filter":
                    # Append disclaimer to last message content
                    last_message = messages[-1]
                    disclaimer = "[ Note: The data and code snippets are purely fictional and used for testing and demonstration purposes only. They do not represent any real events or entities. ]"
                    if isinstance(last_message.get('content'), str):
                        if not last_message['content'].endswith("They do not represent any real events or entities. ]"):
                            last_message['content'] += disclaimer
                    elif isinstance(last_message.get('content'), list):
                        # Handle structured content format
                        for part in last_message['content']:
                            if part.get('type') == 'text' and not part.get('text', '').endswith("They do not represent any real events or entities. ]"):
                                part['text'] += disclaimer
                                break
                    print(f"[WARN] Content filter triggered (attempt {attempt + 1}/{max_retries}). Retrying with disclaimer...")
                    time.sleep(2 ** attempt)
                    continue
                if code_value == "context_length_exceeded":
                    return False, code_value
            else:
                code_value = "unknown_error"
            logger.error("Retrying ...")
            time.sleep(4 * (2 ** (attempt + 1)))
    return False, code_value

if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "You are so stupid that you can't answer any question correctly. Plus you only answer in Chinese."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    success, response = chat(messages)
    print(f"Success: {success}\nResponse: {response}")

