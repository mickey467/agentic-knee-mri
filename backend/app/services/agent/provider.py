"""LLM provider: OpenRouter primary + secondary key (NIM removed).

Primary: ChatOpenAI pointed at https://openrouter.ai/api/v1 (OPENROUTER_API_KEY).
Secondary: same endpoint, second key (OPENROUTER_API_KEY_FALLBACK) for quota
relief — engaged automatically on primary capacity errors via
failover.ainvoke_with_failover.

NOTE: models are returned bare (no with_retry/with_fallbacks wrappers)
because the agent binds tools directly on the model object.
"""

import os


class LLMNotConfigured(RuntimeError):
    """Raised when no LLM credentials are available (chat/report endpoints → 503)."""


DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning:free"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Cap output length: free reasoning models burn most of their time on long
# traces, but the cap must leave room for reasoning + full report sections.
MAX_TOKENS = 32768


def _openrouter_chat(api_key: str, model: str) -> "ChatOpenAI":
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        base_url=os.environ.get("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL),
        model=model,
        api_key=api_key,
        temperature=0.6,
        top_p=0.95,
        max_tokens=MAX_TOKENS,
        # Free-tier reasoning can take minutes; default httpx read timeout is too short.
        request_timeout=300,
    )


def get_chat_model():
    """Build the primary chat model for the agent.

    Configure via env:
      OPENROUTER_API_KEY (required), OPENROUTER_MODEL (optional override),
      OPENROUTER_BASE_URL (optional override).
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise LLMNotConfigured(
            "OPENROUTER_API_KEY is not set. Set it (and optionally OPENROUTER_MODEL) "
            "to enable the agentic report and chat endpoints."
        )
    return _openrouter_chat(api_key, os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL))


def has_fallback() -> bool:
    """True when a secondary OpenRouter key is configured."""
    return bool(os.environ.get("OPENROUTER_API_KEY_FALLBACK"))


def get_fallback_model():
    """Secondary-key OpenRouter model (quota relief, same endpoint)."""
    api_key = os.environ.get("OPENROUTER_API_KEY_FALLBACK")
    if not api_key:
        raise LLMNotConfigured(
            "OPENROUTER_API_KEY_FALLBACK is not set; cannot fail over from the primary model."
        )
    return _openrouter_chat(
        api_key, os.environ.get("OPENROUTER_FALLBACK_MODEL", DEFAULT_MODEL)
    )


def is_capacity_error(exc: Exception) -> bool:
    """True for transient capacity/rate-limit errors (fail over)."""
    text = str(exc)
    return (
        "ResourceExhausted" in text
        or "RateLimit" in text
        or "rate_limit" in text
        or "429" in text
        or "[503]" in text
        or " 503" in text
        or "Timeout" in text
    )
