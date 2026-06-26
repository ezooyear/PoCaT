"""
Global settings for LLM access.
"""

import os

from dotenv import load_dotenv

load_dotenv()


def clear_blocking_proxy() -> None:
    """Remove local dummy proxy values that break outbound LLM requests."""

    blocked_values = {
        "http://127.0.0.1:9",
        "https://127.0.0.1:9",
        "http://localhost:9",
        "https://localhost:9",
    }

    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        value = os.environ.get(key, "").strip().rstrip("/")
        if value in blocked_values:
            os.environ.pop(key, None)


clear_blocking_proxy()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b:free")
LLM_MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "4096"))


def _resolve_max_output_tokens() -> int:
    return max(256, min(8192, LLM_MAX_OUTPUT_TOKENS))


def _build_openrouter_llm_kwargs(
    *,
    model_name: str,
    temperature: float,
    streaming: bool,
) -> dict:
    max_output_tokens = _resolve_max_output_tokens()

    kwargs = {
        "model": model_name,
        "temperature": temperature,
        "api_key": OPENROUTER_API_KEY,
        "streaming": streaming,
        # OpenRouter-compatible cap for all routed models.
        "max_tokens": max_output_tokens,
    }

    return kwargs


def get_llm(model: str = None, temperature: float = 0, streaming: bool = False):
    """Create an OpenRouter chat model instance."""

    clear_blocking_proxy()

    from langchain_openrouter import ChatOpenRouter

    if not OPENROUTER_API_KEY.strip():
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to the .env file before running the app."
        )

    model_name = model or LLM_MODEL

    return ChatOpenRouter(
        **_build_openrouter_llm_kwargs(
            model_name=model_name,
            temperature=temperature,
            streaming=streaming,
        )
    )
