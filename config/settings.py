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


def get_llm(model: str = None, temperature: float = 0):
    """Create an OpenRouter chat model instance."""

    clear_blocking_proxy()

    from langchain_openrouter import ChatOpenRouter

    if not OPENROUTER_API_KEY.strip():
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to the .env file before running the app."
        )

    return ChatOpenRouter(
        model=model or LLM_MODEL,
        temperature=temperature,
        api_key=OPENROUTER_API_KEY,
    )
