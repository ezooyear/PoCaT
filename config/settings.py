"""
Global settings for LLM access.
"""

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv(override=False)

DEFAULT_LLM_MODEL = "openai/gpt-oss-120b:free"


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


def get_openrouter_api_key() -> str:
    return os.environ.get("OPENROUTER_API_KEY", "").strip()


def get_resolved_llm_model(model: str | None = None) -> str:
    if model and str(model).strip():
        return str(model).strip()
    return os.environ.get("LLM_MODEL", DEFAULT_LLM_MODEL).strip() or DEFAULT_LLM_MODEL


def get_llm_settings(model: str | None = None) -> dict[str, Any]:
    resolved_model = get_resolved_llm_model(model)
    api_key = get_openrouter_api_key()

    return {
        "resolved_model": resolved_model,
        "api_key": api_key,
        "has_api_key": bool(api_key),
        "env_model": os.environ.get("LLM_MODEL", "").strip(),
        "env_source_priority": "process_env_over_dotenv",
    }


def log_llm_configuration(prefix: str = "LLM") -> None:
    settings = get_llm_settings()
    print(
        f"[{prefix}] resolved_model={settings['resolved_model']} "
        f"api_key_present={settings['has_api_key']} "
        f"env_priority={settings['env_source_priority']}"
    )


clear_blocking_proxy()


def get_llm(model: str = None, temperature: float = 0, streaming: bool = False):
    """Create an OpenRouter chat model instance."""

    clear_blocking_proxy()

    from langchain_openrouter import ChatOpenRouter

    settings = get_llm_settings(model)
    api_key = settings["api_key"]
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to the .env file before running the app."
        )

    llm = ChatOpenRouter(
        model=settings["resolved_model"],
        temperature=temperature,
        api_key=api_key,
        streaming=streaming,
    )
    setattr(llm, "_resolved_model", settings["resolved_model"])
    return llm
