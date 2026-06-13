"""
Global settings module.
- Load environment variables
- Initialize the shared LLM factory
"""

import os

from dotenv import load_dotenv

load_dotenv()

from observability.langfuse import get_langfuse_handler


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b:free")


def get_llm(model: str = None, temperature: float = 0):
    """Create a ChatOpenRouter instance with optional Langfuse callbacks."""

    from langchain_openrouter import ChatOpenRouter

    callbacks = []
    handler = get_langfuse_handler()
    if handler is not None:
        callbacks.append(handler)

    return ChatOpenRouter(
        model=model or LLM_MODEL,
        temperature=temperature,
        api_key=OPENROUTER_API_KEY,
        callbacks=callbacks or None,
    )
