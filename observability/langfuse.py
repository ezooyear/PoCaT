"""
Langfuse helpers.
Keeps observability optional and initializes the client lazily.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any

_LANGFUSE_CLIENT: Any = None
_LANGFUSE_HANDLER: Any = None
_LANGFUSE_INIT_FAILED = False


def _get_langfuse_host() -> str:
    return (
        os.getenv("LANGFUSE_HOST")
        or os.getenv("LANGFUSE_BASE_URL")
        or "https://cloud.langfuse.com"
    ).strip()


def _clean_metadata(metadata: Any) -> dict[str, str]:
    if not isinstance(metadata, dict):
        return {}

    cleaned: dict[str, str] = {}
    for key, value in metadata.items():
        if key is None or value is None:
            continue

        normalized_key = "".join(char for char in str(key) if char.isalnum() or char == "_")
        if not normalized_key:
            continue

        cleaned[normalized_key] = str(value)[:200]

    return cleaned


def is_langfuse_enabled() -> bool:
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
    return bool(public_key and secret_key)


def get_langfuse_client():
    global _LANGFUSE_CLIENT, _LANGFUSE_INIT_FAILED

    if not is_langfuse_enabled() or _LANGFUSE_INIT_FAILED:
        return None

    if _LANGFUSE_CLIENT is None:
        try:
            from langfuse import Langfuse

            _LANGFUSE_CLIENT = Langfuse(
                public_key=os.getenv("LANGFUSE_PUBLIC_KEY", "").strip(),
                secret_key=os.getenv("LANGFUSE_SECRET_KEY", "").strip(),
                host=_get_langfuse_host(),
            )
        except Exception:
            _LANGFUSE_INIT_FAILED = True
            return None

    return _LANGFUSE_CLIENT


def get_langfuse_handler():
    global _LANGFUSE_HANDLER

    client = get_langfuse_client()
    if client is None:
        return None

    if _LANGFUSE_HANDLER is None:
        try:
            from langfuse.langchain import CallbackHandler

            _LANGFUSE_HANDLER = CallbackHandler()
        except Exception:
            return None

    return _LANGFUSE_HANDLER


@contextmanager
def langfuse_trace_context(
    *,
    trace_name: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
):
    if get_langfuse_client() is None:
        yield
        return

    attributes: dict[str, Any] = {}

    if trace_name:
        attributes["trace_name"] = trace_name
    if session_id:
        attributes["session_id"] = str(session_id)[:200]
    if user_id:
        attributes["user_id"] = str(user_id)[:200]
    if tags:
        attributes["tags"] = [str(tag)[:200] for tag in tags if tag]

    cleaned_metadata = _clean_metadata(metadata)
    if cleaned_metadata:
        attributes["metadata"] = cleaned_metadata

    if not attributes:
        yield
        return

    from langfuse import propagate_attributes

    with propagate_attributes(**attributes):
        yield


@contextmanager
def langfuse_observation(
    *,
    name: str,
    as_type: str = "span",
    input: Any = None,
    output: Any = None,
    metadata: Any = None,
):
    client = get_langfuse_client()
    if client is None:
        yield None
        return

    try:
        with client.start_as_current_observation(
            name=name,
            as_type=as_type,
            input=input,
            output=output,
            metadata=metadata,
        ) as observation:
            yield observation
    except Exception:
        yield None


def flush_langfuse() -> None:
    client = get_langfuse_client()
    if client is None:
        return

    try:
        client.flush()
    except Exception:
        pass
