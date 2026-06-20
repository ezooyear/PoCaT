"""
Langfuse helpers.
Keeps observability optional and initializes the client lazily.
"""

from __future__ import annotations

import json
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


def safe_jsonable(value: Any, *, max_depth: int = 8, _seen: set[int] | None = None) -> Any:
    if _seen is None:
        _seen = set()

    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if max_depth <= 0:
        return str(value)

    object_id = id(value)
    if object_id in _seen:
        return "[circular]"

    if isinstance(value, dict):
        _seen.add(object_id)
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            sanitized[str(key)] = safe_jsonable(item, max_depth=max_depth - 1, _seen=_seen)
        _seen.remove(object_id)
        return sanitized

    if isinstance(value, (list, tuple, set)):
        _seen.add(object_id)
        sanitized_items = [
            safe_jsonable(item, max_depth=max_depth - 1, _seen=_seen)
            for item in value
        ]
        _seen.remove(object_id)
        return sanitized_items

    if hasattr(value, "model_dump"):
        try:
            return safe_jsonable(value.model_dump(), max_depth=max_depth - 1, _seen=_seen)
        except Exception:
            return str(value)

    if hasattr(value, "__dict__"):
        try:
            return safe_jsonable(vars(value), max_depth=max_depth - 1, _seen=_seen)
        except Exception:
            return str(value)

    return str(value)


def summarize_for_langfuse(value: Any, *, max_length: int = 800) -> Any:
    value = safe_jsonable(value)

    if value is None:
        return None

    if isinstance(value, (bool, int, float)):
        return value

    if isinstance(value, str):
        return value[:max_length]

    if isinstance(value, dict):
        summarized: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 20:
                summarized["_truncated"] = True
                break
            summarized[str(key)[:80]] = summarize_for_langfuse(item, max_length=max_length)
        return summarized

    if isinstance(value, (list, tuple)):
        summarized_items = [
            summarize_for_langfuse(item, max_length=max_length)
            for item in list(value)[:10]
        ]
        if len(value) > 10:
            summarized_items.append({"_truncated": True, "original_count": len(value)})
        return summarized_items

    return str(value)[:max_length]


def _serialize_preview(value: Any, *, max_length: int = 800) -> str:
    summarized = summarize_for_langfuse(value, max_length=max_length)

    if summarized is None:
        return ""

    if isinstance(summarized, str):
        return summarized[:max_length]

    try:
        return json.dumps(summarized, ensure_ascii=False)[:max_length]
    except Exception:
        return str(summarized)[:max_length]


def is_langfuse_enabled() -> bool:
    if str(os.getenv("LANGFUSE_DISABLED", "")).strip().lower() in {"1", "true", "yes", "on"}:
        return False
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
        observation_context = client.start_as_current_observation(
            name=name,
            as_type=as_type,
            input=input,
            output=output,
            metadata=metadata,
        )
    except Exception:
        yield None
        return

    try:
        with observation_context as observation:
            yield observation
    finally:
        flush_langfuse()


def update_observation(
    observation: Any,
    *,
    input: Any = None,
    output: Any = None,
    metadata: Any = None,
) -> None:
    if observation is None:
        return

    payload: dict[str, Any] = {}

    if input is not None:
        payload["input"] = _serialize_preview(input)
    if output is not None:
        payload["output"] = _serialize_preview(output)
    if metadata is not None:
        payload["metadata"] = summarize_for_langfuse(metadata)

    existing_metadata = payload.get("metadata")
    if not isinstance(existing_metadata, dict):
        existing_metadata = {}

    if input is not None:
        existing_metadata["debug_input_preview"] = _serialize_preview(input)
    if output is not None:
        existing_metadata["debug_output_preview"] = _serialize_preview(output)

    if existing_metadata:
        payload["metadata"] = existing_metadata

    if not payload:
        return

    try:
        observation.update(**payload)
    except Exception:
        pass


def flush_langfuse() -> None:
    client = get_langfuse_client()
    if client is None:
        return

    try:
        client.flush()
    except Exception:
        pass
