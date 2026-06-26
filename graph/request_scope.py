"""Helpers for building per-request graph inputs."""

from __future__ import annotations


def build_request_messages(graph_prompt: str) -> list[tuple[str, str]]:
    """Return a fresh message list for a single graph invocation."""

    return [("user", graph_prompt)]
