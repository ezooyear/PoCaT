"""
Simple Langfuse connectivity and trace smoke test.
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

from observability.langfuse import flush_langfuse, get_langfuse_client, langfuse_observation, langfuse_trace_context


def main() -> int:
    client = get_langfuse_client()
    if client is None:
        print("langfuse_enabled=False")
        return 1

    auth_ok = client.auth_check()
    print(f"auth_check={auth_ok}")
    if not auth_ok:
        return 1

    with langfuse_trace_context(
        trace_name="langfuse-smoke-test",
        session_id="local-smoke-test",
        user_id="codex",
        tags=["smoke-test", "pocat"],
        metadata={"surface": "script", "app": "pocat"},
    ):
        with langfuse_observation(
            name="langfuse_smoke_test",
            as_type="span",
            input={"message": "hello from PoCaT"},
            metadata={"kind": "smoke_test"},
        ) as observation:
            if observation is not None:
                observation.update(
                    output={"status": "ok", "detail": "test trace emitted from local script"}
                )

    flush_langfuse()
    print("trace_sent=True")
    print("trace_name=langfuse-smoke-test")
    print("observation_name=langfuse_smoke_test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
