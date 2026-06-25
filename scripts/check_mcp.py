"""PoCaT MCP 외부 연결 검증 스크립트.

MCP 서버가 이미 실행 중이라는 전제에서 외부 client 관점의 연결 상태를 확인한다.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_PATH = PROJECT_ROOT / "docs" / "mcp_check_result.md"


def _load_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(PROJECT_ROOT / ".env.example")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _print_step(ok: bool, label: str, detail: str) -> None:
    status = "[OK]" if ok else "[FAIL]"
    print(f"{status} {label}: {detail}")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _json_like(value: Any) -> str:
    import json

    try:
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return str(value)


@dataclass
class CheckReport:
    server_url: str
    timestamp_utc: str = field(default_factory=_utc_now)
    steps: list[tuple[bool, str, str]] = field(default_factory=list)
    tool_names: list[str] = field(default_factory=list)
    sample_tool_name: str | None = None
    sample_tool_args: dict[str, Any] | None = None
    sample_tool_output: str | None = None
    initialize_payload: dict[str, Any] | None = None

    def add_step(self, ok: bool, label: str, detail: str) -> None:
        self.steps.append((ok, label, detail))
        _print_step(ok, label, detail)

    @property
    def ok(self) -> bool:
        return all(ok for ok, _, _ in self.steps)

    def to_markdown(self) -> str:
        lines = [
            "# MCP 검증 결과",
            "",
            f"- 실행 시각(UTC): {self.timestamp_utc}",
            f"- 서버 URL: `{self.server_url}`",
            f"- 전체 상태: `{'PASS' if self.ok else 'FAIL'}`",
            "",
            "## 단계별 결과",
        ]

        for ok, label, detail in self.steps:
            lines.append(f"- {'PASS' if ok else 'FAIL'} | **{label}** | {detail}")

        lines.extend(["", "## Tool 목록", f"- Tool 수: {len(self.tool_names)}"])

        for tool_name in self.tool_names:
            lines.append(f"- `{tool_name}`")

        if self.initialize_payload:
            lines.extend(
                [
                    "",
                    "## Initialize 결과",
                    "```json",
                    _json_like(self.initialize_payload),
                    "```",
                ]
            )

        if self.sample_tool_name:
            lines.extend(
                [
                    "",
                    "## 샘플 Tool 호출",
                    f"- Tool: `{self.sample_tool_name}`",
                    f"- Args: `{self.sample_tool_args or {}}`",
                    "",
                    "```text",
                    self.sample_tool_output or "",
                    "```",
                ]
            )

        return "\n".join(lines) + "\n"


def _extract_text_from_call_result(result: Any, types_module: Any) -> str:
    if getattr(result, "isError", False):
        parts = []
        for content in getattr(result, "content", []):
            if isinstance(content, types_module.TextContent):
                parts.append(content.text)
            else:
                parts.append(str(content))
        return "\n".join(parts) if parts else "Tool returned an error with no text payload."

    parts = []
    for content in getattr(result, "content", []):
        if isinstance(content, types_module.TextContent):
            parts.append(content.text)
        else:
            parts.append(str(content))

    if parts:
        return "\n".join(parts)

    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return _json_like(structured)

    return str(result)


def _select_sample_tool(tool_names: list[str]) -> tuple[str | None, dict[str, Any] | None]:
    preferred = [
        ("get_db_schema", {}),
        ("check_db_connection", {}),
        ("execute_select_query", {"sql": "SELECT 1 AS health_check", "max_rows": 1}),
    ]

    for tool_name, tool_args in preferred:
        if tool_name in tool_names:
            return tool_name, tool_args

    if tool_names:
        return tool_names[0], {}

    return None, None


async def _run_check(server_url: str, report: CheckReport) -> int:
    try:
        from mcp import types
        from mcp.client.session import ClientSession
        from mcp.client.streamable_http import streamable_http_client
    except ImportError as error:
        report.add_step(False, "dependency_import", f"MCP SDK not available: {error}")
        return 2

    try:
        try:
            response = requests.get(server_url, timeout=3)
            report.add_step(
                True,
                "server_reachable",
                f"HTTP 사전 확인 응답 수신 (status={response.status_code})",
            )
        except Exception as error:
            report.add_step(False, "server_reachable", f"MCP server not reachable: {error}")
            return 1

        stream_context = streamable_http_client(server_url)
        async with stream_context as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                try:
                    init_result = await session.initialize()
                    report.initialize_payload = (
                        init_result.model_dump()
                        if hasattr(init_result, "model_dump")
                        else {"raw": str(init_result)}
                    )
                    protocol = getattr(init_result, "protocolVersion", "unknown")
                    server_info = getattr(init_result, "serverInfo", None)
                    server_name = getattr(server_info, "name", "unknown") if server_info else "unknown"
                    report.add_step(
                        True,
                        "session_initialize",
                        f"세션 초기화 성공 (protocol={protocol}, server={server_name})",
                    )
                except Exception as error:
                    report.add_step(False, "session_initialize", str(error))
                    return 1

                try:
                    tool_result = await session.list_tools()
                    report.tool_names = [tool.name for tool in getattr(tool_result, "tools", [])]
                    report.add_step(
                        True,
                        "tools_list",
                        f"등록된 tool {len(report.tool_names)}개 조회 성공",
                    )
                    print("[INFO] 조회된 MCP tools")
                    for tool_name in report.tool_names:
                        print(f"  - {tool_name}")
                except Exception as error:
                    report.add_step(False, "tools_list", str(error))
                    return 1

                sample_tool_name, sample_tool_args = _select_sample_tool(report.tool_names)
                report.sample_tool_name = sample_tool_name
                report.sample_tool_args = sample_tool_args

                if not sample_tool_name:
                    report.add_step(False, "sample_tool_call", "호출 가능한 tool이 없습니다.")
                    return 1

                try:
                    sample_result = await session.call_tool(
                        sample_tool_name,
                        arguments=sample_tool_args,
                    )
                    report.sample_tool_output = _extract_text_from_call_result(sample_result, types)
                    is_error = bool(getattr(sample_result, "isError", False))
                    report.add_step(
                        not is_error,
                        "sample_tool_call",
                        f"`{sample_tool_name}` 호출 완료 (args={sample_tool_args or {}})",
                    )
                    print("")
                    print("[INFO] Sample tool output")
                    print(report.sample_tool_output)
                    return 0 if not is_error else 1
                except Exception as error:
                    report.add_step(False, "sample_tool_call", str(error))
                    return 1
    except Exception as error:
        report.add_step(False, "streamable_http_session", f"MCP session setup failed: {error}")
        return 1


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check PoCaT MCP server connectivity.")
    parser.add_argument(
        "--server-url",
        default=os.getenv("MCP_POSTGRES_URL", "http://localhost:8000/mcp"),
        help="MCP server URL. Defaults to MCP_POSTGRES_URL or http://localhost:8000/mcp",
    )
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="Write a markdown report after the check finishes.",
    )
    parser.add_argument(
        "--report-path",
        default=str(DEFAULT_REPORT_PATH),
        help=f"Markdown report path. Default: {DEFAULT_REPORT_PATH}",
    )
    return parser


def main() -> int:
    _load_env()
    parser = _build_arg_parser()
    args = parser.parse_args()

    print("PoCaT MCP 외부 연결 검증")
    print(f"대상 서버: {args.server_url}")
    print("")

    report = CheckReport(server_url=args.server_url)
    exit_code = asyncio.run(_run_check(args.server_url, report))

    if args.write_report:
        report_path = Path(args.report_path)
        _ensure_parent(report_path)
        report_path.write_text(report.to_markdown(), encoding="utf-8")
        print("")
        print(f"[INFO] 리포트 저장 완료: {report_path}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
