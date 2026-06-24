"""PoCaT A2A 외부 연결 검증 스크립트.

A2A 서버가 이미 실행 중이라는 전제에서 다음을 확인한다.
1. AgentCard discovery
2. JSON-RPC SendMessage 요청
3. task_id 추출
4. JSON-RPC GetTask 후속 조회
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_PATH = PROJECT_ROOT / "docs" / "a2a_check_result.md"
DEFAULT_BASE_URL = "http://127.0.0.1:9999"
A2A_VERSION = "1.0"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _print_step(ok: bool, label: str, detail: str) -> None:
    status = "[OK]" if ok else "[FAIL]"
    print(f"{status} {label}: {detail}")


def _json_like(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return str(value)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class A2ACheckReport:
    base_url: str
    timestamp_utc: str = field(default_factory=_utc_now)
    steps: list[tuple[bool, str, str]] = field(default_factory=list)
    discovery_url: str | None = None
    card_payload: dict[str, Any] | None = None
    rpc_url: str | None = None
    request_headers: dict[str, str] | None = None
    send_request: dict[str, Any] | None = None
    send_response: Any = None
    get_task_request: dict[str, Any] | None = None
    get_task_response: Any = None
    task_id: str | None = None
    context_id: str | None = None

    def add_step(self, ok: bool, label: str, detail: str) -> None:
        self.steps.append((ok, label, detail))
        _print_step(ok, label, detail)

    @property
    def ok(self) -> bool:
        return all(ok for ok, _, _ in self.steps)

    def to_markdown(self) -> str:
        lines = [
            "# A2A 검증 결과",
            "",
            f"- 실행 시각(UTC): {self.timestamp_utc}",
            f"- Base URL: `{self.base_url}`",
            f"- 전체 상태: `{'PASS' if self.ok else 'FAIL'}`",
            "",
            "## 단계별 결과",
        ]

        for ok, label, detail in self.steps:
            lines.append(f"- {'PASS' if ok else 'FAIL'} | **{label}** | {detail}")

        if self.discovery_url:
            lines.extend(["", "## Discovery URL", f"`{self.discovery_url}`"])

        if self.rpc_url:
            lines.extend(["", "## JSON-RPC Endpoint", f"`{self.rpc_url}`"])

        if self.task_id or self.context_id:
            lines.extend(
                [
                    "",
                    "## 추출된 식별자",
                    f"- task_id: `{self.task_id or '-'}`",
                    f"- context_id: `{self.context_id or '-'}`",
                ]
            )

        if self.request_headers:
            lines.extend(["", "## 요청 헤더", "```json", _json_like(self.request_headers), "```"])

        if self.card_payload:
            lines.extend(["", "## Agent Card", "```json", _json_like(self.card_payload), "```"])

        if self.send_request:
            lines.extend(["", "## SendMessage 요청", "```json", _json_like(self.send_request), "```"])

        if self.send_response is not None:
            lines.extend(["", "## SendMessage 응답", "```json", _json_like(self.send_response), "```"])

        if self.get_task_request:
            lines.extend(["", "## GetTask 요청", "```json", _json_like(self.get_task_request), "```"])

        if self.get_task_response is not None:
            lines.extend(["", "## GetTask 응답", "```json", _json_like(self.get_task_response), "```"])

        return "\n".join(lines) + "\n"


def _candidate_discovery_urls(base_url: str) -> list[str]:
    trimmed = base_url.rstrip("/")
    return [
        f"{trimmed}/.well-known/agent.json",
        f"{trimmed}/.well-known/agent-card.json",
    ]


def _extract_skills(card: dict[str, Any]) -> list[dict[str, Any]]:
    skills = card.get("skills")
    return skills if isinstance(skills, list) else []


def _extract_rpc_url(base_url: str, card: dict[str, Any]) -> str:
    supported = card.get("supportedInterfaces") or card.get("supported_interfaces")
    if isinstance(supported, list):
        for item in supported:
            if isinstance(item, dict) and item.get("url"):
                return str(item["url"]).rstrip("/") or base_url.rstrip("/")

    if card.get("url"):
        return str(card["url"]).rstrip("/") or base_url.rstrip("/")

    return base_url.rstrip("/")


def _build_send_request() -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": "pocat-a2a-sendmessage-check",
        "method": "SendMessage",
        "params": {
            "message": {
                "messageId": str(uuid.uuid4()),
                "role": "ROLE_USER",
                "parts": [
                    {
                        "text": "A2A connectivity check. Please return a short acknowledgement.",
                        "mediaType": "text/plain",
                    }
                ],
            },
            "configuration": {
                "acceptedOutputModes": ["text/plain"],
                "returnImmediately": True,
            },
        },
    }


def _build_get_task_request(task_id: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": "pocat-a2a-gettask-check",
        "method": "GetTask",
        "params": {
            "id": task_id,
        },
    }


def _extract_task_info(payload: Any) -> dict[str, Any]:
    info: dict[str, Any] = {}
    if not isinstance(payload, dict):
        return info

    result = payload.get("result")
    if isinstance(result, dict):
        task = result.get("task", result)
        message = result.get("message")

        if isinstance(task, dict):
            if task.get("id"):
                info["task_id"] = task["id"]
            if task.get("contextId"):
                info["context_id"] = task["contextId"]
            if isinstance(task.get("status"), dict):
                info["status"] = task["status"]

        if isinstance(message, dict):
            if message.get("messageId"):
                info["message_id"] = message["messageId"]
            if message.get("contextId"):
                info["context_id"] = message["contextId"]
            if message.get("taskId"):
                info["task_id"] = message["taskId"]

    if isinstance(payload.get("error"), dict):
        info["error"] = payload["error"]
    return info


def _post_json(session: requests.Session, url: str, payload: dict[str, Any], headers: dict[str, str], timeout_seconds: float) -> requests.Response:
    return session.post(
        url,
        json=payload,
        headers=headers,
        timeout=timeout_seconds,
    )


def _run_check(base_url: str, timeout_seconds: float, report: A2ACheckReport) -> int:
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    card = None
    discovery_error = None
    for url in _candidate_discovery_urls(base_url):
        try:
            response = session.get(url, timeout=timeout_seconds)
            if response.ok:
                card = response.json()
                report.discovery_url = url
                report.card_payload = card
                report.add_step(True, "agent_card_discovery", f"AgentCard 조회 성공 ({url})")
                break
            discovery_error = f"HTTP {response.status_code} from {url}"
        except Exception as error:
            discovery_error = f"{url} -> {error}"

    if card is None:
        report.add_step(False, "agent_card_discovery", f"A2A server not reachable: {discovery_error}")
        return 1

    name = card.get("name", "unknown")
    description = str(card.get("description", "")).strip()
    skills = _extract_skills(card)
    print("")
    print(f"[INFO] Agent name: {name}")
    print(f"[INFO] Description: {description[:200] or '-'}")
    print(f"[INFO] Skill count: {len(skills)}")
    for skill in skills:
        if isinstance(skill, dict):
            print(f"  - {skill.get('id', 'unknown')}: {skill.get('name', '-')}")

    rpc_url = _extract_rpc_url(base_url, card)
    report.rpc_url = rpc_url
    report.add_step(True, "jsonrpc_endpoint_resolved", f"JSON-RPC endpoint 확인 ({rpc_url})")

    headers = {
        "Content-Type": "application/json",
        "A2A-Version": A2A_VERSION,
    }
    report.request_headers = headers

    send_payload = _build_send_request()
    report.send_request = send_payload

    try:
        send_response = _post_json(session, rpc_url or base_url, send_payload, headers, timeout_seconds)
    except Exception as error:
        report.add_step(False, "send_message", f"A2A server not reachable: {error}")
        return 1

    try:
        send_response_payload = send_response.json()
        report.send_response = send_response_payload
    except ValueError:
        report.send_response = send_response.text
        report.add_step(False, "send_message", f"Endpoint responded with non-JSON body (HTTP {send_response.status_code}).")
        return 1

    send_info = _extract_task_info(send_response_payload)
    send_ok = (
        200 <= send_response.status_code < 300
        and isinstance(send_response_payload, dict)
        and send_response_payload.get("jsonrpc") == "2.0"
        and "result" in send_response_payload
        and "error" not in send_response_payload
        and bool(send_info.get("task_id"))
    )
    report.task_id = send_info.get("task_id")
    report.context_id = send_info.get("context_id")

    send_detail = f"HTTP {send_response.status_code}"
    if report.task_id:
        send_detail += f", task_id={report.task_id}"
    if report.context_id:
        send_detail += f", context_id={report.context_id}"
    if send_info.get("status"):
        send_detail += f", lifecycle={send_info['status']}"
    if send_info.get("error"):
        send_detail += f", error={send_info['error']}"
    report.add_step(send_ok, "send_message", send_detail)

    print("")
    print("[INFO] SendMessage response")
    print(_json_like(send_response_payload)[:4000])

    if not send_ok or not report.task_id:
        return 1

    get_task_payload = _build_get_task_request(report.task_id)
    report.get_task_request = get_task_payload

    try:
        get_task_response = _post_json(session, rpc_url or base_url, get_task_payload, headers, timeout_seconds)
    except Exception as error:
        report.add_step(False, "get_task", f"GetTask request failed: {error}")
        return 1

    try:
        get_task_response_payload = get_task_response.json()
        report.get_task_response = get_task_response_payload
    except ValueError:
        report.get_task_response = get_task_response.text
        report.add_step(False, "get_task", f"Endpoint responded with non-JSON body (HTTP {get_task_response.status_code}).")
        return 1

    get_info = _extract_task_info(get_task_response_payload)
    get_ok = (
        200 <= get_task_response.status_code < 300
        and isinstance(get_task_response_payload, dict)
        and get_task_response_payload.get("jsonrpc") == "2.0"
        and "result" in get_task_response_payload
        and "error" not in get_task_response_payload
        and get_info.get("task_id") == report.task_id
    )

    get_detail = f"HTTP {get_task_response.status_code}"
    if get_info.get("task_id"):
        get_detail += f", task_id={get_info['task_id']}"
    if get_info.get("context_id"):
        get_detail += f", context_id={get_info['context_id']}"
    if get_info.get("status"):
        get_detail += f", lifecycle={get_info['status']}"
    if get_info.get("error"):
        get_detail += f", error={get_info['error']}"
    report.add_step(get_ok, "get_task", get_detail)

    print("")
    print("[INFO] GetTask response")
    print(_json_like(get_task_response_payload)[:4000])

    return 0 if report.ok else 1


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check PoCaT A2A server connectivity.")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"A2A server base URL. Default: {DEFAULT_BASE_URL}",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="HTTP timeout in seconds. Default: 10.0",
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
    parser = _build_arg_parser()
    args = parser.parse_args()

    print("PoCaT A2A 외부 연결 검증")
    print(f"대상 Base URL: {args.base_url}")
    print(f"A2A-Version header: {A2A_VERSION}")
    print("")

    report = A2ACheckReport(base_url=args.base_url)
    exit_code = _run_check(args.base_url, args.timeout, report)

    if args.write_report:
        report_path = Path(args.report_path)
        _ensure_parent(report_path)
        report_path.write_text(report.to_markdown(), encoding="utf-8")
        print("")
        print(f"[INFO] 리포트 저장 완료: {report_path}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
