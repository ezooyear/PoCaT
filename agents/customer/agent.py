"""
Customer Agent - 고객 기본 정보, 가입 계좌, 납입 이력 조회 전담
"""

import re
from typing import Any

from langchain_core.messages import AIMessage

from graph.state import AgentState
from agents.base import make_agent_result, run_agent_loop
from agents.customer.prompts import CUSTOMER_SYSTEM_PROMPT
from agents.customer.tools import CUSTOMER_TOOLS


def _extract_customer_id(state: AgentState) -> int | None:
    customer_id = state.get("customer_id")
    if isinstance(customer_id, int):
        return customer_id

    if isinstance(customer_id, str) and customer_id.strip().isdigit():
        return int(customer_id.strip())

    search_texts: list[str] = []
    if state.get("user_query"):
        search_texts.append(str(state["user_query"]))

    for message in state.get("messages") or []:
        content = getattr(message, "content", message[1] if isinstance(message, tuple) and len(message) > 1 else "")
        if isinstance(content, str):
            search_texts.append(content)

    combined = "\n".join(search_texts)
    patterns = [
        r"고객\s*ID\s*(\d+)",
        r"고객\s*번호\s*(\d+)",
        r"테스트\s*고객번호\s*(\d+)",
        r"customer[_\s-]*id\s*(\d+)",
        r"\bID\s*(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, combined, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))

    return None


def _customer_name_from_id(customer_id: int) -> str:
    return f"고객_{customer_id:03d}"


def _run_required_customer_lookups(customer_name: str) -> tuple[list[dict[str, Any]], list[str]]:
    tool_results: list[dict[str, Any]] = []
    tool_errors: list[str] = []

    for tool in CUSTOMER_TOOLS:
        tool_name = tool.name
        tool_args = {"customer_name": customer_name}
        try:
            result = tool.invoke(tool_args)
        except Exception as error:
            result = f"Tool execution error: {error}"
            tool_errors.append(result)

        tool_results.append(
            {
                "tool_name": tool_name,
                "tool_args": tool_args,
                "tool_result": result,
            }
        )

    return tool_results, tool_errors


def _build_lookup_summary(customer_name: str, tool_results: list[dict[str, Any]]) -> str:
    sections = [f"{customer_name} 고객 정보 조회 결과입니다."]
    for item in tool_results:
        tool_name = item.get("tool_name", "")
        tool_result = item.get("tool_result", "")
        sections.append(f"\n[{tool_name}]\n{tool_result}")
    return "\n".join(sections).strip()


def customer_agent_node(state: AgentState) -> dict:
    """
    Customer Agent 노드.

    고객 기본 정보, 가입 계좌, 납입 이력을 조회하고
    결과를 state["customer_agent"]에 저장합니다.
    """
    customer_id = _extract_customer_id(state)
    if customer_id is not None:
        customer_name = _customer_name_from_id(customer_id)
        tool_results, tool_errors = _run_required_customer_lookups(customer_name)
        summary = _build_lookup_summary(customer_name, tool_results)
        status = "failed" if tool_errors else "success"
        error = "\n".join(tool_errors) if tool_errors else None

        structured_result = make_agent_result(
            status=status,
            result={
                "summary": summary,
                "tool_results": tool_results,
            },
            evidence=tool_results,
            error=error,
        )

        outputs = dict(state.get("agent_outputs") or {})
        outputs["customer_agent"] = structured_result

        completed_agents = list(state.get("completed_agents") or [])
        if "customer_agent" not in completed_agents:
            completed_agents.append("customer_agent")

        response = AIMessage(content=summary)
        return_data = {
            "messages": [response],
            "agent_outputs": outputs,
            "current_step": (state.get("current_step") or 0) + 1,
            "current_agent": "customer_agent",
            "completed_agents": completed_agents,
            "customer_result": structured_result,
        }

        if tool_errors:
            errors = list(state.get("errors") or [])
            errors.append({"agent": "customer_agent", "error": error})
            return_data["errors"] = errors

        return return_data

    return run_agent_loop(
        state=state,
        system_prompt=CUSTOMER_SYSTEM_PROMPT,
        tools=CUSTOMER_TOOLS,
        output_key="customer_agent",
        result_key="customer_result", # 추가 : validation에서 활용
        max_iterations=3,
    )
