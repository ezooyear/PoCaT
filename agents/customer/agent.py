"""
Customer Agent - 고객 기본 정보, 가입 계좌, 납입 이력 조회 전담
"""

import re
from typing import Any

from langchain_core.messages import AIMessage

from graph.state import AgentState
from agents.base import (
    build_agent_trace_input,
    build_agent_trace_output,
    make_agent_result,
    run_agent_loop,
)
from agents.customer.prompts import CUSTOMER_SYSTEM_PROMPT
from agents.customer.tools import CUSTOMER_TOOLS
from agents.eligibility.tools import parse_customer_profile
from observability.langfuse import flush_langfuse, langfuse_observation, update_observation


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
        try:
            with langfuse_observation(
                name="customer_agent",
                as_type="span",
                input=build_agent_trace_input(
                    state,
                    agent_name="customer_agent",
                    result_key="customer_result",
                ),
                metadata={"agent": "customer_agent"},
            ) as observation:
                customer_name = _customer_name_from_id(customer_id)

                with langfuse_observation(
                    name="customer_agent.prepare_inputs",
                    as_type="span",
                    input={"customer_id": customer_id},
                    metadata={"agent": "customer_agent", "step": "prepare_inputs"},
                ) as step_observation:
                    update_observation(
                        step_observation,
                        output={"customer_name": customer_name},
                        metadata={"agent": "customer_agent"},
                    )

                with langfuse_observation(
                    name="customer_agent.evaluate",
                    as_type="span",
                    input={"customer_name": customer_name, "tool_count": len(CUSTOMER_TOOLS)},
                    metadata={"agent": "customer_agent", "step": "evaluate"},
                ) as evaluation_observation:
                    tool_results, tool_errors = _run_required_customer_lookups(customer_name)
                    summary = _build_lookup_summary(customer_name, tool_results)
                    customer_profile = _extract_customer_profile(tool_results)
                    update_observation(
                        evaluation_observation,
                        output={
                            "tool_names": [item.get("tool_name") for item in tool_results],
                            "customer_profile": customer_profile,
                            "summary_preview": summary[:500],
                        },
                        metadata={"agent": "customer_agent"},
                    )

                status = "failed" if tool_errors else "success"
                error = "\n".join(tool_errors) if tool_errors else None

                structured_result = make_agent_result(
                    status=status,
                    result={
                        "summary": summary,
                        "tool_results": tool_results,
                        "customer_profile": customer_profile,
                    },
                    evidence=tool_results,
                    error=error,
                )

                update_observation(
                    observation,
                    output=build_agent_trace_output(
                        structured_result,
                        agent_name="customer_agent",
                        state=state,
                        result_key="customer_result",
                        extra_output={
                            "customer_name": customer_name,
                            "customer_profile": customer_profile,
                            "tool_count": len(tool_results),
                        },
                    ),
                    metadata={"agent": "customer_agent", "status": status},
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
                "customer_profile": customer_profile,
            }

            if tool_errors:
                errors = list(state.get("errors") or [])
                errors.append({"agent": "customer_agent", "error": error})
                return_data["errors"] = errors

            return return_data
        finally:
            flush_langfuse()

    return run_agent_loop(
        state=state,
        system_prompt=CUSTOMER_SYSTEM_PROMPT,
        tools=CUSTOMER_TOOLS,
        output_key="customer_agent",
        result_key="customer_result", # 추가 : validation에서 활용
        max_iterations=3,
    )


def _extract_customer_profile(tool_results: list[dict[str, Any]]) -> dict[str, Any]:
    for item in tool_results:
        if item.get("tool_name") != "get_customer_profile":
            continue
        return parse_customer_profile(item.get("tool_result"))
    return {}
