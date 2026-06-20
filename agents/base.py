"""
Shared agent execution utilities.
"""

import time
from typing import Any, Optional

from langchain_core.messages import SystemMessage, ToolMessage

from config.settings import get_llm
from graph.state import AgentState
from observability.langfuse import (
    flush_langfuse,
    langfuse_observation,
    safe_jsonable,
    summarize_for_langfuse,
    update_observation,
)


def make_agent_result(
    status: str = "success",
    result: Optional[dict[str, Any]] = None,
    evidence: Optional[list[dict[str, Any]]] = None,
    error: Optional[str] = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "result": result or {},
        "evidence": evidence or [],
        "error": error,
    }


def build_prev_context(
    agent_outputs: dict,
    intro: str = "이전 단계에서 확인한 정보",
    label_map: Optional[dict] = None,
) -> str:
    if not agent_outputs:
        return ""

    parts = [f"\n\n## {intro}"]

    for name, output in agent_outputs.items():
        label = (label_map or {}).get(name, name)

        if isinstance(output, dict):
            summary = output.get("summary")
            if summary is None and isinstance(output.get("result"), dict):
                summary = output["result"].get("summary")
            if summary is None:
                summary = str(output)
        else:
            summary = str(output)

        parts.append(f"### {label}\n{summary}")

    return "\n".join(parts)


def build_agent_trace_input(
    state: AgentState,
    *,
    agent_name: str,
    result_key: Optional[str] = None,
    max_iterations: Optional[int] = None,
) -> dict[str, Any]:
    return {
        "agent_name": agent_name,
        "result_key": result_key,
        "max_iterations": max_iterations,
        "user_query": state.get("user_query"),
        "task_type": state.get("task_type"),
        "current_step": state.get("current_step"),
        "plan": state.get("plan"),
        "completed_agents": state.get("completed_agents"),
        "message_count": len(state.get("messages") or []),
    }


def build_agent_trace_output(
    agent_result: dict[str, Any],
    *,
    agent_name: str,
    state: AgentState,
    result_key: Optional[str] = None,
    extra_output: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    result_payload = agent_result.get("result", {}) if isinstance(agent_result, dict) else {}
    summary = result_payload.get("summary") if isinstance(result_payload, dict) else None

    payload: dict[str, Any] = {
        "agent_name": agent_name,
        "current_step": (state.get("current_step") or 0) + 1,
        "result_key": result_key,
        "status": agent_result.get("status") if isinstance(agent_result, dict) else None,
        "error": agent_result.get("error") if isinstance(agent_result, dict) else None,
        "summary": summary,
        "input_summary": summarize_for_langfuse(build_agent_trace_input(state, agent_name=agent_name, result_key=result_key)),
        "output_summary": summarize_for_langfuse(summary),
        "completed_agents": state.get("completed_agents"),
        "plan": state.get("plan"),
        "result": result_payload,
        "evidence_preview": agent_result.get("evidence") if isinstance(agent_result, dict) else None,
    }

    if extra_output:
        payload.update(extra_output)

    return payload


def run_agent_loop(
    state: AgentState,
    system_prompt: str,
    tools: list,
    output_key: str,
    result_key: Optional[str] = None,
    max_iterations: int = 3,
    prev_context_intro: str = "이전 단계에서 확인한 정보",
    prev_context_labels: Optional[dict] = None,
    prev_context_suffix: str = "",
    span_name: Optional[str] = None,
) -> dict:
    structured_result: dict[str, Any] | None = None
    tool_results: list[dict[str, Any]] = []
    tool_errors: list[str] = []
    response = None

    try:
        with langfuse_observation(
            name=output_key,
            as_type="span",
            input=build_agent_trace_input(
                state,
                agent_name=output_key,
                result_key=result_key,
                max_iterations=max_iterations,
            ),
            metadata={"agent": output_key},
        ) as observation:
            llm = get_llm()
            llm_with_tools = llm.bind_tools(tools)

            prev = build_prev_context(
                state.get("agent_outputs") or {},
                intro=prev_context_intro,
                label_map=prev_context_labels,
            )

            if prev_context_suffix and prev:
                prev += f"\n{prev_context_suffix}"

            prompt = system_prompt + prev if prev else system_prompt
            messages = [SystemMessage(content=prompt)] + list(state.get("messages") or [])

            for _ in range(max_iterations):
                response = llm_with_tools.invoke(messages)
                messages.append(response)

                if not response.tool_calls:
                    break

                tool_map = {tool.name: tool for tool in tools}

                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]

                    try:
                        if tool_name in tool_map:
                            result = tool_map[tool_name].invoke(tool_args)
                        else:
                            result = f"Unknown tool: {tool_name}"
                            tool_errors.append(result)
                    except Exception as error:
                        result = f"Tool execution error: {error}"
                        tool_errors.append(result)

                    tool_result_record = {
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "tool_result": result,
                    }
                    tool_results.append(tool_result_record)

                    messages.append(
                        ToolMessage(
                            content=str(result),
                            tool_call_id=tool_call["id"],
                        )
                    )
            else:
                response = llm.invoke(messages)

            summary = response.content if response else ""
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

            update_observation(
                observation,
                output=build_agent_trace_output(
                    structured_result,
                    agent_name=output_key,
                    state=state,
                    result_key=result_key,
                    extra_output={
                        "tool_count": len(tool_results),
                        "tool_names": [item.get("tool_name") for item in tool_results[:10]],
                        "tool_error_count": len(tool_errors),
                    },
                ),
                metadata={
                    "agent": output_key,
                    "status": status,
                    "has_tool_errors": bool(tool_errors),
                },
            )
    finally:
        flush_langfuse()

    outputs = dict(state.get("agent_outputs") or {})
    outputs[output_key] = structured_result

    completed_agents = list(state.get("completed_agents") or [])
    if output_key not in completed_agents:
        completed_agents.append(output_key)

    return_data = {
        "messages": [response] if response else [],
        "agent_outputs": outputs,
        "current_step": (state.get("current_step") or 0) + 1,
        "current_agent": output_key,
        "completed_agents": completed_agents,
    }

    if result_key:
        return_data[result_key] = structured_result

    if tool_errors:
        errors = list(state.get("errors") or [])
        errors.append({"agent": output_key, "error": structured_result.get("error")})
        return_data["errors"] = errors

    return return_data
