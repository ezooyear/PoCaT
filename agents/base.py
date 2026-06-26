"""
Shared agent execution utilities.
"""

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Optional

from langchain_core.messages import SystemMessage, ToolMessage
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import get_llm
from graph.state import AgentState
from observability.langfuse import (
    flush_langfuse,
    langfuse_observation,
    safe_jsonable,
    summarize_for_langfuse,
    update_observation,
)

LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "20"))
LLM_MAX_RETRIES = max(1, int(os.getenv("LLM_MAX_RETRIES", "3")))
LLM_RETRY_MULTIPLIER = float(os.getenv("LLM_RETRY_MULTIPLIER", "1"))


class LLMInvocationTimeoutError(TimeoutError):
    """Raised when an LLM call exceeds the configured timeout."""


def _build_llm_retry_kwargs() -> dict[str, Any]:
    return {
        "stop": stop_after_attempt(LLM_MAX_RETRIES),
        "wait": wait_exponential(
            multiplier=LLM_RETRY_MULTIPLIER,
            min=LLM_RETRY_MULTIPLIER,
            max=max(LLM_RETRY_MULTIPLIER * 4, LLM_RETRY_MULTIPLIER),
        ),
        "reraise": True,
    }


async def _ainvoke_with_retry(llm: Any, messages: list[Any]) -> Any:
    @retry(**_build_llm_retry_kwargs())
    async def _call() -> Any:
        try:
            return await asyncio.wait_for(
                llm.ainvoke(messages),
                timeout=LLM_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as error:
            raise LLMInvocationTimeoutError(
                f"LLM ainvoke timeout after {LLM_TIMEOUT_SECONDS} seconds"
            ) from error

    return await _call()


def _invoke_with_retry(llm: Any, messages: list[Any]) -> Any:
    @retry(**_build_llm_retry_kwargs())
    def _call() -> Any:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(llm.invoke, messages)
            try:
                return future.result(timeout=LLM_TIMEOUT_SECONDS)
            except FuturesTimeoutError as error:
                raise LLMInvocationTimeoutError(
                    f"LLM invoke timeout after {LLM_TIMEOUT_SECONDS} seconds"
                ) from error

    return _call()


def make_agent_result(
    status: str = "success",
    result: Optional[dict[str, Any]] = None,
    evidence: Optional[list[dict[str, Any]]] = None,
    error: Optional[str] = None,
) -> dict[str, Any]:
    payload = dict(result or {})
    summary = payload.get("summary")

    output: dict[str, Any] = {
        "status": status,
        "summary": summary,
        "result": payload,
        "evidence": evidence or [],
        "error": error,
    }

    for key, value in payload.items():
        if key in output:
            continue
        output[key] = value

    return output


def _get_nested_value(data: Any, path: list[str]) -> tuple[Any, bool]:
    value = data
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None, False
        value = value[key]
    return value, True


def _find_value_by_paths(data: Any, paths: list[list[str]]) -> dict[str, Any]:
    for path in paths:
        value, found = _get_nested_value(data, path)
        if found:
            return {"data": value, "source": ".".join(path)}
    return {"data": None, "source": ""}


def get_customer_profile(state: dict[str, Any]) -> dict[str, Any]:
    paths = [
        ["customer_result", "customer_profile"],
        ["customer_result", "result", "customer_profile"],
        ["agent_outputs", "customer_agent", "customer_profile"],
        ["agent_outputs", "customer_agent", "result", "customer_profile"],
    ]
    result = _find_value_by_paths(state, paths)
    return result


def get_product_candidates(state: dict[str, Any]) -> dict[str, Any]:
    paths = [
        ["product_result", "products"],
        ["product_result", "result", "products"],
        ["product_result", "result", "product_candidates"],
        ["agent_outputs", "product_agent", "product_result", "products"],
        ["agent_outputs", "product_agent", "product_result", "result", "products"],
        ["agent_outputs", "product_agent", "result", "products"],
        ["agent_outputs", "product_agent", "result", "product_candidates"],
    ]
    return _find_value_by_paths(state, paths)


def get_financial_calculations(state: dict[str, Any]) -> dict[str, Any]:
    paths = [
        ["financial_result", "calculations"],
        ["financial_result", "result", "calculations"],
        ["agent_outputs", "financial_agent", "financial_result", "calculations"],
        ["agent_outputs", "financial_agent", "financial_result", "result", "calculations"],
        ["agent_outputs", "financial_agent", "result", "calculations"],
    ]
    return _find_value_by_paths(state, paths)


def get_recommendations(state: dict[str, Any]) -> dict[str, Any]:
    paths = [
        ["recommend_result", "recommendations"],
        ["recommend_result", "result", "recommendations"],
        ["agent_outputs", "recommend_agent", "recommend_result", "recommendations"],
        ["agent_outputs", "recommend_agent", "recommend_result", "result", "recommendations"],
        ["agent_outputs", "recommend_agent", "result", "recommendations"],
    ]
    return _find_value_by_paths(state, paths)


def get_validation_result(state: dict[str, Any]) -> dict[str, Any]:
    paths = [
        ["validation_result"],
        ["agent_outputs", "validation_agent", "validation_result"],
    ]
    return _find_value_by_paths(state, paths)


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


def _to_float_or_none(value: Any) -> float | None:
    if value in (None, "", []):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_tool_args(tool_name: str, tool_args: Any) -> Any:
    if tool_name != "calculate_interest" or not isinstance(tool_args, dict):
        return tool_args

    normalized_args = dict(tool_args)
    principal = _to_float_or_none(normalized_args.get("principal"))
    if principal is not None and principal > 0:
        return normalized_args

    monthly_payment = _to_float_or_none(normalized_args.get("monthly_payment")) or 0.0
    months = _to_float_or_none(normalized_args.get("months")) or 0.0

    if monthly_payment > 0 and months > 0:
        normalized_args["principal"] = monthly_payment * months
        return normalized_args

    normalized_args["principal"] = principal or 0.0
    return normalized_args


async def run_agent_loop_async(
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
            name=span_name or output_key,
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
                response = await _ainvoke_with_retry(llm_with_tools, messages)
                messages.append(response)

                if not response.tool_calls:
                    break

                tool_map = {tool.name: tool for tool in tools}

                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = _normalize_tool_args(tool_name, tool_call["args"])

                    try:
                        if tool_name in tool_map:
                            result = await tool_map[tool_name].ainvoke(tool_args)
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
                response = await _ainvoke_with_retry(llm, messages)

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
            name=span_name or output_key,
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
                response = _invoke_with_retry(llm_with_tools, messages)
                messages.append(response)

                if not response.tool_calls:
                    break

                tool_map = {tool.name: tool for tool in tools}

                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = _normalize_tool_args(tool_name, tool_call["args"])

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
                response = _invoke_with_retry(llm, messages)

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
