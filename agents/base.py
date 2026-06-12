"""
에이전트 공통 실행 로직
- 모든 전문 에이전트가 공유하는 도구 호출 루프를 제공합니다.
- 각 Agent 결과를 공통 포맷으로 저장합니다.
"""

from typing import Any, Optional

from langchain_core.messages import SystemMessage, ToolMessage
from config.settings import get_llm
from graph.state import AgentState


def make_agent_result(
    status: str = "success",
    result: Optional[dict[str, Any]] = None,
    evidence: Optional[list[dict[str, Any]]] = None,
    error: Optional[str] = None,
) -> dict[str, Any]:
    """
    Agent 결과를 공통 포맷으로 생성합니다.

    공통 포맷:
    {
        "status": "success" | "failed",
        "result": {},
        "evidence": [],
        "error": None | str
    }
    """
    return {
        "status": status,
        "result": result or {},
        "evidence": evidence or [],
        "error": error,
    }


def build_prev_context(
    agent_outputs: dict,
    intro: str = "이전 단계에서 확인된 정보 (참고용)",
    label_map: Optional[dict] = None,
) -> str:
    """이전 에이전트 결과를 컨텍스트 문자열로 변환합니다."""
    if not agent_outputs:
        return ""

    parts = [f"\n\n## {intro}"]

    for name, output in agent_outputs.items():
        label = (label_map or {}).get(name, name)

        # agent_outputs가 dict 형태인 경우 summary를 우선 사용
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


def run_agent_loop(
    state: AgentState,
    system_prompt: str,
    tools: list,
    output_key: str,
    result_key: Optional[str] = None,
    max_iterations: int = 3,
    prev_context_intro: str = "이전 단계에서 확인된 정보 (참고용)",
    prev_context_labels: Optional[dict] = None,
    prev_context_suffix: str = "",
) -> dict:
    """
    공통 에이전트 실행 루프.

    Args:
        state: 현재 AgentState
        system_prompt: 에이전트의 시스템 프롬프트
        tools: 바인딩할 도구 리스트
        output_key: agent_outputs에 저장할 키
        result_key: 구조화 결과를 저장할 state 키
            예: customer_result, product_result, financial_result,
                eligibility_result, recommend_result
        max_iterations: 도구 호출 최대 반복 횟수
        prev_context_intro: 이전 컨텍스트 헤더 문구
        prev_context_labels: 이전 에이전트 이름 → 라벨 매핑
        prev_context_suffix: 이전 컨텍스트 끝에 붙일 안내문
    """
    llm = get_llm()
    llm_with_tools = llm.bind_tools(tools)

    # 이전 에이전트 결과 컨텍스트 구성
    prev = build_prev_context(
        state.get("agent_outputs") or {},
        intro=prev_context_intro,
        label_map=prev_context_labels,
    )

    if prev_context_suffix and prev:
        prev += f"\n{prev_context_suffix}"

    prompt = system_prompt + prev if prev else system_prompt
    messages = [SystemMessage(content=prompt)] + list(state.get("messages") or [])

    tool_results = []
    tool_errors = []
    response = None

    # 도구 호출 루프
    for _ in range(max_iterations):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        tool_map = {tool.name: tool for tool in tools}

        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]

            try:
                if tool_name in tool_map:
                    result = tool_map[tool_name].invoke(tool_args)
                else:
                    result = f"알 수 없는 Tool: {tool_name}"
                    tool_errors.append(result)

            except Exception as e:
                result = f"Tool 실행 오류: {e}"
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
                    tool_call_id=tc["id"],
                )
            )

    else:
        # 반복 초과 시 도구 없이 최종 응답 생성
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

    # agent_outputs는 전체 에이전트 출력 기록용으로 유지
    outputs = dict(state.get("agent_outputs") or {})
    outputs[output_key] = structured_result

    # 완료된 agent 목록 업데이트
    completed_agents = list(state.get("completed_agents") or [])
    if output_key not in completed_agents:
        completed_agents.append(output_key)

    return_data = {
        "messages": [response],
        "agent_outputs": outputs,
        "current_step": (state.get("current_step") or 0) + 1,
        "current_agent": output_key,
        "completed_agents": completed_agents,
    }

    # customer_result, product_result 등 구조화 결과 저장
    if result_key:
        return_data[result_key] = structured_result

    # 에러 기록 업데이트
    if tool_errors:
        errors = list(state.get("errors") or [])
        errors.append(
            {
                "agent": output_key,
                "error": error,
            }
        )
        return_data["errors"] = errors

    return return_data
