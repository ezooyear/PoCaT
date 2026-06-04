"""
에이전트 공통 실행 로직
모든 전문 에이전트가 공유하는 도구 호출 루프를 제공합니다.
"""
from langchain_core.messages import SystemMessage, ToolMessage
from config.settings import get_llm
from graph.state import AgentState


def build_prev_context(agent_outputs: dict, intro: str = "이전 단계에서 확인된 정보 (참고용)", label_map: dict | None = None) -> str:
    """이전 에이전트 결과를 컨텍스트 문자열로 변환합니다."""
    if not agent_outputs:
        return ""
    parts = [f"\n\n## {intro}"]
    for name, output in agent_outputs.items():
        label = (label_map or {}).get(name, name)
        parts.append(f"### {label}\n{output}")
    return "\n".join(parts)


def run_agent_loop(
    state: AgentState,
    system_prompt: str,
    tools: list,
    output_key: str,
    max_iterations: int = 3,
    prev_context_intro: str = "이전 단계에서 확인된 정보 (참고용)",
    prev_context_labels: dict | None = None,
    prev_context_suffix: str = "",
) -> dict:
    """
    공통 에이전트 실행 루프.

    Args:
        state: 현재 AgentState
        system_prompt: 에이전트의 시스템 프롬프트
        tools: 바인딩할 도구 리스트
        output_key: agent_outputs에 저장할 키 (예: "customer_agent")
        max_iterations: 도구 호출 최대 반복 횟수
        prev_context_intro: 이전 컨텍스트 헤더 문구
        prev_context_labels: 이전 에이전트 이름→라벨 매핑
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
    messages = [SystemMessage(content=prompt)] + list(state["messages"])

    # 도구 호출 루프
    for _ in range(max_iterations):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            break
        tool_map = {t.name: t for t in tools}
        for tc in response.tool_calls:
            try:
                result = tool_map[tc["name"]].invoke(tc["args"]) if tc["name"] in tool_map else f"알 수 없는 Tool: {tc['name']}"
            except Exception as e:
                result = f"Tool 실행 오류: {e}"
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
    else:
        # 반복 초과 시 도구 없이 최종 응답
        response = llm.invoke(messages)

    outputs = dict(state.get("agent_outputs") or {})
    outputs[output_key] = response.content
    return {
        "messages": [response],
        "agent_outputs": outputs,
        "current_step": (state.get("current_step") or 0) + 1,
    }
