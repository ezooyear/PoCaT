"""
Product 에이전트 (상품 조회) - Tool 기반
- LLM이 필요한 Tool을 선택하여 호출
- 상품 조회, 납입 현황, 만기 확인 등
"""
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage

from config.settings import get_llm
from graph.state import AgentState
from prompts.product_prompt import PRODUCT_SYSTEM_PROMPT
from tools.banking_tools import PRODUCT_TOOLS


def product_agent_node(state: AgentState) -> dict:
    """
    Product 에이전트 노드 함수 (Tool 기반)
    - LLM에게 Tool을 바인딩하고 호출을 위임
    - Tool 실행 결과를 다시 LLM에 전달하여 최종 응답 생성
    """
    llm = get_llm()
    llm_with_tools = llm.bind_tools(PRODUCT_TOOLS)

    # 고객 ID 정보를 시스템 프롬프트에 주입
    member_id = state.get("member_id")
    system_prompt = PRODUCT_SYSTEM_PROMPT
    if member_id:
        system_prompt += f"\n\n## 현재 상담 중인 고객 ID: {member_id}\nTool 호출 시 이 고객 ID를 사용하세요."
    else:
        system_prompt += "\n\n## 현재 상담 중인 고객: 없음 (일반 상담 모드)"

    messages = [SystemMessage(content=system_prompt)] + list(state["messages"])

    # Tool 호출 루프 (최대 3회 반복)
    for _ in range(3):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # Tool 호출이 없으면 최종 응답
        if not response.tool_calls:
            return {"messages": [response]}

        # Tool 실행
        for tool_call in response.tool_calls:
            tool_result = _execute_tool(tool_call)
            messages.append(ToolMessage(
                content=tool_result,
                tool_call_id=tool_call["id"],
            ))

    # 루프 종료 후 최종 응답 생성
    final_response = llm.invoke(messages)
    return {"messages": [final_response]}


def _execute_tool(tool_call: dict) -> str:
    """Tool을 실행하고 결과를 반환합니다."""
    tool_map = {t.name: t for t in PRODUCT_TOOLS}
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    if tool_name in tool_map:
        try:
            return tool_map[tool_name].invoke(tool_args)
        except Exception as e:
            return f"Tool 실행 오류: {e}"
    return f"알 수 없는 Tool: {tool_name}"
