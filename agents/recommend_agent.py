"""
Recommend 에이전트 (추천/안내) - Tool 기반
- LLM이 필요한 Tool을 선택하여 호출
- NL2SQL로 DB 조회 + RAG로 약관 검색
"""
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage

from config.settings import get_llm
from graph.state import AgentState
from prompts.recommend_prompt import RECOMMEND_SYSTEM_PROMPT
from tools.banking_tools import RECOMMEND_TOOLS


def recommend_agent_node(state: AgentState) -> dict:
    """
    Recommend 에이전트 노드 함수 (Tool 기반)
    - LLM에게 Tool을 바인딩하고 호출을 위임
    - Tool 실행 결과를 다시 LLM에 전달하여 최종 응답 생성
    """
    llm = get_llm()
    llm_with_tools = llm.bind_tools(RECOMMEND_TOOLS)

    system_prompt = RECOMMEND_SYSTEM_PROMPT
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
    tool_map = {t.name: t for t in RECOMMEND_TOOLS}
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    if tool_name in tool_map:
        try:
            return tool_map[tool_name].invoke(tool_args)
        except Exception as e:
            return f"Tool 실행 오류: {e}"
    return f"알 수 없는 Tool: {tool_name}"
