"""
Supervisor 에이전트
- 사용자 의도를 분석하여 적절한 전문 에이전트로 라우팅
- FINISH가 선택되면 직접 인사/일상 대화에 응답
"""
import json
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from config.settings import get_llm
from graph.state import AgentState
from prompts.supervisor_prompt import (
    SUPERVISOR_SYSTEM_PROMPT,
    SUPERVISOR_DIRECT_RESPONSE_PROMPT,
)

# 라우팅 가능한 에이전트 목록
AGENT_OPTIONS = ["product_agent", "analysis_agent", "recommend_agent", "FINISH"]


def supervisor_node(state: AgentState) -> dict:
    """
    Supervisor 노드 함수
    - 사용자 메시지를 분석하여 라우팅할 에이전트를 결정
    - FINISH일 경우 직접 응답 생성
    """
    llm = get_llm()

    # 사용자 최신 메시지 추출
    messages = [SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT)] + state["messages"]

    # LLM에게 라우팅 판단 요청
    response = llm.invoke(messages)
    response_text = response.content.strip()

    # JSON 응답 파싱
    next_agent = _parse_routing(response_text)

    # FINISH인 경우: Supervisor가 직접 응답 생성
    if next_agent == "FINISH":
        direct_messages = [
            SystemMessage(content=SUPERVISOR_DIRECT_RESPONSE_PROMPT)
        ] + state["messages"]
        direct_response = llm.invoke(direct_messages)
        return {
            "messages": [direct_response],
            "next": "FINISH",
        }

    # 전문 에이전트로 라우팅
    return {
        "next": next_agent,
    }


def _parse_routing(response_text: str) -> str:
    """
    LLM 응답에서 라우팅 대상을 파싱합니다.
    JSON 형식이 아닌 경우에도 최대한 파싱을 시도합니다.
    """
    # JSON 파싱 시도
    try:
        # 응답에서 JSON 부분만 추출
        text = response_text
        if "```" in text:
            # 코드 블록 안에 JSON이 있는 경우
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        data = json.loads(text)
        next_agent = data.get("next", "FINISH")

        if next_agent in AGENT_OPTIONS:
            return next_agent
    except (json.JSONDecodeError, AttributeError, IndexError):
        pass

    # JSON 파싱 실패 시 텍스트에서 에이전트명 검색
    response_lower = response_text.lower()
    for agent in AGENT_OPTIONS:
        if agent.lower() in response_lower:
            return agent

    # 기본값: FINISH
    return "FINISH"
