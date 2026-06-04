"""
Supervisor 에이전트 — 계획 수립 / 최종 취합
"""
import json
from langchain_core.messages import SystemMessage
from config.settings import get_llm
from graph.state import AgentState
from agents.supervisor.prompts import (
    SUPERVISOR_PLAN_PROMPT,
    SUPERVISOR_SYNTHESIZE_PROMPT,
    SUPERVISOR_DIRECT_RESPONSE_PROMPT,
)

AGENT_OPTIONS = [
    "customer_agent", "product_agent", "eligibility_agent",
    "financial_agent", "recommend_agent", "validation_agent", "FINISH",
]
MAX_PLAN_LENGTH = 3


def supervisor_node(state: AgentState) -> dict:
    plan = state.get("plan") or []
    if not plan:
        return _plan_mode(state)
    return _synthesize_mode(state)


def _plan_mode(state: AgentState) -> dict:
    llm = get_llm()
    messages = [SystemMessage(content=SUPERVISOR_PLAN_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    plan = _parse_plan(response.content.strip())

    if plan == ["FINISH"]:
        direct = [SystemMessage(content=SUPERVISOR_DIRECT_RESPONSE_PROMPT)] + state["messages"]
        resp = llm.invoke(direct)
        return {"messages": [resp], "next": "FINISH", "plan": ["FINISH"],
                "current_step": 1, "agent_outputs": {}}

    return {"next": plan[0], "plan": plan, "current_step": 0, "agent_outputs": {}}


def _synthesize_mode(state: AgentState) -> dict:
    agent_outputs = state.get("agent_outputs") or {}
    plan = state.get("plan") or []

    if len(plan) <= 1:
        return {"next": "FINISH"}

    llm = get_llm()
    results_text = ""
    label_map = {
        "customer_agent": "고객 정보 조회 결과",
        "product_agent": "상품 조회 결과",
        "eligibility_agent": "자격 판단 결과",
        "financial_agent": "계산/분석 결과",
        "recommend_agent": "추천 결과",
        "validation_agent": "검증 결과",
    }
    for i, name in enumerate(plan, 1):
        output = agent_outputs.get(name, "(결과 없음)")
        results_text += f"\n### {i}. {label_map.get(name, name)}\n{output}\n"

    prompt = SUPERVISOR_SYNTHESIZE_PROMPT.format(agent_results=results_text)
    msgs = [SystemMessage(content=prompt)] + state["messages"]
    resp = llm.invoke(msgs)
    return {"messages": [resp], "next": "FINISH"}


def _parse_plan(text: str) -> list:
    # 이중 중괄호가 생성되는 경우 단일 중괄호로 치환
    if text.startswith("{{") and text.endswith("}}"):
        text = text[1:-1]
    elif "{{" in text and "}}" in text:
        text = text.replace("{{", "{").replace("}}", "}")
        
    try:
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        data = json.loads(text)
        if "plan" in data and isinstance(data["plan"], list):
            valid = [a for a in data["plan"] if a in AGENT_OPTIONS]
            if valid:
                return valid[:MAX_PLAN_LENGTH]
        if "next" in data and data["next"] in AGENT_OPTIONS:
            return [data["next"]]
    except (json.JSONDecodeError, AttributeError, IndexError):
        pass

    # JSON 파싱 실패 시, 텍스트에 포함된 에이전트 목록을 순서대로 추출
    found = []
    for agent in AGENT_OPTIONS:
        if agent == "FINISH":
            continue
        pos = text.lower().find(agent.lower())
        if pos != -1:
            found.append((pos, agent))
            
    if found:
        found.sort()
        return [item[1] for item in found][:MAX_PLAN_LENGTH]
        
    if "finish" in text.lower():
        return ["FINISH"]
        
    return ["FINISH"]
