"""
Eligibility 에이전트 — 가입 자격 판단 + 우대금리 확인
"""
from graph.state import AgentState
from agents.base import run_agent_loop
from agents.eligibility.prompts import ELIGIBILITY_SYSTEM_PROMPT
from agents.eligibility.tools import ELIGIBILITY_TOOLS


def eligibility_agent_node(state: AgentState) -> dict:
    return run_agent_loop(
        state=state,
        system_prompt=ELIGIBILITY_SYSTEM_PROMPT,
        tools=ELIGIBILITY_TOOLS,
        output_key="eligibility_agent",
        max_iterations=4,
    )
