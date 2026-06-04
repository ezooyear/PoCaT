"""
Financial 에이전트 — 이자 계산, 만기, 납입, 중도해지, 비교, 갈아타기
"""
from graph.state import AgentState
from agents.base import run_agent_loop
from agents.financial.prompts import FINANCIAL_SYSTEM_PROMPT
from agents.financial.tools import FINANCIAL_TOOLS


def financial_agent_node(state: AgentState) -> dict:
    return run_agent_loop(
        state=state,
        system_prompt=FINANCIAL_SYSTEM_PROMPT,
        tools=FINANCIAL_TOOLS,
        output_key="financial_agent",
        max_iterations=5,
    )
