"""
Customer Agent - 고객 기본 정보, 가입 계좌, 납입 이력 조회 전담
"""

from graph.state import AgentState
from agents.base import run_agent_loop
from agents.customer.prompts import CUSTOMER_SYSTEM_PROMPT
from agents.customer.tools import CUSTOMER_TOOLS


def customer_agent_node(state: AgentState) -> dict:
    """
    Customer Agent 노드.

    고객 기본 정보, 가입 계좌, 납입 이력을 조회하고
    결과를 state["customer_agent"]에 저장합니다.
    """
    return run_agent_loop(
        state=state,
        system_prompt=CUSTOMER_SYSTEM_PROMPT,
        tools=CUSTOMER_TOOLS,
        output_key="customer_agent",
        max_iterations=3,
    )