"""
Product 에이전트 — 상품 정보 조회 + 약관 RAG 검색 전담
"""
from graph.state import AgentState
from agents.base import run_agent_loop
from agents.product.prompts import PRODUCT_SYSTEM_PROMPT
from agents.product.tools import PRODUCT_TOOLS


def product_agent_node(state: AgentState) -> dict:
    return run_agent_loop(
        state=state,
        system_prompt=PRODUCT_SYSTEM_PROMPT,
        tools=PRODUCT_TOOLS,
        output_key="product_agent",
        result_key="product_result",
        max_iterations=3,
    )
