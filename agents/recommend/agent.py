"""
Recommend 에이전트 — 목적별/조건별 추천
"""
from graph.state import AgentState
from agents.base import run_agent_loop
from agents.recommend.prompts import RECOMMEND_SYSTEM_PROMPT
from agents.recommend.tools import RECOMMEND_TOOLS


def recommend_agent_node(state: AgentState) -> dict:
    return run_agent_loop(
        state=state,
        system_prompt=RECOMMEND_SYSTEM_PROMPT,
        tools=RECOMMEND_TOOLS,
        output_key="recommend_agent",
        max_iterations=3,
        prev_context_intro="이전 단계에서 확인된 정보 (이 정보를 바탕으로 추천하세요)",
        prev_context_labels={
            "customer_agent": "고객 정보 조회 결과",
            "eligibility_agent": "고객 자격/조건 분석 결과",
            "product_agent": "상품 정보",
            "financial_agent": "계산/분석 결과",
        },
        prev_context_suffix="※ 위 분석 결과를 참고하여 구체적 근거와 함께 추천하세요.",
    )
