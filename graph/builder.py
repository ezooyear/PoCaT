"""
그래프 빌더 — Supervisor 통제형 A2A 협업 루프
START → Supervisor(계획 수립) → Agent1 → Agent2 → ... → Supervisor(최종 취합) → END
"""
from langgraph.graph import StateGraph, START, END
from graph.state import AgentState
from agents.supervisor import supervisor_node
from agents.customer import customer_agent_node
from agents.product import product_agent_node
from agents.eligibility import eligibility_agent_node
from agents.financial import financial_agent_node
from agents.recommend import recommend_agent_node
from agents.validation import validation_agent_node


def _supervisor_router(state: AgentState) -> str:
    next_agent = state.get("next", "FINISH")
    return END if next_agent == "FINISH" else next_agent


def _agent_router(state: AgentState) -> str:
    plan = state.get("plan") or []
    current_step = state.get("current_step") or 0

    # 계획된 다음 단계가 있으면 해당 에이전트로 직접 토스 (A2A)
    if current_step < len(plan):
        return plan[current_step]

    # 모든 계획이 끝나면 최종 취합을 위해 Supervisor로 반환
    return "supervisor"


def build_graph():
    builder = StateGraph(AgentState)

    # 노드 추가
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("customer_agent", customer_agent_node)
    builder.add_node("product_agent", product_agent_node)
    builder.add_node("eligibility_agent", eligibility_agent_node)
    builder.add_node("financial_agent", financial_agent_node)
    builder.add_node("recommend_agent", recommend_agent_node)
    builder.add_node("validation_agent", validation_agent_node)

    # START → supervisor
    builder.add_edge(START, "supervisor")

    # 1. Supervisor 라우터 (최초 계획 수립 후 첫 에이전트 분기 혹은 END)
    builder.add_conditional_edges(
        "supervisor", _supervisor_router,
        {
            "customer_agent": "customer_agent",
            "product_agent": "product_agent",
            "eligibility_agent": "eligibility_agent",
            "financial_agent": "financial_agent",
            "recommend_agent": "recommend_agent",
            "validation_agent": "validation_agent",
            END: END,
        },
    )

    # 2. 에이전트 간 직접 라우팅 (A2A)
    agent_routing_map = {
        "customer_agent": "customer_agent",
        "product_agent": "product_agent",
        "eligibility_agent": "eligibility_agent",
        "financial_agent": "financial_agent",
        "recommend_agent": "recommend_agent",
        "validation_agent": "validation_agent",
        "supervisor": "supervisor",
    }

    builder.add_conditional_edges("customer_agent", _agent_router, agent_routing_map)
    builder.add_conditional_edges("product_agent", _agent_router, agent_routing_map)
    builder.add_conditional_edges("eligibility_agent", _agent_router, agent_routing_map)
    builder.add_conditional_edges("financial_agent", _agent_router, agent_routing_map)
    builder.add_conditional_edges("recommend_agent", _agent_router, agent_routing_map)
    builder.add_conditional_edges("validation_agent", _agent_router, agent_routing_map)

    return builder.compile()
