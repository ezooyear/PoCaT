"""
그래프 빌더 — Supervisor 통제형 A2A 협업 루프
START → Supervisor(계획 수립) → Agent1 → Agent2 → ... → Supervisor(최종 취합) → END
"""


from langgraph.graph import StateGraph, START, END
from graph.state import AgentState

# 실제 프로젝트 구조에 맞게 import 경로는 조정 필요
from agents.supervisor import supervisor_node
from agents.customer import customer_agent_node
from agents.product import product_agent_node
from agents.eligibility import eligibility_agent_node
from agents.financial import financial_agent_node
from agents.recommend import recommend_agent_node
from agents.validation import validation_agent_node


AGENT_NODES = {
    "customer_agent",
    "product_agent",
    "eligibility_agent",
    "financial_agent",
    "recommend_agent",
    "validation_agent",
}

# retry loop: validation 이후 다시 시작할 수 있는 agent 목록
RETRYABLE_START_AGENTS = {
    "product_agent",
    "eligibility_agent",
    "financial_agent",
    "recommend_agent",
}


def _supervisor_router(state: AgentState) -> str:
    """
    Supervisor가 반환한 next 값에 따라
    첫 agent로 보내거나 최종 종료한다.
    """

    next_agent = state.get("next", "FINISH")

    if next_agent == "FINISH":
        return END

    if next_agent in AGENT_NODES:
        return next_agent

    return END

def _route_after(current_node: str):
    """현재 실행된 agent를 기준으로 다음 node를 결정한다.

    retry loop:
    - validation_agent가 실패했고 retry_start_agent가 있으면 해당 agent부터 재실행
    - retry_start_agent가 없으면 기존처럼 supervisor로 반환
    """
    def router(state: AgentState) -> str:
        plan = state.get("plan") or []

        # retry loop: validation 실패 후 필요한 agent부터 재실행
        if current_node == "validation_agent":
            retry_start_agent = state.get("retry_start_agent")
            awaiting_user_input = bool(state.get("awaiting_user_input"))

            if (
                retry_start_agent in RETRYABLE_START_AGENTS
                and not awaiting_user_input
                and (not plan or retry_start_agent in plan)
            ):
                return retry_start_agent

            return "supervisor"

        if not plan:
            return "supervisor"

        if current_node in plan:
            current_index = plan.index(current_node)
            next_index = current_index + 1

            if next_index < len(plan):
                next_agent = plan[next_index]
                if next_agent in AGENT_NODES:
                    return next_agent

        # plan의 마지막 agent 실행 후 supervisor가 최종 취합
        return "supervisor"

    return router

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

    # Supervisor → 첫 agent 또는 END
    builder.add_conditional_edges(
        "supervisor",
        _supervisor_router,
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

    agent_routing_map = {
        "customer_agent": "customer_agent",
        "product_agent": "product_agent",
        "eligibility_agent": "eligibility_agent",
        "financial_agent": "financial_agent",
        "recommend_agent": "recommend_agent",
        "validation_agent": "validation_agent",
        "supervisor": "supervisor",
    }

    # Agent → 다음 Agent 또는 Supervisor
    builder.add_conditional_edges(
        "customer_agent",
        _route_after("customer_agent"),
        agent_routing_map,
    )

    builder.add_conditional_edges(
        "product_agent",
        _route_after("product_agent"),
        agent_routing_map,
    )

    builder.add_conditional_edges(
        "eligibility_agent",
        _route_after("eligibility_agent"),
        agent_routing_map,
    )

    builder.add_conditional_edges(
        "financial_agent",
        _route_after("financial_agent"),
        agent_routing_map,
    )

    builder.add_conditional_edges(
        "recommend_agent",
        _route_after("recommend_agent"),
        agent_routing_map,
    )

    builder.add_conditional_edges(
        "validation_agent",
        _route_after("validation_agent"),
        agent_routing_map,
    )

    return builder.compile()