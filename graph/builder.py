"""
그래프 빌더
- Supervisor 패턴
- START → supervisor → agent(s) → supervisor_final → END
- plan에 따라 여러 Agent가 순차 실행됨
- 무거운 Agent는 lazy wrapper로 실행 시점에 import
"""

from langgraph.graph import StateGraph, START, END
from graph.state import AgentState


AGENT_NODES = {
    "customer_agent",
    "calculation_agent",
    "product_agent",
    "recommend_agent",
    "validation_agent",
    "supervisor_final",
}


def _supervisor_router(state: AgentState) -> str:
    next_agent = state.get("next")
    plan = state.get("plan") or []

    if not next_agent and plan:
        next_agent = plan[0]

    if next_agent == "FINISH":
        return "supervisor_final"

    if next_agent in AGENT_NODES:
        return next_agent

    if plan and plan[0] in AGENT_NODES:
        return plan[0]

    return "supervisor_final"


def _route_after(current_node: str):
    def router(state: AgentState) -> str:
        plan = state.get("plan") or []

        if not plan:
            return "supervisor_final"

        if current_node in plan:
            current_index = plan.index(current_node)
            next_index = current_index + 1

            if next_index < len(plan):
                next_node = plan[next_index]

                if next_node in AGENT_NODES:
                    return next_node

        return "supervisor_final"

    return router


def supervisor_lazy_node(state: AgentState) -> dict:
    from agents.supervisor import supervisor_node
    result = supervisor_node(state)
    return result if result is not None else {"next": "supervisor_final"}


def customer_lazy_node(state: AgentState) -> dict:
    from agents.customer_agent import customer_agent_node
    result = customer_agent_node(state)
    return result if result is not None else {
        "customer_result": {
            "ok": False,
            "error": "customer_agent_node returned None",
        },
        "errors": ["customer_agent_node returned None"],
    }


def calculation_lazy_node(state: AgentState) -> dict:
    from agents.calculation_agent import calculation_agent_node
    result = calculation_agent_node(state)
    return result if result is not None else {
        "calculation_result": {
            "ok": False,
            "error": "calculation_agent_node returned None",
        },
        "errors": ["calculation_agent_node returned None"],
    }


def product_lazy_node(state: AgentState) -> dict:
    from agents.product_agent import product_agent_node
    result = product_agent_node(state)
    return result if result is not None else {
        "product_result": {
            "ok": False,
            "error": "product_agent_node returned None",
        },
        "errors": ["product_agent_node returned None"],
    }


def recommend_lazy_node(state: AgentState) -> dict:
    from agents.recommend_agent import recommend_agent_node
    result = recommend_agent_node(state)
    return result if result is not None else {
        "recommendation_result": {
            "ok": False,
            "error": "recommend_agent_node returned None",
        },
        "errors": ["recommend_agent_node returned None"],
    }


def validation_lazy_node(state: AgentState) -> dict:
    from agents.validation_agent import validation_agent_node
    result = validation_agent_node(state)
    return result if result is not None else {
        "validation_result": {
            "ok": False,
            "error": "validation_agent_node returned None",
        },
        "errors": ["validation_agent_node returned None"],
    }


def supervisor_final_lazy_node(state: AgentState) -> dict:
    from agents.supervisor_final import supervisor_final_node
    result = supervisor_final_node(state)
    return result if result is not None else {
        "final_answer": "최종 답변 생성 중 오류가 발생했습니다.",
        "errors": ["supervisor_final_node returned None"],
    }


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("supervisor", supervisor_lazy_node)
    builder.add_node("customer_agent", customer_lazy_node)
    builder.add_node("calculation_agent", calculation_lazy_node)
    builder.add_node("product_agent", product_lazy_node)
    builder.add_node("recommend_agent", recommend_lazy_node)
    builder.add_node("validation_agent", validation_lazy_node)
    builder.add_node("supervisor_final", supervisor_final_lazy_node)

    builder.add_edge(START, "supervisor")

    builder.add_conditional_edges(
        "supervisor",
        _supervisor_router,
        {
            "customer_agent": "customer_agent",
            "calculation_agent": "calculation_agent",
            "product_agent": "product_agent",
            "recommend_agent": "recommend_agent",
            "validation_agent": "validation_agent",
            "supervisor_final": "supervisor_final",
        },
    )

    possible_next_nodes = {
        "customer_agent": "customer_agent",
        "calculation_agent": "calculation_agent",
        "product_agent": "product_agent",
        "recommend_agent": "recommend_agent",
        "validation_agent": "validation_agent",
        "supervisor_final": "supervisor_final",
    }

    builder.add_conditional_edges(
        "customer_agent",
        _route_after("customer_agent"),
        possible_next_nodes,
    )

    builder.add_conditional_edges(
        "calculation_agent",
        _route_after("calculation_agent"),
        possible_next_nodes,
    )

    builder.add_conditional_edges(
        "product_agent",
        _route_after("product_agent"),
        possible_next_nodes,
    )

    builder.add_conditional_edges(
        "recommend_agent",
        _route_after("recommend_agent"),
        possible_next_nodes,
    )

    builder.add_conditional_edges(
        "validation_agent",
        _route_after("validation_agent"),
        possible_next_nodes,
    )

    builder.add_edge("supervisor_final", END)

    return builder.compile()