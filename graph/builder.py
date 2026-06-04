"""
그래프 빌더
- Supervisor 패턴
- 기존: START → supervisor → agent → END
- 수정: START → supervisor → agent(s) → validation_agent → supervisor_final → END
"""
from langgraph.graph import StateGraph, START, END

from graph.state import AgentState
from agents.supervisor import supervisor_node
from agents.product_agent import product_agent_node
from agents.analysis_agent import analysis_agent_node
from agents.recommend_agent import recommend_agent_node

# 새로 추가할 노드
from agents.validation_agent import validation_agent_node
from agents.supervisor_final import supervisor_final_node


def _supervisor_router(state: AgentState) -> str:
    """
    Supervisor의 라우팅 결과에 따라 다음 노드를 결정합니다.
    """
    next_agent = state.get("next", "FINISH")

    if next_agent == "FINISH":
        return "supervisor_final"

    return next_agent


def _after_analysis_router(state: AgentState) -> str:
    """
    Analysis Agent 이후 흐름 결정.
    복합 추천/갈아타기 질문이면 Product Agent로 이어가고,
    단순 분석 질문이면 supervisor_final로 이동.
    """
    task_type = state.get("task_type")

    if task_type in ["recommendation", "switch_analysis", "early_termination"]:
        return "product_agent"

    return "supervisor_final"


def _after_product_router(state: AgentState) -> str:
    """
    Product Agent 이후 흐름 결정.
    복합 추천/갈아타기 질문이면 Recommend Agent로 이어가고,
    단순 상품 설명 질문이면 supervisor_final로 이동.
    """
    task_type = state.get("task_type")

    if task_type in ["recommendation", "switch_analysis", "early_termination"]:
        return "recommend_agent"

    return "supervisor_final"


def _after_recommend_router(state: AgentState) -> str:
    """
    Recommend Agent 이후에는 추천 결과 검증을 수행.
    """
    return "validation_agent"


def build_graph():
    """
    수정된 Supervisor 기반 그래프.

    단순 상품 질문:
    START → supervisor → product_agent → supervisor_final → END

    단순 고객/계산 질문:
    START → supervisor → analysis_agent → supervisor_final → END

    복합 추천/갈아타기/중도해지 질문:
    START → supervisor
          → analysis_agent
          → product_agent
          → recommend_agent
          → validation_agent
          → supervisor_final
          → END
    """
    builder = StateGraph(AgentState)

    # 노드 추가
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("product_agent", product_agent_node)
    builder.add_node("analysis_agent", analysis_agent_node)
    builder.add_node("recommend_agent", recommend_agent_node)
    builder.add_node("validation_agent", validation_agent_node)
    builder.add_node("supervisor_final", supervisor_final_node)

    # START → supervisor
    builder.add_edge(START, "supervisor")

    # supervisor → 첫 실행 Agent 또는 supervisor_final
    builder.add_conditional_edges(
        "supervisor",
        _supervisor_router,
        {
            "product_agent": "product_agent",
            "analysis_agent": "analysis_agent",
            "recommend_agent": "recommend_agent",
            "supervisor_final": "supervisor_final",
        },
    )

    # analysis 이후
    builder.add_conditional_edges(
        "analysis_agent",
        _after_analysis_router,
        {
            "product_agent": "product_agent",
            "supervisor_final": "supervisor_final",
        },
    )

    # product 이후
    builder.add_conditional_edges(
        "product_agent",
        _after_product_router,
        {
            "recommend_agent": "recommend_agent",
            "supervisor_final": "supervisor_final",
        },
    )

    # recommend 이후
    builder.add_conditional_edges(
        "recommend_agent",
        _after_recommend_router,
        {
            "validation_agent": "validation_agent",
        },
    )

    # validation 이후 최종 답변 종합
    builder.add_edge("validation_agent", "supervisor_final")

    # 최종 종료
    builder.add_edge("supervisor_final", END)

    graph = builder.compile()
    return graph