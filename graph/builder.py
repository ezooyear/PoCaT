"""
그래프 빌더
- Phase 1: 단순 챗봇 그래프 (START → coordinator → END)
- Phase 2: Supervisor 패턴 (START → supervisor → agents → supervisor → END)
"""
from langgraph.graph import StateGraph, START, END

from graph.state import AgentState
from agents.supervisor import supervisor_node
from agents.product_agent import product_agent_node
from agents.analysis_agent import analysis_agent_node
from agents.recommend_agent import recommend_agent_node


def _supervisor_router(state: AgentState) -> str:
    """
    Supervisor의 라우팅 결과에 따라 다음 노드를 결정합니다.
    """
    next_agent = state.get("next", "FINISH")
    if next_agent == "FINISH":
        return END
    return next_agent


def build_graph_phase2():
    """
    Phase 2: Supervisor 패턴 그래프

    START → Supervisor → (conditional) → product_agent / analysis_agent / recommend_agent / END
                  ↑                              |
                  └──────────────────────────────┘
    """
    builder = StateGraph(AgentState)

    # 노드 추가
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("product_agent", product_agent_node)
    builder.add_node("analysis_agent", analysis_agent_node)
    builder.add_node("recommend_agent", recommend_agent_node)

    # 엣지 연결
    # 1. START → supervisor
    builder.add_edge(START, "supervisor")

    # 2. supervisor → (conditional) → 각 에이전트 또는 END
    builder.add_conditional_edges(
        "supervisor",
        _supervisor_router,
        {
            "product_agent": "product_agent",
            "analysis_agent": "analysis_agent",
            "recommend_agent": "recommend_agent",
            END: END,
        },
    )

    # 3. 각 에이전트 → END (에이전트가 응답 생성 후 종료)
    builder.add_edge("product_agent", END)
    builder.add_edge("analysis_agent", END)
    builder.add_edge("recommend_agent", END)

    # 컴파일
    graph = builder.compile()
    return graph


def build_graph():
    """현재 Phase에 맞는 그래프를 반환합니다."""
    # Phase 2: Supervisor 패턴
    return build_graph_phase2()
