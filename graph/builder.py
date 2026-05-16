"""
그래프 빌더
- Phase 1: 단순 챗봇 그래프 (START → coordinator → END)
- Phase 2+: Supervisor 패턴으로 확장
"""
from langgraph.graph import StateGraph, START, END

from graph.state import AgentState
from agents.coordinator import coordinator_node


def build_graph_phase1():
    """
    Phase 1: 기본 챗봇 그래프
    
    START → Coordinator(챗봇) → END
    
    사용자와 자유 대화가 가능한 최소 구성
    """
    builder = StateGraph(AgentState)

    # 노드 추가
    builder.add_node("coordinator", coordinator_node)

    # 엣지 연결
    builder.add_edge(START, "coordinator")
    builder.add_edge("coordinator", END)

    # 컴파일
    graph = builder.compile()
    return graph


def build_graph():
    """현재 Phase에 맞는 그래프를 반환합니다."""
    # Phase 1: 기본 챗봇
    return build_graph_phase1()
