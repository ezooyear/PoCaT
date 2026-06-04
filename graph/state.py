"""
그래프 State 정의
- AgentState: 멀티 에이전트 전체에서 공유하는 상태
"""
from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """멀티 에이전트 시스템의 공유 상태"""
    # 대화 히스토리 (add_messages로 자동 누적)
    messages: Annotated[list, add_messages]
    # 다음 라우팅 대상 에이전트 (Supervisor가 설정)
    next: str
    # 현재 대화 중인 회원 ID
    member_id: Optional[str]
    # 작업 컨텍스트 (조회 결과 등 에이전트 간 데이터 전달)
    context: Optional[dict]
    # Supervisor가 수립한 에이전트 실행 계획 (예: ["customer_agent", "recommend_agent"])
    plan: list[str]
    # 현재 plan에서 완료한 단계 수
    current_step: int
    # 각 에이전트의 작업 결과 저장소 (예: {"customer_agent": "고객 정보...", "recommend_agent": "추천 결과..."})
    agent_outputs: dict
