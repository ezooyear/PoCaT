"""
그래프 State 정의
- AgentState: 멀티 에이전트 전체에서 공유하는 상태
"""

from typing import Annotated, Optional, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """멀티 에이전트 시스템의 공유 상태"""

    # 대화 히스토리
    messages: Annotated[list, add_messages]

    # Supervisor가 결정한 다음 노드
    next: str

    # 현재 상담 중인 고객 ID
    # 기존 코드 호환을 위해 member_id 유지
    member_id: Optional[int]

    # 고객 ID
    customer_id: Optional[int]

    # 사용자 원문 질문
    user_query: Optional[str]

    # 질문 유형
    # 예: product_info, customer_lookup, calculation, recommendation, switch_analysis, casual
    task_type: Optional[str]

    # 실행 계획
    # 예: ["customer_agent", "calculation_agent", "product_agent", "recommend_agent", "validation_agent"]
    plan: Optional[list[str]]

    # 현재 실행 중인 plan 위치
    current_step: Optional[int]

    # 기존 호환용 context
    context: Optional[dict[str, Any]]

    # Agent별 구조화 결과
    customer_result: Optional[dict[str, Any]]
    calculation_result: Optional[dict[str, Any]]
    product_result: Optional[dict[str, Any]]
    recommendation_result: Optional[dict[str, Any]]
    validation_result: Optional[dict[str, Any]]

    # Agent 실행 로그
    agent_logs: Optional[list]

    # 최종 답변
    final_answer: Optional[str]

    # 에러 기록
    errors: Optional[list]