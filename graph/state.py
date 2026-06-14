"""
그래프 State 정의
- AgentState: 멀티 에이전트 전체에서 공유하는 상태
"""

from typing import Annotated, Optional, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """멀티 에이전트 시스템의 공유 상태"""

    # 1. 대화 히스토리
    messages: Annotated[list, add_messages]

    # 2. Supervisor 라우팅 및 실행 관리
    next: Optional[str]

    # 사용자 원문 질문
    user_query: Optional[str]

    # 질문 유형
    # 예: product_info, customer_lookup, financial_analysis,
    # eligibility_check, recommendation, switch_analysis, casual
    task_type: Optional[str]

    # 실행 계획
    # 예: ["customer_agent", "product_agent", "eligibility_agent", "recommend_agent", "validation_agent"]
    plan: Optional[list[str]]

    # 현재 실행 중인 plan 위치
    current_step: Optional[int]

    # 현재 실행 중인 agent 이름
    current_agent: Optional[str]

    # 실행 완료된 agent 목록
    completed_agents: Optional[list[str]]

    # 3. 고객 식별 정보
    member_id: Optional[str]
    customer_id: Optional[int]

    # 4. 에이전트 간 공통 데이터 전달
    context: Optional[dict[str, Any]]

    # 에이전트별 전체 출력 저장소
    agent_outputs: Optional[dict[str, Any]]


    # 5. Agent별 구조화 결과
    customer_result: Optional[dict[str, Any]]
    product_result: Optional[dict[str, Any]]
    financial_result: Optional[dict[str, Any]]
    eligibility_result: Optional[dict[str, Any]]
    recommend_result: Optional[dict[str, Any]]

    ## 오류로 인해 추가 
    # 5-1. Agent 간 handoff용 세부 구조화 결과
    # Product Agent → Eligibility Agent
    product_candidates: Optional[list[dict[str, Any]]]
    # Eligibility Agent → Recommend Agent
    eligibility_results: Optional[list[dict[str, Any]]]
    # Financial Agent / Recommend Agent에서 활용하는 계산 결과
    financial_results: Optional[list[dict[str, Any]]]
    # Recommend Agent → Validation / Supervisor
    recommendation_results: Optional[list[dict[str, Any]]]
    # Eligibility / Recommend에서 재사용하는 고객 파싱 결과
    customer_profile: Optional[dict[str, Any]]
    customer_accounts: Optional[list[dict[str, Any]]]

    # 6. Validation 관련 결과
    validation_result: Optional[dict[str, Any]]

    # Validation 전 임시 답변
    draft_answer: Optional[str]

    # 최종 답변
    final_answer: Optional[str]

    # 7. 로그 및 에러 기록
    agent_logs: Optional[list[dict[str, Any]]]
    errors: Optional[list[dict[str, Any]]]

"""
- agent_outputs는 전체 에이전트 출력 기록용으로 유지

- customer_result, product_result, financial_result, eligibility_result, recommend_result는
Validation Agent가 검증하기 쉽도록 주요 에이전트 결과를 구조화해서 저장하는 필드

- member_id는 현재 코드에서 사용하는 회원 식별자

- customer_id는 DB의 customers 테이블 기준 고객 ID를 명확히 저장하기 위한 필드
"""