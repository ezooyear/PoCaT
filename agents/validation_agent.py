"""
Validation Agent
- 추천 결과가 고객 조건, 상품 조건, 계산 결과와 모순되지 않는지 검증
"""
from langchain_core.messages import AIMessage
from graph.state import AgentState


def validation_agent_node(state: AgentState) -> dict:
    """
    추천 결과 검증 노드.
    초기 버전은 규칙 기반으로 최소 검증을 수행한다.
    """
    issues = []
    checked_items = []

    customer_result = state.get("customer_result")
    calculation_result = state.get("calculation_result")
    product_result = state.get("product_result")
    recommendation_result = state.get("recommendation_result")

    checked_items.append("추천 결과 존재 여부")
    if not recommendation_result:
        issues.append("추천 결과가 없습니다.")

    checked_items.append("고객 정보 조회 결과 존재 여부")
    if not customer_result:
        issues.append("고객 정보 조회 결과가 없습니다.")

    checked_items.append("상품 정보/RAG 결과 존재 여부")
    if not product_result:
        issues.append("상품 정보 또는 RAG 결과가 없습니다.")

    checked_items.append("계산 결과 존재 여부")
    if not calculation_result:
        issues.append("계산 결과가 없습니다.")

    passed = len(issues) == 0

    validation_result = {
        "passed": passed,
        "issues": issues,
        "checked_items": checked_items,
    }

    if passed:
        content = "Validation Agent 검증 완료: 추천 결과에 필요한 기본 근거가 확인되었습니다."
    else:
        content = f"Validation Agent 검증 결과 확인 필요: {issues}"

    return {
        "messages": [AIMessage(content=content)],
        "validation_result": validation_result,
    }