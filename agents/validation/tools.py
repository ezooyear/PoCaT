"""
Validation Agent 검증 도구
- Agent 결과의 누락 여부, 공통 포맷, 실행 계획 완료 여부를 검증합니다.
- 세부 검증은 팀원 Agent 결과 구조가 확정된 뒤 보강합니다.
"""

"""
[팀원 코드 공유 이전 버전의 Validation Agent 목표]
1. state.py와 base.py 구조에 맞게 validation_result를 저장한다.
2. 각 Agent 결과가 공통 포맷인지 확인한다.
3. task_type에 따라 필요한 결과가 있는지 확인한다.
4. plan에 포함된 Agent가 완료되었는지 확인한다.
5. errors가 있으면 검증 실패로 기록한다.
6. 추천 결과와 eligibility 결과가 있으면 가능한 범위에서만 가볍게 비교한다


[추가 예정 검증 항목]
- 조건 충돌 검증
- 금리/금액/납입 횟수 일치 검증
- RAG 근거 검증
- 부적절 추천 여부 검증
"""

from typing import Any

from graph.state import AgentState


REQUIRED_RESULT_KEYS = ["status", "result", "evidence", "error"]


REQUIRED_RESULTS_BY_TASK = {
    "customer_lookup": ["customer_result"],
    "product_info": ["product_result"],
    "financial_analysis": ["customer_result", "financial_result"],
    "eligibility_check": ["customer_result", "product_result", "eligibility_result"],
    "recommendation": [
        "customer_result",
        "product_result",
        "eligibility_result",
        "recommend_result",
    ],
    "switch_analysis": [
        "customer_result",
        "product_result",
        "financial_result",
    ],
}


def validate_common_result_format(state: AgentState) -> list[str]:
    """Agent별 구조화 결과가 공통 포맷을 따르는지 검증합니다."""
    issues = []

    result_map = {
        "customer_result": state.get("customer_result"),
        "product_result": state.get("product_result"),
        "financial_result": state.get("financial_result"),
        "eligibility_result": state.get("eligibility_result"),
        "recommend_result": state.get("recommend_result"),
    }

    for result_name, result_value in result_map.items():
        if result_value is None:
            continue

        if not isinstance(result_value, dict):
            issues.append(f"{result_name}가 dict 형식이 아닙니다.")
            continue

        missing_keys = [
            key for key in REQUIRED_RESULT_KEYS
            if key not in result_value
        ]

        if missing_keys:
            issues.append(f"{result_name}에 필수 키가 없습니다: {missing_keys}")

        if result_value.get("status") == "failed":
            issues.append(
                f"{result_name}의 status가 failed입니다. error={result_value.get('error')}"
            )

    return issues


def validate_required_results_by_task(state: AgentState) -> list[str]:
    """task_type에 따라 필요한 Agent 결과가 존재하는지 검증합니다."""
    issues = []

    task_type = state.get("task_type")
    required_results = REQUIRED_RESULTS_BY_TASK.get(task_type, [])

    for result_name in required_results:
        if not state.get(result_name):
            issues.append(
                f"task_type='{task_type}'에 필요한 {result_name}가 없습니다."
            )

    return issues


def validate_plan_completion(state: AgentState) -> list[str]:
    """Supervisor가 세운 plan의 Agent들이 완료되었는지 검증합니다."""
    issues = []

    plan = state.get("plan") or []
    completed_agents = state.get("completed_agents") or []

    for agent_name in plan:
        if agent_name in ["validation_agent", "FINISH", "END"]:
            continue

        if agent_name not in completed_agents:
            issues.append(
                f"plan에 포함된 {agent_name}가 completed_agents에 없습니다."
            )

    return issues


def validate_recorded_errors(state: AgentState) -> list[str]:
    """이전 Agent 실행 중 기록된 errors가 있는지 검증합니다."""
    errors = state.get("errors") or []

    if not errors:
        return []

    return [f"이전 Agent 실행 중 errors가 기록되어 있습니다: {errors}"]


def validate_recommendation_consistency(state: AgentState) -> list[str]:
    """
    추천 결과와 가입 가능 여부 결과의 기본 일관성을 검증합니다.

    현재는 팀원 Agent 결과 구조가 확정되지 않았으므로,
    recommend_result.result.recommended_products와
    eligibility_result.result.eligible_products가 모두 list일 때만 검사합니다.
    """
    issues = []

    recommend_result = state.get("recommend_result")
    eligibility_result = state.get("eligibility_result")

    if not recommend_result or not eligibility_result:
        return issues

    recommend_data = recommend_result.get("result", {})
    eligibility_data = eligibility_result.get("result", {})

    recommended_products = recommend_data.get("recommended_products")
    eligible_products = eligibility_data.get("eligible_products")

    if not isinstance(recommended_products, list):
        return issues

    if not isinstance(eligible_products, list):
        return issues

    recommended_names = extract_product_names(recommended_products)
    eligible_names = extract_product_names(eligible_products)

    for product_name in recommended_names:
        if product_name not in eligible_names:
            issues.append(
                f"추천 상품 '{product_name}'이 eligible_products에 없습니다."
            )

    return issues


def extract_product_names(products: list[Any]) -> list[str]:
    """상품 리스트에서 상품명을 추출합니다."""
    names = []

    for product in products:
        if isinstance(product, str):
            names.append(product)

        elif isinstance(product, dict):
            name = (
                product.get("product_name")
                or product.get("name")
                or product.get("상품명")
            )

            if name:
                names.append(str(name))

    return names


def run_validation_checks(state: AgentState) -> tuple[list[str], dict[str, bool]]:
    """Validation Agent에서 수행할 전체 검증을 실행합니다."""
    issues = []

    common_format_issues = validate_common_result_format(state)
    required_result_issues = validate_required_results_by_task(state)
    plan_completion_issues = validate_plan_completion(state)
    recorded_error_issues = validate_recorded_errors(state)
    recommendation_issues = validate_recommendation_consistency(state)

    issues.extend(common_format_issues)
    issues.extend(required_result_issues)
    issues.extend(plan_completion_issues)
    issues.extend(recorded_error_issues)
    issues.extend(recommendation_issues)

    checked_items = {
        "common_format_checked": True,
        "required_results_checked": True,
        "plan_completion_checked": True,
        "recorded_errors_checked": True,
        "recommendation_consistency_checked": True,

        # 팀원 Agent 결과 구조 확정 후 보강 예정
        "condition_conflict_checked": False,
        "rate_amount_payment_checked": False,
        "rag_evidence_checked": False,
        "inappropriate_recommendation_checked": False,
    }

    return issues, checked_items