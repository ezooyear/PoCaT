"""
Validation Agent 검증 도구
- Agent 결과의 공통 포맷, 필수 결과 존재 여부, 추천 결과 일관성을 검증합니다.
- 복구 가능한 fallback 상태는 실패로 오판하지 않도록 보정합니다.
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
        "financial_result",
        "product_result",
        "eligibility_result",
        "recommend_result",
    ],
    "early_termination": [
        "customer_result",
        "product_result",
        "financial_result",
        "recommend_result",
    ],
    "switch_analysis": [
        "customer_result",
        "financial_result",
        "product_result",
        "eligibility_result",
        "recommend_result",
    ],
}

RECOVERABLE_RESULT_STATUSES = {
    "fallback_success",
    "success",
    "recommended",
    "passed",
    "passed_with_warnings",
}


def _normalize_name(value: Any) -> str:
    return "".join(str(value or "").lower().split())


def _extract_eligibility_results(state: AgentState) -> list[dict[str, Any]]:
    results = state.get("eligibility_results")
    if isinstance(results, list):
        return results

    eligibility_result = state.get("eligibility_result") or {}
    if isinstance(eligibility_result, dict):
        payload = eligibility_result.get("result") or {}
        if isinstance(payload, dict) and isinstance(payload.get("results"), list):
            return payload.get("results") or []

    return []


def _extract_financial_results(state: AgentState) -> list[dict[str, Any]]:
    results = state.get("financial_results")
    if isinstance(results, list) and results:
        return results

    financial_result = state.get("financial_result") or {}
    if isinstance(financial_result, dict):
        payload = financial_result.get("result") or {}
        if isinstance(payload, dict):
            for key in ("financial_results", "calculations"):
                value = payload.get(key)
                if isinstance(value, list) and value:
                    return value

    return []


def _extract_recommendation_items(state: AgentState) -> list[dict[str, Any]]:
    recommend_result = state.get("recommend_result") or {}
    if not isinstance(recommend_result, dict):
        return []

    payload = recommend_result.get("result") or {}
    if not isinstance(payload, dict):
        payload = recommend_result

    for key in ("recommendations", "recommended_products", "recommendation_results"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]

    return []


def _is_recoverable_result_failure(result_value: Any) -> bool:
    if not isinstance(result_value, dict):
        return False

    payload = result_value.get("result") or {}
    if not isinstance(payload, dict):
        return False

    payload_status = str(payload.get("status") or "").strip().lower()
    return payload_status in RECOVERABLE_RESULT_STATUSES


def has_actionable_recommendation_state(state: AgentState) -> bool:
    if str(state.get("task_type") or "") != "recommendation":
        return False

    eligibility_results = _extract_eligibility_results(state)
    financial_results = _extract_financial_results(state)
    recommendation_items = _extract_recommendation_items(state)

    eligible_names = {
        _normalize_name(item.get("product_name"))
        for item in eligibility_results
        if item.get("eligible") is True
        and str(item.get("status") or "").strip().lower() == "eligible"
        and item.get("product_name")
    }
    calculated_names = {
        _normalize_name(item.get("product_name"))
        for item in financial_results
        if str(item.get("status") or "").strip().lower() == "calculated"
        and item.get("product_name")
    }
    recommended_names = {
        _normalize_name(item.get("product_name"))
        for item in recommendation_items
        if item.get("product_name")
    }

    if not eligible_names or not calculated_names:
        return False

    if recommended_names:
        return any(
            name in eligible_names and name in calculated_names
            for name in recommended_names
        )

    return bool(eligible_names.intersection(calculated_names))


def _has_customer_context(state: AgentState) -> bool:
    customer_result = state.get("customer_result") or {}
    if isinstance(customer_result, dict):
        payload = customer_result.get("result") or {}
        if isinstance(payload, dict) and isinstance(payload.get("customer_profile"), dict):
            return True
        if isinstance(customer_result.get("customer_profile"), dict):
            return True

    return isinstance(state.get("customer_profile"), dict) and bool(state.get("customer_profile"))


def validate_common_result_format(state: AgentState) -> list[str]:
    issues = []
    actionable_recommendation = has_actionable_recommendation_state(state)

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

        missing_keys = [key for key in REQUIRED_RESULT_KEYS if key not in result_value]
        if missing_keys:
            issues.append(f"{result_name}에 필수 키가 없습니다: {missing_keys}")

        if result_value.get("status") != "failed":
            continue

        if result_name == "financial_result" and _is_recoverable_result_failure(result_value):
            continue

        if result_name == "customer_result" and (_has_customer_context(state) or actionable_recommendation):
            continue

        issues.append(f"{result_name}의 status가 failed입니다.")

    return issues


def validate_required_results_by_task(state: AgentState) -> list[str]:
    issues = []
    task_type = state.get("task_type")

    for result_name in REQUIRED_RESULTS_BY_TASK.get(task_type, []):
        if not state.get(result_name):
            issues.append(f"task_type='{task_type}'에 필요한 {result_name}가 없습니다.")

    return issues


def validate_plan_completion(state: AgentState) -> list[str]:
    issues = []
    plan = state.get("plan") or []
    completed_agents = state.get("completed_agents") or []

    for agent_name in plan:
        if agent_name in ["validation_agent", "FINISH", "END"]:
            continue
        if agent_name not in completed_agents:
            issues.append(f"plan에 포함된 {agent_name}가 completed_agents에 없습니다.")

    return issues


def validate_recorded_errors(state: AgentState) -> list[str]:
    errors = state.get("errors") or []
    if not errors:
        return []

    blocking_errors = []
    for item in errors:
        if not isinstance(item, dict):
            blocking_errors.append(item)
            continue
        if item.get("recoverable") is True or item.get("user_visible") is False:
            continue
        if item.get("fallback_applied") is True:
            continue
        blocking_errors.append(item)

    if not blocking_errors:
        return []

    return ["이전 Agent 실행 중 복구되지 않은 오류가 남아 있습니다."]


def extract_product_names(products: list[Any]) -> list[str]:
    names = []
    for product in products:
        if isinstance(product, str):
            names.append(product)
        elif isinstance(product, dict):
            name = product.get("product_name") or product.get("name") or product.get("product")
            if name:
                names.append(str(name))
    return names


def validate_recommendation_consistency(state: AgentState) -> list[str]:
    issues = []

    recommendation_items = _extract_recommendation_items(state)
    if not recommendation_items:
        return issues

    eligible_names = {
        _normalize_name(item.get("product_name"))
        for item in _extract_eligibility_results(state)
        if item.get("eligible") is True
        and str(item.get("status") or "").strip().lower() == "eligible"
        and item.get("product_name")
    }
    calculated_names = {
        _normalize_name(item.get("product_name"))
        for item in _extract_financial_results(state)
        if str(item.get("status") or "").strip().lower() == "calculated"
        and item.get("product_name")
    }

    for item in recommendation_items:
        product_name = str(item.get("product_name") or "").strip()
        normalized = _normalize_name(product_name)
        if not normalized:
            continue
        if normalized not in eligible_names:
            issues.append(f"추천 상품 '{product_name}'이 eligibility 결과상 가입 가능 상품에 없습니다.")
        elif normalized not in calculated_names:
            issues.append(f"추천 상품 '{product_name}'에 계산된 financial 결과가 없습니다.")

    return issues


def run_validation_checks(state: AgentState) -> tuple[list[str], dict[str, bool]]:
    issues = []
    issues.extend(validate_common_result_format(state))
    issues.extend(validate_required_results_by_task(state))
    issues.extend(validate_plan_completion(state))
    issues.extend(validate_recorded_errors(state))
    issues.extend(validate_recommendation_consistency(state))

    if has_actionable_recommendation_state(state):
        issues = [
            issue for issue in issues
            if "customer_result의 status가 failed입니다." not in issue
            and "financial_result의 status가 failed입니다." not in issue
            and "복구되지 않은 오류" not in issue
        ]

    checked_items = {
        "common_format_checked": True,
        "required_results_checked": True,
        "plan_completion_checked": True,
        "recorded_errors_checked": True,
        "recommendation_consistency_checked": True,
        "condition_conflict_checked": False,
        "rate_amount_payment_checked": False,
        "rag_evidence_checked": False,
        "inappropriate_recommendation_checked": False,
    }

    return issues, checked_items


def _sanitize_errors_for_validation(errors: list[Any]) -> list[dict[str, Any]]:
    sanitized = []
    for item in errors or []:
        if not isinstance(item, dict):
            continue
        if item.get("recoverable") is True or item.get("user_visible") is False:
            continue
        copied = dict(item)
        copied.pop("error", None)
        sanitized.append(copied)
    return sanitized


def _normalize_result_for_validation(result_value: Any, *, state: AgentState, result_name: str) -> Any:
    if not isinstance(result_value, dict):
        return result_value

    normalized = dict(result_value)
    payload = normalized.get("result")
    if isinstance(payload, dict):
        payload = dict(payload)
    else:
        payload = {}

    if result_name == "financial_result" and _is_recoverable_result_failure(normalized):
        normalized["status"] = "success"
        payload["status"] = payload.get("status") or "fallback_success"

    if result_name == "customer_result" and normalized.get("status") == "failed":
        if _has_customer_context(state) or has_actionable_recommendation_state(state):
            normalized["status"] = "success"
            if payload:
                payload["status"] = payload.get("status") or "success"

    normalized["error"] = None
    if payload:
        normalized["result"] = payload
    return normalized


def build_validation_context(state: AgentState) -> dict[str, Any]:
    agent_outputs = state.get("agent_outputs") or {}

    customer_result = state.get("customer_result") or agent_outputs.get("customer_agent")
    product_result = state.get("product_result") or agent_outputs.get("product_agent")
    financial_result = state.get("financial_result") or agent_outputs.get("financial_agent")
    eligibility_result = state.get("eligibility_result") or agent_outputs.get("eligibility_agent")
    recommend_result = state.get("recommend_result") or agent_outputs.get("recommend_agent")

    return {
        "user_query": state.get("user_query"),
        "task_type": state.get("task_type"),
        "plan": state.get("plan"),
        "completed_agents": state.get("completed_agents"),
        "current_step": state.get("current_step"),
        "current_agent": state.get("current_agent"),
        "errors": _sanitize_errors_for_validation(state.get("errors") or []),
        "customer_result": _normalize_result_for_validation(customer_result, state=state, result_name="customer_result"),
        "product_result": _normalize_result_for_validation(product_result, state=state, result_name="product_result"),
        "financial_result": _normalize_result_for_validation(financial_result, state=state, result_name="financial_result"),
        "eligibility_result": _normalize_result_for_validation(eligibility_result, state=state, result_name="eligibility_result"),
        "recommend_result": _normalize_result_for_validation(recommend_result, state=state, result_name="recommend_result"),
        "agent_outputs_summary": {
            agent_name: {
                "status": (result.get("status") if isinstance(result, dict) else None),
            }
            for agent_name, result in agent_outputs.items()
        },
        "actionable_recommendation_state": has_actionable_recommendation_state(state),
    }


def build_rule_based_verify_result(
    rule_issues: list[str],
    rule_checked_items: dict[str, bool],
    llm_skipped: bool = False,
) -> dict[str, Any]:
    is_valid = len(rule_issues) == 0

    if is_valid:
        summary = "검증을 통과했습니다."
    elif llm_skipped:
        summary = "rule 기반 검증에서 이슈가 발견되었습니다."
    else:
        summary = "LLM 검증에 실패하여 rule 기반 검증 결과만 사용했습니다."

    return {
        "status": "passed" if is_valid else "failed",
        "is_valid": is_valid,
        "summary": summary,
        "issues": [
            {
                "level": "error",
                "type": "rule_based_issue",
                "message": issue,
                "related_agent": None,
                "suggestion": "해당 Agent 결과 구조와 state 저장 방식을 확인하세요.",
            }
            for issue in rule_issues
        ],
        "checked_items": {
            **rule_checked_items,
            "condition_conflict_checked": False,
            "rate_amount_payment_checked": False,
            "rag_evidence_checked": False,
            "inappropriate_recommendation_checked": False,
        },
        "final_notes": [] if is_valid else ["검증 이슈가 있으므로 최종 답변 전 관련 Agent 결과를 확인해야 합니다."],
        "revision_required": not is_valid,
    }
