"""
Validation Agent
- State에 저장된 Agent별 구조화 결과를 검증합니다.
- rule 기반 1차 검증은 항상 수행합니다.
- 복잡한 task_type에서만 LLM 기반 verify_result 검증을 수행합니다.

항상 실행:
- run_validation_checks(state)
- 결과 누락 / plan 완료 / errors / 기본 추천 충돌 검사

복잡한 질문에서만 실행:
- LLM verify_result
- 조건 충돌 / 금리·금액·납입 횟수 / RAG 근거 / 부적절 추천 검증

"""

import json
from typing import Any
import re
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import get_llm
from graph import state
from graph.state import AgentState
from agents.base import build_agent_trace_input, build_agent_trace_output, make_agent_result
from agents.validation.prompts import VALIDATION_SYSTEM_PROMPT
from agents.validation.tools import (
    build_validation_context,
    build_rule_based_verify_result,
    run_validation_checks,
)
from observability.langfuse import flush_langfuse, langfuse_observation, update_observation


# LLM 검증이 필요한 복잡한 작업만 지정합니다.
# 단순 고객 조회나 단순 상품 정보 조회는 rule 검증만으로 충분합니다.
LLM_VALIDATION_TASKS = {
    "financial_analysis",
    "eligibility_check",
    "recommendation",
    "early_termination",
    "switch_analysis",
}

# retry loop: validation 실패 시 어느 agent부터 다시 실행할지 판단하는 설정
DEFAULT_MAX_VALIDATION_RETRIES = 1

RETRYABLE_START_AGENTS = {
    "product_agent",
    "eligibility_agent",
    "financial_agent",
    "recommend_agent",
}

STALE_FIELDS_BY_RETRY_START = {
    "product_agent": [
        "product_result",
        "products",
        "product_candidates",
        "searched_products",
        "eligibility_result",
        "eligibility_results",
        "financial_result",
        "financial_results",
        "recommend_result",
        "recommendation_results",
        "draft_answer",
        "final_answer",
    ],
    "eligibility_agent": [
        "eligibility_result",
        "eligibility_results",
        "financial_result",
        "financial_results",
        "recommend_result",
        "recommendation_results",
        "draft_answer",
        "final_answer",
    ],
    "financial_agent": [
        "financial_result",
        "financial_results",
        "recommend_result",
        "recommendation_results",
        "draft_answer",
        "final_answer",
    ],
    "recommend_agent": [
        "recommend_result",
        "recommendation_results",
        "draft_answer",
        "final_answer",
    ],
}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _get_validation_retry_count_for_query(state: AgentState) -> int:
    """현재 user_query 기준 retry 횟수를 가져옵니다.

    이전 질문의 retry_count가 다음 질문에 영향을 주지 않도록
    validation_retry_query와 현재 user_query가 다르면 0부터 시작합니다.
    """
    current_query = str(state.get("user_query") or "")
    previous_query = str(state.get("validation_retry_query") or "")

    if current_query and current_query != previous_query:
        return 0

    return max(0, _safe_int(state.get("validation_retry_count"), 0))


def _collect_validation_issue_text(
    verify_result: dict[str, Any],
    validation_summary: dict[str, Any],
) -> str:
    """LLM/rule 검증 이슈를 문자열로 모아 retry agent 판단에 사용합니다."""
    parts: list[str] = []

    for issue in verify_result.get("issues") or []:
        if isinstance(issue, dict):
            for key in ["type", "message", "suggestion", "related_agent"]:
                value = issue.get(key)
                if value:
                    parts.append(str(value))
        else:
            parts.append(str(issue))

    for issue in validation_summary.get("blocking_issues") or []:
        parts.append(str(issue))

    summary = verify_result.get("summary")
    if summary:
        parts.append(str(summary))

    return " ".join(parts).lower()


def _collect_validation_issue_types(verify_result: dict[str, Any]) -> set[str]:
    issue_types: set[str] = set()

    for issue in verify_result.get("issues") or []:
        if isinstance(issue, dict) and issue.get("type"):
            issue_types.add(str(issue.get("type")))

    return issue_types


def _collect_related_agents(verify_result: dict[str, Any]) -> set[str]:
    related_agents: set[str] = set()

    for issue in verify_result.get("issues") or []:
        if isinstance(issue, dict) and issue.get("related_agent"):
            related_agents.add(str(issue.get("related_agent")))

    return related_agents


def _infer_retry_start_agent(
    verify_result: dict[str, Any],
    validation_summary: dict[str, Any],
    state: AgentState,
) -> str | None:
    """검증 실패 원인에 따라 어느 agent부터 다시 실행할지 결정합니다.

    upstream 문제일수록 앞단 agent부터 다시 실행합니다.
    product → eligibility → financial → recommend 순서로 판단합니다.
    """
    issue_text = _collect_validation_issue_text(verify_result, validation_summary)
    issue_types = _collect_validation_issue_types(verify_result)
    related_agents = _collect_related_agents(verify_result)

    # 추천 질문인데 상품 후보 자체가 비어 있으면 product부터 다시 검색
    if state.get("task_type") == "recommendation" and not state.get("product_candidates"):
        return "product_agent"

    # 상품 후보/RAG 근거/상품 조건 문제
    product_keywords = [
        "product",
        "product_candidates",
        "rag",
        "근거",
        "상품 후보",
        "상품후보",
        "상품 조건",
        "상품조건",
        "상품 정보",
        "상품정보",
        "상품명",
    ]
    if (
        "product_agent" in related_agents
        or "rag_evidence_missing" in issue_types
        or any(keyword in issue_text for keyword in product_keywords)
    ):
        return "product_agent"

    # 가입 가능 여부/대상자/군인 상품 오추천 문제
    eligibility_keywords = [
        "eligibility",
        "가입 가능",
        "가입가능",
        "가입 대상",
        "가입대상",
        "가입 조건",
        "가입조건",
        "부적절",
        "군인",
        "직업군인",
        "장병",
        "간부",
        "대상자가",
        "inappropriate",
    ]
    if (
        "eligibility_agent" in related_agents
        or "inappropriate_recommendation" in issue_types
        or any(keyword in issue_text for keyword in eligibility_keywords)
    ):
        return "eligibility_agent"

    # 금리/이자/만기/납입/계산 문제
    financial_keywords = [
        "financial",
        "calculation",
        "계산",
        "금리",
        "이자",
        "만기",
        "납입",
        "금액",
        "세전",
        "세후",
        "기간",
        "개월",
    ]
    financial_issue_types = {
        "calculation_mismatch",
        "amount_mismatch",
        "rate_mismatch",
        "unverifiable",
    }
    if (
        "financial_agent" in related_agents
        or issue_types.intersection(financial_issue_types)
        or any(keyword in issue_text for keyword in financial_keywords)
    ):
        return "financial_agent"

    # 추천 결과 누락/추천 요약 문제
    recommend_keywords = [
        "recommend",
        "recommendation",
        "추천",
        "추천 결과",
        "추천결과",
        "추천 없음",
        "추천없음",
    ]
    if (
        "recommend_agent" in related_agents
        or "recommendation_consistency" in issue_types
        or any(keyword in issue_text for keyword in recommend_keywords)
    ):
        return "recommend_agent"

    # 구체 원인 분류가 애매하지만 agent 출력 문제면 recommend부터 최소 재생성
    if validation_summary.get("failure_type") == "agent_output_error":
        return "recommend_agent"

    return None


def _build_retry_reason(
    verify_result: dict[str, Any],
    validation_summary: dict[str, Any],
) -> str:
    blocking_issues = validation_summary.get("blocking_issues") or []

    if blocking_issues:
        return str(blocking_issues[0])[:300]

    summary = str(verify_result.get("summary") or "").strip()
    if summary:
        return summary[:300]

    return "Validation 실패로 추천 결과 재생성이 필요합니다."


def _clear_stale_outputs_for_retry(retry_start_agent: str | None) -> dict[str, Any]:
    """재실행 시작 지점 이후의 이전 결과를 비워 stale result 사용을 방지합니다."""
    if not retry_start_agent:
        return {}

    stale_fields = STALE_FIELDS_BY_RETRY_START.get(retry_start_agent, [])
    return {field: None for field in stale_fields}


def _build_validation_retry_update(
    state: AgentState,
    verify_result: dict[str, Any],
    validation_summary: dict[str, Any],
) -> dict[str, Any]:
    """Validation 결과를 보고 retry loop 관련 state update를 생성합니다.

    원칙:
    - validation 통과: retry_start_agent 제거
    - 사용자 정보 부족: 재실행하지 않고 supervisor가 추가 질문
    - agent 출력 문제: 최대 1회 필요한 agent부터 재실행
    """
    current_query = str(state.get("user_query") or "")
    max_retries = max(
        0,
        _safe_int(
            state.get("max_validation_retries"),
            DEFAULT_MAX_VALIDATION_RETRIES,
        ),
    )
    retry_count = _get_validation_retry_count_for_query(state)
    retry_history = list(state.get("retry_history") or [])

    base_update: dict[str, Any] = {
        "max_validation_retries": max_retries,
        "validation_retry_query": current_query,
    }

    is_valid = bool(verify_result.get("is_valid"))

    if is_valid:
        return {
            **base_update,
            "validation_retry_count": 0,
            "retry_start_agent": None,
            "retry_reason": None,
            "last_validation_failed": False,
            "retry_history": retry_history,
        }

    if validation_summary.get("awaiting_user_input"):
        return {
            **base_update,
            "validation_retry_count": retry_count,
            "retry_start_agent": None,
            "retry_reason": "사용자 추가 정보가 필요하여 자동 재생성을 수행하지 않습니다.",
            "last_validation_failed": True,
            "retry_history": retry_history,
        }

    if retry_count >= max_retries:
        return {
            **base_update,
            "validation_retry_count": retry_count,
            "retry_start_agent": None,
            "retry_reason": "Validation retry 최대 횟수에 도달하여 supervisor로 반환합니다.",
            "last_validation_failed": True,
            "retry_history": retry_history,
        }

    retry_start_agent = _infer_retry_start_agent(
        verify_result=verify_result,
        validation_summary=validation_summary,
        state=state,
    )

    if retry_start_agent not in RETRYABLE_START_AGENTS:
        return {
            **base_update,
            "validation_retry_count": retry_count,
            "retry_start_agent": None,
            "retry_reason": "재실행할 agent를 특정하지 못해 supervisor로 반환합니다.",
            "last_validation_failed": True,
            "retry_history": retry_history,
        }

    next_retry_count = retry_count + 1
    retry_reason = _build_retry_reason(verify_result, validation_summary)

    retry_record = {
        "retry_count": next_retry_count,
        "retry_start_agent": retry_start_agent,
        "retry_reason": retry_reason,
        "failure_type": validation_summary.get("failure_type"),
        "blocking_issues": validation_summary.get("blocking_issues") or [],
    }

    return {
        **base_update,
        **_clear_stale_outputs_for_retry(retry_start_agent),
        "validation_retry_count": next_retry_count,
        "retry_start_agent": retry_start_agent,
        "retry_reason": retry_reason,
        "last_validation_failed": True,
        "retry_history": retry_history + [retry_record],
    }

# helper함수
def _add_validation_field_semantics(validation_context: Any) -> Any:
    """
    Validation LLM이 고객 거래기간(transaction_months)을
    사용자의 희망 계약기간(term_months)으로 오해하지 않도록 명시합니다.
    """
    notice = (
        "\n\n[필드 의미 주의]\n"
        "- transaction_months는 고객의 은행 거래 개월 수입니다.\n"
        "- transaction_months는 사용자가 원하는 예금/적금 계약기간이 아닙니다.\n"
        "- 사용자의 희망 계약기간은 질문에 '2년', '24개월'처럼 직접 표현된 경우에만 term_months로 봅니다.\n"
        "- 사용자 질문에 기간이 없으면 희망 계약기간은 None/미정으로 판단합니다.\n"
        "- 따라서 transaction_months 값을 근거로 기간 초과, 기간 충돌, 계약기간 불일치 이슈를 만들지 마십시오.\n"
    )

    if isinstance(validation_context, str):
        return validation_context + notice

    if isinstance(validation_context, dict):
        context = dict(validation_context)
        context["field_semantics_notice"] = notice
        return context

    return validation_context

async def _validation_agent_node_impl(state: AgentState) -> dict[str, Any]:
    """
    Validation Agent 실제 실행 로직

    검증 방향:
    1. state에 저장된 구조화 결과 수집
    2. rule 기반 기본 검증 수행
    3. 복잡한 task_type이면 LLM 기반 verify_result 생성
    4. validation_result와 agent_outputs에 저장
    """

    validation_context = build_validation_context(state)
    validation_context = _add_validation_field_semantics(validation_context)

    # 1차 검증: 항상 실행
    rule_issues, rule_checked_items = run_validation_checks(state)

    # LLM 검증이 필요한 task_type인지 한 번만 판단
    should_run_llm = _should_run_llm_validation(state)

    # 2차 검증: 복잡한 task_type에서만 LLM 실행
    if should_run_llm:
        verify_result = await _run_llm_verify_result_async(
            validation_context=validation_context,
            rule_issues=rule_issues,
            rule_checked_items=rule_checked_items,
        )
    else:
        verify_result = {}

    # LLM 검증을 실행하지 않았거나, LLM 검증이 실패한 경우 rule 기반 결과만 사용
    if not verify_result:
        verify_result = build_rule_based_verify_result(
            rule_issues=rule_issues,
            rule_checked_items=rule_checked_items,
            llm_skipped=not should_run_llm,
        )
    verify_result = _soften_recommendation_verify_result(verify_result, state)
    is_valid = bool(verify_result.get("is_valid"))
    validation_summary = _extract_validation_summary(verify_result, state)
    warnings = [
        issue.get("message") if isinstance(issue, dict) else str(issue)
        for issue in (verify_result.get("issues") or [])
        if isinstance(issue, dict) and issue.get("level") == "warning"
    ]
    failure_reasons = [
        issue.get("message") if isinstance(issue, dict) else str(issue)
        for issue in (verify_result.get("issues") or [])
        if isinstance(issue, dict) and issue.get("level") == "error"
    ]
    validation_status = (
        "passed_with_warnings" if is_valid and warnings else
        "passed" if is_valid else
        "failed"
    )

    retry_loop_update = _build_validation_retry_update(
        state=state,
        verify_result=verify_result,
        validation_summary=validation_summary,
    )

    validation_result = make_agent_result(
        status=validation_status,
        result={
            "verify_result": verify_result,
            "is_valid": is_valid,
            "issues": verify_result.get("issues", []),
            "failure_reasons": failure_reasons,
            "warnings": warnings,
            "checks": verify_result.get("checked_items", rule_checked_items),
            "revision_required": validation_summary["revision_required"],
            "failure_type": validation_summary["failure_type"],
            "missing_fields": validation_summary["missing_fields"],
            "blocking_issues": validation_summary["blocking_issues"],
            "awaiting_user_input": validation_summary["awaiting_user_input"],
            "retry_start_agent": retry_loop_update.get("retry_start_agent"),
            "retry_reason": retry_loop_update.get("retry_reason"),
            "validation_retry_count": retry_loop_update.get("validation_retry_count"),
            "max_validation_retries": retry_loop_update.get("max_validation_retries"),
            "summary": verify_result.get("summary"),
            "final_notes": verify_result.get("final_notes", []),
        },
        evidence=[],
        error=None if is_valid else "검증 이슈가 발견되었습니다.",
    )

    agent_outputs = dict(state.get("agent_outputs") or {})
    agent_outputs["validation_agent"] = validation_result

    completed_agents = list(state.get("completed_agents") or [])
    if "validation_agent" not in completed_agents:
        completed_agents.append("validation_agent")

    result_update = {
        "validation_result": validation_result,
        # 추가: Supervisor가 깊게 파싱하지 않고 바로 볼 수 있는 필드
        "validation_passed": is_valid,
        "revision_required": validation_summary["revision_required"],
        "failure_type": validation_summary["failure_type"],
        "missing_fields": validation_summary["missing_fields"],
        "blocking_issues": validation_summary["blocking_issues"],
        "awaiting_user_input": validation_summary["awaiting_user_input"],
        "agent_outputs": agent_outputs,
        "current_agent": "validation_agent",
        "completed_agents": completed_agents,
        "current_step": (state.get("current_step") or 0) + 1,
    }

    # retry loop: validation 실패 시 builder가 retry_start_agent를 보고 재실행
    result_update.update(retry_loop_update)

    return result_update
def _has_recommendation_result(state: AgentState) -> bool:
    """
    추천 결과가 하나라도 있으면 True.
    top-level recommendation_results와 recommend_result.result.recommendations를 모두 확인합니다.
    """
    top_level = state.get("recommendation_results") or []
    if isinstance(top_level, list) and len(top_level) > 0:
        return True

    recommend_result = state.get("recommend_result") or {}
    if not isinstance(recommend_result, dict):
        return False

    payload = recommend_result.get("result") or {}
    if not isinstance(payload, dict):
        return False

    for key in ["recommendations", "recommended_products", "recommendation_results"]:
        value = payload.get(key)
        if isinstance(value, list) and len(value) > 0:
            return True

    return False


def _get_customer_profile_for_validation(state: AgentState) -> dict[str, Any]:
    profile = state.get("customer_profile")
    if isinstance(profile, dict):
        return profile

    customer_result = state.get("customer_result") or {}
    if isinstance(customer_result, dict):
        payload = customer_result.get("result") or {}
        if isinstance(payload, dict) and isinstance(payload.get("customer_profile"), dict):
            return payload["customer_profile"]

        if isinstance(customer_result.get("customer_profile"), dict):
            return customer_result["customer_profile"]

    return {}


def _is_positive_number(value: Any) -> bool:
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def _soften_recommendation_verify_result(
    verify_result: dict[str, Any],
    state: AgentState,
) -> dict[str, Any]:
    """
    일반 추천 질문에서 추천 결과가 이미 생성된 경우,
    warning만으로 최종 추천을 차단하지 않도록 완화합니다.

    단, error 레벨 이슈가 있으면 그대로 실패 처리합니다.
    """
    if state.get("task_type") != "recommendation":
        return verify_result

    if not _has_recommendation_result(state):
        return verify_result

    issues = verify_result.get("issues") or []
    filtered_issues = []
    for issue in issues:
        if not isinstance(issue, dict):
            filtered_issues.append(issue)
            continue

        message = str(issue.get("message") or "")
        suggestion = str(issue.get("suggestion") or "")

        # transaction_months=81을 희망 계약기간으로 오해한 validation 이슈 제거
        if (
            "81개월" in message
            and ("계약 기간" in message or "계약기간" in message or "허용 기간" in message)
        ):
            continue

        if (
            "transaction_months" in message
            and ("계약 기간" in message or "계약기간" in message)
        ):
            continue

        filtered_issues.append(issue)

    verify_result = dict(verify_result)
    verify_result["issues"] = filtered_issues
    issues = filtered_issues
    has_error_issue = any(
        isinstance(issue, dict)
        and str(issue.get("level") or "").lower() == "error"
        for issue in issues
    )

    if has_error_issue:
        return verify_result

    softened = dict(verify_result)
    softened["status"] = "warning" if issues else "passed"
    softened["is_valid"] = True
    softened["revision_required"] = False

    final_notes = list(softened.get("final_notes") or [])
    final_notes.append(
        "일반 추천 질문이므로 계약기간·적용금리 미확정 항목은 최종 차단이 아니라 주의사항으로 안내합니다."
    )
    softened["final_notes"] = final_notes

    return softened

def _extract_validation_summary(
    verify_result: dict[str, Any],
    state: AgentState,
) -> dict[str, Any]:
    """
    Validation 결과를 Supervisor final이 바로 읽을 수 있는 형태로 요약합니다.
    - missing_fields: 사용자가 추가로 알려줘야 하는 정보
    - blocking_issues: 확정 추천을 막아야 하는 검증 이슈
    - failure_type: missing_user_input / agent_output_error / passed
    """
    is_valid = bool(verify_result.get("is_valid"))
    revision_required = bool(verify_result.get("revision_required", not is_valid))

    issues = verify_result.get("issues") or []
    missing_fields = _infer_missing_user_fields(state)
    blocking_issues: list[str] = []
    has_recommendation = _has_recommendation_result(state)

    for issue in issues:
        if isinstance(issue, dict):
            level = str(issue.get("level") or "").lower()
            message = str(issue.get("message") or "").strip()
            suggestion = str(issue.get("suggestion") or "").strip()

            # error는 항상 차단
            if level == "error":
                if message:
                    blocking_issues.append(message)
                elif suggestion:
                    blocking_issues.append(suggestion)

        # 추천 결과가 없는 상태의 warning만 차단
        elif level == "warning" and not has_recommendation:
            if message:
                blocking_issues.append(message)
            elif suggestion:
                blocking_issues.append(suggestion)

        else:
            message = str(issue).strip()
            if message:
                blocking_issues.append(message)

    # 검증 실패인데 구체 이슈가 비어 있으면 summary라도 넣어둡니다.
    if not is_valid and not blocking_issues and not missing_fields:
        summary = str(verify_result.get("summary") or "").strip()
        blocking_issues.append(summary or "검증 이슈가 발견되었습니다.")

    if is_valid:
        failure_type = "passed"
    elif missing_fields:
        failure_type = "missing_user_input"
    else:
        failure_type = "agent_output_error"

    return {
        "revision_required": revision_required,
        "failure_type": failure_type,
        "missing_fields": missing_fields,
        "blocking_issues": blocking_issues,
        "awaiting_user_input": bool(missing_fields),
    }


def _infer_missing_user_fields(state: AgentState) -> list[str]:
    """
    사용자 질문에서 추천에 필요한 최소 입력값이 빠졌는지 확인합니다.

    일반 추천:
    - 고객 ID와 고객 profile의 월 저축 가능액이 있으면 추천 가능
    - 희망 가입 기간이 없어도 추천 자체는 막지 않음

    정확한 이자 계산 요청:
    - 사용자가 예상 이자/만기 금액/계산을 요구하면 기간이 필요함
    """
    task_type = state.get("task_type")
    if task_type != "recommendation":
        return []

    user_query = str(state.get("user_query") or "").replace(" ", "")
    customer_profile = _get_customer_profile_for_validation(state)

    missing_fields: list[str] = []

    has_customer_id = bool(state.get("customer_id")) or bool(
        re.search(r"(고객|customer|id|ID|고객ID|고객id|고객_)?\d+번?", user_query)
    )

    has_query_monthly_amount = bool(
        re.search(r"월?\d+만?원|\d{1,3}(,\d{3})*원", user_query)
    )

    profile_monthly_amount = (
        customer_profile.get("monthly_saving_amount")
        or customer_profile.get("available_monthly_saving")
        or customer_profile.get("monthly_amount")
    )

    has_monthly_basis = has_query_monthly_amount or _is_positive_number(profile_monthly_amount)

    has_period = bool(
        re.search(r"\d+개월|\d+년|[0-9]+개월|[0-9]+년", user_query)
    )

    has_product_type = "적금" in user_query or "예금" in user_query
    has_product_basis = (
        has_product_type
        or bool(state.get("product_candidates") or [])
        or _has_recommendation_result(state)
    )

    wants_exact_calculation = any(
        keyword in user_query
        for keyword in ["예상이자", "만기금액", "얼마받", "계산", "정확"]
    )

    if not has_customer_id:
        missing_fields.append("고객 ID")

    if not has_monthly_basis:
        missing_fields.append("월 납입 예정 금액 또는 월 저축 가능액")

    # 일반 추천에서는 기간이 없어도 추천 가능.
    # 다만 예상 이자/만기금액 계산을 직접 요구한 경우에는 기간 필요.
    if wants_exact_calculation and not has_period:
        missing_fields.append("희망 가입 기간")

    if not has_product_basis:
        missing_fields.append("상품 유형(예금/적금)")

    return missing_fields

def _should_run_llm_validation(state: AgentState) -> bool:
    """
    LLM 검증이 필요한 task_type인지 판단합니다.
     기준:
    1. 복잡한 task_type인가?
    2. 여러 Agent 결과를 종합해야 하는가?
    3. 추천/가입가능/계산/갈아타기/중도해지처럼 충돌 가능성이 있는가?
    """

    task_type = state.get("task_type")
    return task_type in LLM_VALIDATION_TASKS


async def _run_llm_verify_result_async(
    validation_context: dict[str, Any],
    rule_issues: list[str],
    rule_checked_items: dict[str, bool],
) -> dict[str, Any]:
    """
    LLM 기반 verify_result를 생성합니다.
    """

    llm = get_llm()

    payload = {
        "validation_context": validation_context,
        "rule_based_validation": {
            "issues": rule_issues,
            "checked_items": rule_checked_items,
        },
    }

    messages = [
        SystemMessage(content=VALIDATION_SYSTEM_PROMPT),
        HumanMessage(
            content=json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        ),
    ]

    try:
        response = await llm.ainvoke(messages)
        return _parse_json_response(response.content)

    except Exception:
        return {}


def _parse_json_response(text: str) -> dict[str, Any]:
    """
    LLM 응답에서 JSON만 추출해 dict로 변환합니다.
    """

    if not text:
        return {}

    cleaned = text.strip()

    if "```" in cleaned:
        parts = cleaned.split("```")
        if len(parts) >= 2:
            cleaned = parts[1].strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()

    try:
        data = json.loads(cleaned)

    except json.JSONDecodeError:
        return {
            "status": "warning",
            "is_valid": False,
            "summary": "Validation LLM 응답이 JSON 형식이 아니어서 검증 결과를 제한적으로 처리했습니다.",
            "issues": [
                {
                    "level": "warning",
                    "type": "invalid_llm_json",
                    "message": cleaned,
                    "related_agent": "validation_agent",
                    "suggestion": "VALIDATION_SYSTEM_PROMPT의 JSON 출력 지시를 확인하세요.",
                }
            ],
            "checked_items": {
                "common_format_checked": True,
                "required_results_checked": True,
                "plan_completion_checked": True,
                "recorded_errors_checked": True,
                "recommendation_consistency_checked": True,
                "condition_conflict_checked": False,
                "rate_amount_payment_checked": False,
                "rag_evidence_checked": False,
                "inappropriate_recommendation_checked": False,
            },
            "final_notes": [
                "검증 결과 형식이 불안정하므로 최종 답변에서 실제 금리, 가입 조건, 약관 확인 안내가 필요합니다."
            ],
            "revision_required": True,
        }

    if isinstance(data, str):
        nested = data.strip()
        if not nested:
            return {}
        try:
            data = json.loads(nested)
        except json.JSONDecodeError:
            return {}

    if not isinstance(data, dict):
        return {}

    return _normalize_verify_result(data)


def _normalize_verify_result(data: dict[str, Any]) -> dict[str, Any]:
    """
    LLM verify_result의 필수 키를 보정합니다.
    """

    status = data.get("status", "warning")

    if status not in ["passed", "warning", "failed"]:
        status = "warning"

    issues = data.get("issues") or []
    if not isinstance(issues, list):
        issues = []

    is_valid = data.get("is_valid")
    if is_valid is None:
        is_valid = status == "passed" and len(issues) == 0

    revision_required = data.get("revision_required")
    if revision_required is None:
        revision_required = status in ["warning", "failed"] or len(issues) > 0

    checked_items = data.get("checked_items") or {}

    default_checked_items = {
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

    default_checked_items.update(checked_items)

    issue_types = {
        issue.get("type")
        for issue in issues
        if isinstance(issue, dict)
    }

    has_error_issue = any(
        isinstance(issue, dict) and issue.get("level") == "error"
        for issue in issues
    )

    has_warning_issue = any(
        isinstance(issue, dict) and issue.get("level") == "warning"
        for issue in issues
    )

    if has_error_issue:
        status = "failed"
        is_valid = False
        revision_required = True
    elif has_warning_issue:
        status = "warning"
        is_valid = True
        revision_required = False

    if "condition_conflict" in issue_types:
        default_checked_items["condition_conflict_checked"] = True

    if "calculation_mismatch" in issue_types or "unverifiable" in issue_types:
        default_checked_items["rate_amount_payment_checked"] = True

    if "rag_evidence_missing" in issue_types:
        default_checked_items["rag_evidence_checked"] = True

    if "inappropriate_recommendation" in issue_types:
        default_checked_items["inappropriate_recommendation_checked"] = True

    return {
        "status": status,
        "is_valid": bool(is_valid),
        "summary": data.get("summary", "검증이 완료되었습니다."),
        "issues": issues,
        "checked_items": default_checked_items,
        "final_notes": data.get("final_notes") or [],
        "revision_required": bool(revision_required),
    }




async def validation_agent_node(state: AgentState) -> dict[str, Any]:
    try:
        with langfuse_observation(
            name="validation_agent",
            as_type="span",
            input=build_agent_trace_input(
                state,
                agent_name="validation_agent",
                result_key="validation_result",
            ),
            metadata={"agent": "validation_agent"},
        ) as observation:
            result = await _validation_agent_node_impl(state)
            validation_result = result.get("validation_result", {})
            update_observation(
                observation,
                output=build_agent_trace_output(
                    validation_result,
                    agent_name="validation_agent",
                    state=state,
                    result_key="validation_result",
                    extra_output={
                        "validation_passed": result.get("validation_passed"),
                        "revision_required": result.get("revision_required"),
                        "failure_type": result.get("failure_type"),
                        "missing_fields": result.get("missing_fields"),
                        "blocking_issues": result.get("blocking_issues"),
                        "retry_start_agent": result.get("retry_start_agent"),
                        "retry_reason": result.get("retry_reason"),
                        "validation_retry_count": result.get("validation_retry_count"),
                    },
                ),
                metadata={
                    "agent": "validation_agent",
                    "status": validation_result.get("status") if isinstance(validation_result, dict) else None,
                    "failure_type": result.get("failure_type"),
                },
            )
            return result
    finally:
        flush_langfuse()
