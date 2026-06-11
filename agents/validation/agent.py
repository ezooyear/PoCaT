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

from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import get_llm
from graph.state import AgentState
from agents.base import make_agent_result
from agents.validation.prompts import VALIDATION_SYSTEM_PROMPT
from agents.validation.tools import (
    build_validation_context,
    build_rule_based_verify_result,
    run_validation_checks,
)


# LLM 검증이 필요한 복잡한 작업만 지정합니다.
# 단순 고객 조회나 단순 상품 정보 조회는 rule 검증만으로 충분합니다.
LLM_VALIDATION_TASKS = {
    "financial_analysis",
    "eligibility_check",
    "recommendation",
    "early_termination",
    "switch_analysis",
}


def validation_agent_node(state: AgentState) -> dict[str, Any]:
    """
    Validation Agent 노드 함수.

    검증 방향:
    1. state에 저장된 구조화 결과 수집
    2. rule 기반 기본 검증 수행
    3. 복잡한 task_type이면 LLM 기반 verify_result 생성
    4. validation_result와 agent_outputs에 저장
    """

    validation_context = build_validation_context(state)

    # 1차 검증: 항상 실행
    rule_issues, rule_checked_items = run_validation_checks(state)

    # LLM 검증이 필요한 task_type인지 한 번만 판단
    should_run_llm = _should_run_llm_validation(state)

    # 2차 검증: 복잡한 task_type에서만 LLM 실행
    if should_run_llm:
        verify_result = _run_llm_verify_result(
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

    is_valid = bool(verify_result.get("is_valid"))

    validation_result = make_agent_result(
        status="success" if is_valid else "failed",
        result={
            "verify_result": verify_result,
            "is_valid": is_valid,
            "issues": verify_result.get("issues", []),
            "revision_required": verify_result.get("revision_required", not is_valid),
            "checked_items": verify_result.get("checked_items", rule_checked_items),
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

    return {
        "validation_result": validation_result,
        "agent_outputs": agent_outputs,
        "current_agent": "validation_agent",
        "completed_agents": completed_agents,
        "current_step": (state.get("current_step") or 0) + 1,
    }


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


def _run_llm_verify_result(
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
        response = llm.invoke(messages)
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
    elif has_warning_issue and status == "passed":
        status = "warning"

    if has_error_issue or has_warning_issue:
        revision_required = True

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