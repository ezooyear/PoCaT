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
from agents.base import build_agent_trace_input, build_agent_trace_output, make_agent_result, mirror_result_fields
from agents.validation.prompts import VALIDATION_SYSTEM_PROMPT
from agents.validation.tools import (
    build_validation_context,
    build_rule_based_verify_result,
    run_validation_checks,
)
from observability.langfuse import (
    flush_langfuse,
    langfuse_observation,
    safe_jsonable,
    update_observation,
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
    validation_summary = _extract_validation_summary(verify_result, state)

    validation_result = make_agent_result(
        status="success" if is_valid else "failed",
        result={
            "verify_result": verify_result,
            "status": "passed_with_warnings" if is_valid and verify_result.get("status") == "warning" else ("passed" if is_valid else "failed"),
            "is_valid": is_valid,
            "issues": verify_result.get("issues", []),
            "failure_reasons": validation_summary["blocking_issues"],
            "warnings": [
                issue.get("message")
                for issue in verify_result.get("issues", [])
                if isinstance(issue, dict) and str(issue.get("level")).lower() == "warning" and issue.get("message")
            ],
            "revision_required": validation_summary["revision_required"],

            # 추가: Supervisor final이 바로 읽을 수 있는 구조화 필드
            "failure_type": validation_summary["failure_type"],
            "missing_fields": validation_summary["missing_fields"],
            "blocking_issues": validation_summary["blocking_issues"],
            "awaiting_user_input": validation_summary["awaiting_user_input"],

            "checked_items": verify_result.get("checked_items", rule_checked_items),
            "checks": verify_result.get("checked_items", rule_checked_items),
            "summary": verify_result.get("summary"),
            "final_notes": verify_result.get("final_notes", []),
        },
        evidence=[],
        error=None if is_valid else "검증 이슈가 발견되었습니다.",
    )
    validation_result = mirror_result_fields(
        validation_result,
        field_names=[
            "verify_result",
            "is_valid",
            "issues",
            "failure_reasons",
            "warnings",
            "revision_required",
            "failure_type",
            "missing_fields",
            "blocking_issues",
            "awaiting_user_input",
            "checked_items",
            "checks",
            "final_notes",
        ],
    )

    agent_outputs = dict(state.get("agent_outputs") or {})
    agent_outputs["validation_agent"] = validation_result

    completed_agents = list(state.get("completed_agents") or [])
    if "validation_agent" not in completed_agents:
        completed_agents.append("validation_agent")

    return {
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

    for issue in issues:
        if isinstance(issue, dict):
            level = str(issue.get("level") or "").lower()
            message = str(issue.get("message") or "").strip()
            suggestion = str(issue.get("suggestion") or "").strip()

            if level in ["error", "warning"]:
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
    사용자 질문에서 추천에 필요한 최소 입력값이 빠졌는지 가볍게 확인합니다.
    복잡한 판단은 하지 않고, Supervisor가 물어볼 항목만 정리합니다.
    """
    task_type = state.get("task_type")
    if task_type != "recommendation":
        return []

    user_query = str(state.get("user_query") or "").replace(" ", "")
    missing_fields: list[str] = []

    has_customer_id = bool(state.get("customer_id")) or bool(
        re.search(r"(고객|customer|id|ID|고객ID|고객id|고객_)?\d+번?", user_query)
    )

    has_monthly_amount = bool(
        re.search(r"월?\d+만?원|\d{1,3}(,\d{3})*원", user_query)
    )

    has_period = bool(
        re.search(r"\d+개월|\d+년|[0-9]+개월|[0-9]+년", user_query)
    )

    has_product_type = "적금" in user_query or "예금" in user_query

    if not has_customer_id:
        missing_fields.append("고객 ID")

    if not has_monthly_amount:
        missing_fields.append("월 납입 예정 금액")

    if not has_period:
        missing_fields.append("희망 가입 기간")

    if not has_product_type:
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
    payload = safe_jsonable(payload)

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
    elif has_warning_issue and status == "passed":
        status = "warning"

    # error 없이 warning만 있으면 경고는 남기되 유효로 처리
    if not has_error_issue and status in ("passed", "warning"):
        is_valid = True

    if has_error_issue:
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


_validation_agent_node_impl = validation_agent_node


def validation_agent_node(state: AgentState) -> dict[str, Any]:
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
            result = _validation_agent_node_impl(state)
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
                    },
                ),
                metadata={
                    "agent": "validation_agent",
                    "status": validation_result.get("status") if isinstance(validation_result, dict) else None,
                    "result_key": "validation_result",
                    "input_sources": "customer_result;product_result;eligibility_result;financial_result;recommend_result",
                    "output_keys": "is_valid,failure_reasons,warnings,checks",
                    "failure_type": result.get("failure_type"),
                },
            )
            return result
    finally:
        flush_langfuse()
