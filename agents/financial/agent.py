"""
Financial 에이전트 - 이자 계산, 만기, 납입, 중도해지, 비교, 갈아타기
"""

import re
from datetime import date, datetime
from typing import Any

from langchain_core.messages import AIMessage

from agents.base import (
    LLMInvocationTimeoutError,
    build_agent_trace_input,
    build_agent_trace_output,
    make_agent_result,
    run_agent_loop,
    run_agent_loop_async,
)
from config.settings import get_resolved_llm_model
from graph.state import AgentState
from agents.financial.prompts import FINANCIAL_SYSTEM_PROMPT
from agents.financial.tools import FINANCIAL_TOOLS, compare_switch_benefit
from observability.langfuse import flush_langfuse, langfuse_observation, update_observation

try:
    from agents.product.tools import get_product_detail_map
except Exception:
    get_product_detail_map = None

DEFAULT_TAX_RATE = 0.154


async def financial_agent_node(state: AgentState) -> dict:
    resolved_model = get_resolved_llm_model()
    llm_error: Exception | None = None

    try:
        print(f"[FINANCIAL DEBUG] resolved_model={resolved_model} current_step={(state.get('current_step') or 0) + 1}")
        with langfuse_observation(
            name="financial_agent",
            as_type="span",
            input=build_agent_trace_input(
                state,
                agent_name="financial_agent",
                result_key="financial_result",
                max_iterations=5,
            ),
            metadata={"agent": "financial_agent", "resolved_model": resolved_model},
        ) as observation:
            try:
                result = await run_agent_loop_async(
                    state=state,
                    system_prompt=FINANCIAL_SYSTEM_PROMPT,
                    tools=FINANCIAL_TOOLS,
                    output_key="financial_agent",
                    result_key="financial_result",
                    max_iterations=5,
                    llm_max_retries=2,
                    span_name="financial_agent_llm",
                )
            except LLMInvocationTimeoutError as error:
                llm_error = error
                result = _build_rule_based_financial_fallback(state, str(error))
            except Exception as error:
                llm_error = error
                result = _build_rule_based_financial_fallback(state, str(error))

            if _should_use_financial_rule_fallback(result):
                recovered_error = _extract_financial_error_message(result)
                if recovered_error and llm_error is None:
                    llm_error = RuntimeError(recovered_error)
                result = _build_rule_based_financial_fallback(
                    state,
                    error_message=recovered_error,
                )

            if state.get("task_type") == "switch_analysis":
                result = _augment_switch_analysis_result(state, result)

            if state.get("task_type") == "recommendation":
                result = _augment_recommendation_financial_results(state, result)

            result = _augment_maturity_estimate_result(state, result)
            result = _ensure_financial_calculations(result)
            result = _enforce_financial_role_boundary(result)

            _update_financial_agent_observation(
                observation,
                state=state,
                result=result,
                resolved_model=resolved_model,
                llm_error=llm_error,
            )
            return result
    finally:
        flush_langfuse()


def _should_use_financial_rule_fallback(result: dict[str, Any]) -> bool:
    financial_result = result.get("financial_result")
    if not isinstance(financial_result, dict):
        return True

    status = str(financial_result.get("status") or "").strip().lower()
    if status == "failed":
        return True

    payload = financial_result.get("result")
    if not isinstance(payload, dict):
        return True

    return str(payload.get("status") or "").strip().lower() == "failed"


def _extract_financial_error_message(result: dict[str, Any]) -> str | None:
    financial_result = result.get("financial_result")
    if isinstance(financial_result, dict) and financial_result.get("error"):
        return str(financial_result.get("error"))
    return None


def _extract_financial_result_payload(result: dict[str, Any]) -> dict[str, Any]:
    financial_result = result.get("financial_result")
    if not isinstance(financial_result, dict):
        return {}

    payload = financial_result.get("result")
    return payload if isinstance(payload, dict) else {}


def _extract_financial_results_list(result: dict[str, Any]) -> list[dict[str, Any]]:
    payload = _extract_financial_result_payload(result)
    for key in ("financial_results", "calculations"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]

    raw = result.get("financial_results")
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    return []


def _is_successful_financial_fallback(result: dict[str, Any]) -> bool:
    payload = _extract_financial_result_payload(result)
    financial_result = result.get("financial_result")
    top_level_status = str((financial_result or {}).get("status") or "").strip().lower()
    payload_status = str(payload.get("status") or "").strip().lower()
    financial_results = _extract_financial_results_list(result)
    has_calculated = any(
        str(item.get("status") or "").strip().lower() == "calculated"
        for item in financial_results
    )
    return (
        payload_status == "fallback_success"
        and top_level_status in {"success", "fallback_success"}
        and has_calculated
    )


def _update_financial_agent_observation(
    observation: Any,
    *,
    state: AgentState,
    result: dict[str, Any],
    resolved_model: str,
    llm_error: Exception | None,
) -> None:
    if observation is None:
        return

    payload = _extract_financial_result_payload(result)
    financial_result = result.get("financial_result") if isinstance(result.get("financial_result"), dict) else {}
    financial_results = _extract_financial_results_list(result)
    fallback_used = bool(payload.get("fallback_applied")) or llm_error is not None
    fallback_success = _is_successful_financial_fallback(result)
    financial_results_count = len(financial_results)
    metadata = {
        "agent": "financial_agent",
        "resolved_model": resolved_model,
        "status": payload.get("status") or financial_result.get("status"),
        "llm_error_type": type(llm_error).__name__ if llm_error else None,
        "llm_error_message": str(llm_error) if llm_error else None,
        "fallback_used": fallback_used,
        "fallback_success": fallback_success,
        "financial_results_count": financial_results_count,
    }
    output = build_agent_trace_output(
        financial_result if isinstance(financial_result, dict) else result,
        agent_name="financial_agent",
        state=state,
        result_key="financial_result",
        extra_output={
            "financial_results_count": financial_results_count,
            "fallback_used": fallback_used,
            "fallback_success": fallback_success,
        },
    )

    try:
        observation.update(
            output=output,
            metadata=metadata,
            level="ERROR" if fallback_used and not fallback_success else "DEFAULT",
            status_message=(
                "financial_fallback_failed"
                if fallback_used and not fallback_success
                else "financial_fallback_success"
                if fallback_success
                else "financial_success"
            ),
        )
        return
    except Exception:
        pass

    update_observation(
        observation,
        output=output,
        metadata=metadata,
    )


def _build_rule_based_financial_fallback(
    state: AgentState,
    error_message: str | None = None,
) -> dict[str, Any]:
    base_result = {
        "financial_result": make_agent_result(
            status="success",
            result={
                "status": "fallback_success",
                "summary": "금융 계산 LLM 응답이 불안정해 규칙 기반 계산으로 대체했습니다.",
                "tool_results": [],
                "calculations": [],
                "financial_results": [],
                "fallback_applied": True,
                "fallback_reason": "rule_based_financial_recovery",
                "reason": "LLM 호출 실패 또는 시간 초과로 rule-based financial fallback을 사용했습니다.",
                "source_agent": "financial_agent",
            },
            evidence=[],
            error=None,
        ),
        "agent_outputs": dict(state.get("agent_outputs") or {}),
        "current_step": (state.get("current_step") or 0) + 1,
        "current_agent": "financial_agent",
        "completed_agents": list(state.get("completed_agents") or []),
        "financial_results": [],
        "context": {
            **(state.get("context") or {}),
            "financial_prompt": FINANCIAL_SYSTEM_PROMPT,
        },
    }

    fallback_result = _augment_recommendation_financial_results(state, base_result)
    fallback_result = _ensure_financial_calculations(fallback_result)
    fallback_result = _finalize_financial_rule_fallback(
        state=state,
        result=fallback_result,
        error_message=error_message,
    )
    return _enforce_financial_role_boundary(fallback_result)


def _build_financial_exception_fallback(state: AgentState, error_message: str) -> dict[str, Any]:
    fallback_result = make_agent_result(
        status="failed",
        result={
            "status": "failed",
            "summary": "금융 계산 처리 중 오류가 발생하여 결과를 안전한 fallback 상태로 반환합니다.",
            "calculations": [],
            "financial_results": [],
            "tool_results": [],
            "fallback_reason": "exception_in_financial_agent",
            "required_next_steps": ["financial_agent 예외 로그 확인"],
            "agent_name": "financial_agent",
            "current_step": (state.get("current_step") or 0) + 1,
            "result_key": "financial_result",
            "source_agent": "financial_agent",
        },
        evidence=[],
        error=error_message,
    )

    agent_outputs = dict(state.get("agent_outputs") or {})
    agent_outputs["financial_agent"] = fallback_result

    completed_agents = list(state.get("completed_agents") or [])
    if "financial_agent" not in completed_agents:
        completed_agents.append("financial_agent")

    errors = list(state.get("errors") or [])
    errors.append({"agent": "financial_agent", "error": error_message})

    return {
        "messages": [AIMessage(content=fallback_result["result"]["summary"])],
        "agent_outputs": agent_outputs,
        "current_step": (state.get("current_step") or 0) + 1,
        "current_agent": "financial_agent",
        "completed_agents": completed_agents,
        "financial_results": [],
        "financial_result": fallback_result,
        "errors": errors,
        "context": {
            **(state.get("context") or {}),
            "financial_prompt": FINANCIAL_SYSTEM_PROMPT,
        },
    }


def _finalize_financial_rule_fallback(
    *,
    state: AgentState,
    result: dict[str, Any],
    error_message: str | None,
) -> dict[str, Any]:
    financial_result = result.get("financial_result")
    if not isinstance(financial_result, dict):
        return _build_financial_exception_fallback(
            state,
            error_message or "financial fallback build failed",
        )

    payload = financial_result.get("result") or {}
    if not isinstance(payload, dict):
        payload = {}

    financial_results = payload.get("financial_results")
    if not isinstance(financial_results, list):
        financial_results = result.get("financial_results") or []
    if not isinstance(financial_results, list):
        financial_results = []

    affordable_products = list(
        dict.fromkeys(
            str(item.get("product_name"))
            for item in financial_results
            if isinstance(item, dict)
            and item.get("product_name")
            and str(item.get("status") or "").strip().lower()
            not in {"failed", "error", "rejected", "invalid_product"}
        )
    )

    calculated_monthly_saving = _extract_monthly_amount_from_customer_profile(state)
    if calculated_monthly_saving is None:
        calculated_monthly_saving = _extract_monthly_amount_from_query(_safe_user_query(state))

    expected_interest_summary = _build_expected_interest_summary(financial_results)
    has_usable_results = any(
        isinstance(item, dict)
        and str(item.get("status") or "").strip().lower()
        in {"calculated", "success", "fallback_success"}
        for item in financial_results
    )

    payload["status"] = "fallback_success" if has_usable_results or affordable_products else "needs_check"
    payload["reason"] = "LLM 호출 실패 또는 시간 초과로 rule-based financial fallback을 사용했습니다."
    payload["fallback_applied"] = True
    payload["fallback_reason"] = "rule_based_financial_recovery"
    payload["calculated_monthly_saving"] = calculated_monthly_saving
    payload["affordable_products"] = affordable_products
    payload["expected_interest_summary"] = expected_interest_summary
    payload["llm_error_recovered"] = bool(error_message)
    payload["llm_error_type"] = _classify_financial_error(error_message)
    payload["summary"] = _build_financial_fallback_summary(
        financial_results=financial_results,
        affordable_products=affordable_products,
        expected_interest_summary=expected_interest_summary,
    )

    financial_result["status"] = "success" if payload["status"] == "fallback_success" else "needs_check"
    financial_result["result"] = payload
    financial_result["evidence"] = financial_results
    financial_result["error"] = None

    agent_outputs = dict(result.get("agent_outputs") or {})
    agent_outputs["financial_agent"] = financial_result

    completed_agents = list(result.get("completed_agents") or state.get("completed_agents") or [])
    if "financial_agent" not in completed_agents:
        completed_agents.append("financial_agent")

    errors = list(state.get("errors") or [])
    if error_message:
        errors.append(
            {
                "agent": "financial_agent",
                "error": error_message,
                "recoverable": True,
                "user_visible": False,
                "fallback_applied": True,
            }
        )

    return {
        **result,
        "messages": [AIMessage(content=payload["summary"])],
        "agent_outputs": agent_outputs,
        "completed_agents": completed_agents,
        "financial_results": financial_results,
        "financial_result": financial_result,
        "errors": errors,
    }


def _build_expected_interest_summary(financial_results: list[dict[str, Any]]) -> str:
    calculated = [
        item for item in financial_results
        if isinstance(item, dict) and item.get("estimated_interest") is not None
    ]
    if not calculated:
        return "정확 계산이 어려운 상품은 금리와 우대조건 기준의 정성 평가를 사용했습니다."

    ranked = sorted(
        calculated,
        key=lambda item: int(item.get("estimated_interest") or 0),
        reverse=True,
    )[:3]
    parts = []
    for item in ranked:
        product_name = str(item.get("product_name") or "상품")
        rate = item.get("applied_rate")
        interest = int(item.get("estimated_interest") or 0)
        if rate is not None:
            parts.append(f"{product_name}: 예상 세후 이자 약 {interest:,}원, 적용금리 {float(rate):.2f}%")
        else:
            parts.append(f"{product_name}: 예상 세후 이자 약 {interest:,}원")
    return " / ".join(parts)


def _build_financial_fallback_summary(
    *,
    financial_results: list[dict[str, Any]],
    affordable_products: list[str],
    expected_interest_summary: str,
) -> str:
    lines = ["금융 계산은 규칙 기반으로 복구해 이어서 추천할 수 있도록 정리했습니다."]
    if affordable_products:
        lines.append(f"- 가입 가능 범위에서 검토 가능한 상품: {', '.join(affordable_products)}")
    if expected_interest_summary:
        lines.append(f"- 이자 판단 요약: {expected_interest_summary}")
    if not financial_results:
        lines.append("- 다만 계산 가능한 상품 정보가 부족해 정밀 이자 계산은 보류했습니다.")
    return "\n".join(lines)


def _classify_financial_error(error_message: str | None) -> str | None:
    if not error_message:
        return None

    lowered = error_message.lower()
    if "toomanyrequests" in lowered or "429" in lowered:
        return "rate_limit"
    if "timeout" in lowered:
        return "timeout"
    if "provider returned error" in lowered:
        return "provider_error"
    return "llm_error"


def _augment_maturity_estimate_result(state: AgentState, result: dict) -> dict:
    user_query = str(state.get("user_query") or _last_user_text(state.get("messages") or ""))
    if not _is_maturity_estimate_query(user_query):
        return result

    customer_text = _extract_tool_result(state.get("customer_result"), "get_customer_accounts")
    if not customer_text:
        return result

    current_account = _pick_current_account(customer_text, user_query)
    if not current_account:
        return result

    estimate = _calculate_active_account_maturity_estimate(current_account)
    if not estimate:
        return result

    estimate_text = _format_maturity_estimate(estimate)
    existing_summary = _extract_financial_summary(result)
    return _append_financial_tool_result(
        result=result,
        tool_name="estimate_active_account_maturity",
        tool_args={
            "account_number": estimate["account_number"],
            "product_name": estimate["product_name"],
            "current_balance": estimate["current_balance"],
            "monthly_amount": estimate["monthly_amount"],
            "applied_rate": estimate["applied_rate"],
            "remaining_months": estimate["remaining_months"],
            "maturity_date": estimate["maturity_date"],
            "tax_rate": DEFAULT_TAX_RATE,
        },
        tool_result=estimate_text,
        replace_missing_summary=True,
        replace_summary=_contains_latex_formula(existing_summary),
    )


def _enforce_financial_role_boundary(result: dict) -> dict:
    financial_result = result.get("financial_result")
    if not isinstance(financial_result, dict):
        return result

    payload = financial_result.get("result", {})
    if not isinstance(payload, dict):
        return result

    summary = str(payload.get("summary") or "")
    if not _contains_recommendation_claim(summary):
        return result

    tool_results = payload.get("tool_results", [])
    if not isinstance(tool_results, list):
        tool_results = []

    payload["summary"] = _build_calculation_only_summary(tool_results)
    payload["role_boundary_enforced"] = True
    payload["removed_recommendation_claim"] = summary
    financial_result["result"] = payload
    result["financial_result"] = financial_result

    agent_outputs = dict(result.get("agent_outputs") or {})
    if isinstance(agent_outputs.get("financial_agent"), dict):
        agent_outputs["financial_agent"]["result"] = payload
    result["agent_outputs"] = agent_outputs

    return result


def _contains_recommendation_claim(summary: str) -> bool:
    normalized = re.sub(r"\s+", "", str(summary or "")).lower()
    if not normalized:
        return False

    recommendation_patterns = [
        "가장잘맞는상품",
        "가장적합한상품",
        "추천상품",
        "추천드립니다",
        "추천합니다",
        "가입을권장",
        "선택하는것이좋",
        "bestproduct",
        "recommend",
    ]
    return any(pattern in normalized for pattern in recommendation_patterns)


def _build_calculation_only_summary(tool_results: list[dict[str, Any]]) -> str:
    calculation_outputs = [
        str(item.get("tool_result"))
        for item in tool_results
        if isinstance(item, dict) and item.get("tool_result")
    ]

    if calculation_outputs:
        return (
            "Financial Agent는 계산 및 비교 결과만 제공합니다. "
            "상품 추천이나 최종 선택 판단은 recommend_agent에서 수행해야 합니다.\n\n"
            + "\n\n".join(calculation_outputs)
        )

    return (
        "Financial Agent는 상품 추천을 생성하지 않습니다. "
        "계산에 필요한 입력값 또는 도구 실행 결과가 없어 추천성 문장을 제거했습니다."
    )


def _append_financial_tool_result(
    result: dict,
    tool_name: str,
    tool_args: dict[str, Any],
    tool_result: str,
    replace_missing_summary: bool = False,
    replace_summary: bool = False,
) -> dict:
    financial_result = result.get("financial_result")
    if not isinstance(financial_result, dict):
        return result

    payload = financial_result.get("result", {})
    if not isinstance(payload, dict):
        return result

    tool_results = payload.get("tool_results", [])
    if not isinstance(tool_results, list):
        tool_results = []

    if not any(isinstance(item, dict) and item.get("tool_name") == tool_name for item in tool_results):
        tool_results.append(
            {
                "tool_name": tool_name,
                "tool_args": tool_args,
                "tool_result": tool_result,
            }
        )
    payload["tool_results"] = tool_results

    summary = str(payload.get("summary") or "")
    if replace_summary or (replace_missing_summary and _looks_like_missing_info_summary(summary)):
        payload["summary"] = tool_result
    elif tool_result not in summary:
        payload["summary"] = f"{summary}\n\n{tool_result}".strip()

    financial_result["result"] = payload
    result["financial_result"] = financial_result

    agent_outputs = dict(result.get("agent_outputs") or {})
    if isinstance(agent_outputs.get("financial_agent"), dict):
        agent_outputs["financial_agent"]["result"] = payload
    result["agent_outputs"] = agent_outputs

    return result


def _extract_financial_summary(result: dict) -> str:
    financial_result = result.get("financial_result")
    if not isinstance(financial_result, dict):
        return ""
    payload = financial_result.get("result", {})
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("summary") or "")


def _contains_latex_formula(summary: str) -> bool:
    latex_markers = [
        r"\text",
        r"\times",
        r"\frac",
        r"\approx",
        r"\[",
        r"\]",
        "[ \\text",
    ]
    return any(marker in str(summary or "") for marker in latex_markers)


def _is_maturity_estimate_query(user_query: str) -> bool:
    normalized = re.sub(r"\s+", "", user_query or "")
    if "만기" not in normalized:
        return False

    targets = ["수령액", "이자", "세전이자", "세후이자", "받을돈", "받는돈"]
    actions = ["계산", "예상", "얼마", "알려", "받"]
    return any(keyword in normalized for keyword in targets) and any(
        keyword in normalized for keyword in actions
    )


def _last_user_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if isinstance(message, tuple) and len(message) >= 2 and message[0] in ["user", "human"]:
            return str(message[1])
        if getattr(message, "type", None) in ["user", "human"]:
            return str(getattr(message, "content", ""))
    return ""


def _calculate_active_account_maturity_estimate(account: dict[str, Any]) -> dict[str, Any] | None:
    current_balance = _to_int(account.get("current_balance"))
    monthly_amount = _to_int(account.get("monthly_amount")) or 0
    applied_rate = _to_float(account.get("applied_rate"))
    remaining_months = _calculate_remaining_months(account)
    maturity_date = _parse_date(account.get("maturity_date"))

    if not current_balance or current_balance <= 0:
        return None
    if applied_rate is None or applied_rate < 0:
        return None
    if not remaining_months or remaining_months <= 0:
        return None

    annual_rate = applied_rate / 100
    current_balance_interest = current_balance * annual_rate * remaining_months / 12
    future_principal = monthly_amount * remaining_months
    future_payment_interest = sum(
        monthly_amount * annual_rate * (remaining_months - payment_index) / 12
        for payment_index in range(remaining_months)
    )
    before_tax_interest = current_balance_interest + future_payment_interest
    tax = before_tax_interest * DEFAULT_TAX_RATE
    after_tax_interest = before_tax_interest - tax
    maturity_amount = current_balance + future_principal + after_tax_interest

    return {
        "account_number": str(account.get("account_number") or "-"),
        "product_name": str(account.get("product_name") or "-"),
        "product_type": str(account.get("product_type") or "-"),
        "current_balance": current_balance,
        "monthly_amount": monthly_amount,
        "future_principal": future_principal,
        "applied_rate": applied_rate,
        "remaining_months": remaining_months,
        "maturity_date": maturity_date.isoformat() if maturity_date else str(account.get("maturity_date") or "-"),
        "before_tax_interest": before_tax_interest,
        "tax": tax,
        "after_tax_interest": after_tax_interest,
        "maturity_amount": maturity_amount,
    }


def _format_maturity_estimate(estimate: dict[str, Any]) -> str:
    return "\n".join(
        [
            "현재 확인 가능한 계좌 정보 기준 만기 예상 수령액입니다.",
            "",
            "계산 기준",
            f"- 상품명: {estimate['product_name']}",
            f"- 계좌번호: {estimate['account_number']}",
            f"- 현재잔액: {_format_won(estimate['current_balance'])}",
            f"- 월 납입액: {_format_won(estimate['monthly_amount'])}",
            f"- 적용금리: {estimate['applied_rate']:.2f}%",
            f"- 남은 기간: {estimate['remaining_months']}개월",
            f"- 만기일: {estimate['maturity_date']}",
            f"- 적용 세율: {DEFAULT_TAX_RATE * 100:.1f}%",
            "",
            "계산 결과",
            f"- 남은 기간 추가 납입 원금: {_format_won(estimate['future_principal'])}",
            f"- 세전 예상 이자: {_format_won(estimate['before_tax_interest'])}",
            f"- 예상 세금: {_format_won(estimate['tax'])}",
            f"- 세후 예상 이자: {_format_won(estimate['after_tax_interest'])}",
            f"- 만기 예상 수령액: {_format_won(estimate['maturity_amount'])}",
            "",
            "이 계산은 현재잔액, 월 납입액, 기본 적용금리만 사용한 추정치입니다. 우대금리, 실제 납입일, 납입 누락 여부, 은행의 실제 이자 계산 방식에 따라 최종 금액은 달라질 수 있습니다.",
        ]
    )


def _format_won(value: Any) -> str:
    return f"{float(value):,.0f}원"


def _augment_switch_analysis_result(state: AgentState, result: dict) -> dict:
    customer_text = _extract_tool_result(state.get("customer_result"), "get_customer_accounts")
    product_candidates = _extract_product_candidates(state.get("product_result"))
    user_query = str(state.get("user_query") or "")

    if not customer_text or not product_candidates:
        return result

    current_account = _pick_current_account(customer_text, user_query)
    target_product = _pick_target_product(product_candidates, user_query)

    if not current_account or not target_product:
        return result

    current_balance = _to_int(current_account.get("current_balance"))
    current_rate = _to_float(current_account.get("applied_rate"))
    remaining_months = _calculate_remaining_months(current_account)
    new_rate = _to_float(target_product.get("annual_rate"))
    new_months = _to_int(target_product.get("target_months"))

    if not all([
        current_balance and current_balance > 0,
        current_rate is not None,
        remaining_months and remaining_months > 0,
        new_rate is not None,
        new_months and new_months > 0,
    ]):
        return result

    tool_args = {
        "current_balance": float(current_balance),
        "current_rate": float(current_rate),
        "remaining_months": int(remaining_months),
        "new_rate": float(new_rate),
        "new_months": int(new_months),
        "monthly_payment": float(_to_int(current_account.get("monthly_amount")) or 0),
        "new_product_info": target_product.get("raw_text", ""),
        "customer_accounts": customer_text,
    }
    switch_result = compare_switch_benefit.invoke(tool_args)

    financial_result = result.get("financial_result")
    if not isinstance(financial_result, dict):
        return result

    payload = financial_result.get("result", {})
    if not isinstance(payload, dict):
        return result

    tool_results = payload.get("tool_results", [])
    if not isinstance(tool_results, list):
        tool_results = []

    tool_results.append(
        {
            "tool_name": "compare_switch_benefit",
            "tool_args": tool_args,
            "tool_result": switch_result,
        }
    )
    payload["tool_results"] = tool_results

    summary = str(payload.get("summary") or "")
    if _looks_like_missing_info_summary(summary):
        payload["summary"] = switch_result
    else:
        payload["summary"] = f"{summary}\n\n{switch_result}".strip()

    financial_result["result"] = payload
    result["financial_result"] = financial_result

    agent_outputs = dict(result.get("agent_outputs") or {})
    if isinstance(agent_outputs.get("financial_agent"), dict):
        agent_outputs["financial_agent"]["result"] = payload
    result["agent_outputs"] = agent_outputs

    return result


def _ensure_financial_calculations(result: dict) -> dict:
    financial_result = result.get("financial_result")
    if not isinstance(financial_result, dict):
        return result

    payload = financial_result.get("result", {})
    if not isinstance(payload, dict):
        return result

    tool_results = payload.get("tool_results") or []
    calculations = _parse_financial_calculations(tool_results)

    payload["calculations"] = calculations
    if not calculations:
        payload["missing_fields"] = payload.get("missing_fields") or ["term_months", "applied_rate"]
        payload["fallback_reason"] = payload.get("fallback_reason") or "missing_required_calculation_fields"
        if payload.get("status") == "success":
            payload["status"] = "needs_check"
        if financial_result.get("status") == "success":
            financial_result["status"] = "needs_check"
        if not payload.get("summary"):
            payload["summary"] = "계산에 필요한 필수 정보가 부족하여 financial_result.calculations를 생성하지 못했습니다."

    financial_result["result"] = payload
    result["financial_result"] = financial_result

    agent_outputs = dict(result.get("agent_outputs") or {})
    if isinstance(agent_outputs.get("financial_agent"), dict):
        agent_outputs["financial_agent"]["result"] = payload
    result["agent_outputs"] = agent_outputs

    return result


def _parse_financial_calculations(tool_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calculations: list[dict[str, Any]] = []
    for item in tool_results:
        if not isinstance(item, dict):
            continue
        tool_name = item.get("tool_name")
        tool_args = item.get("tool_args") or {}
        tool_result = str(item.get("tool_result") or "")

        if tool_name == "estimate_active_account_maturity":
            calculation = {
                "product_name": str(tool_args.get("product_name") or "미확인 상품"),
                "product_type": str(tool_args.get("product_type") or "적금"),
                "monthly_amount": int(tool_args.get("monthly_amount") or 0),
                "term_months": int(tool_args.get("remaining_months") or 0),
                "payment_count": int(tool_args.get("remaining_months") or 0),
                "applied_rate": float(tool_args.get("applied_rate") or 0.0),
                "base_rate": float(tool_args.get("applied_rate") or 0.0),
                "bonus_rate": 0.0,
                "principal": int(tool_args.get("current_balance") or 0) + int(tool_args.get("monthly_amount") or 0) * int(tool_args.get("remaining_months") or 0),
                "estimated_interest_before_tax": _extract_amount_from_text(tool_result, "세전 예상 이자"),
                "estimated_interest_after_tax": _extract_amount_from_text(tool_result, "세후 예상 이자"),
                "estimated_maturity_amount": _extract_amount_from_text(tool_result, "만기 예상 수령액"),
                "calculation_method": "monthly_installment_simple_estimate",
                "calculation_note": "단순 추정 형식입니다.",
            }
            calculations.append(calculation)
        elif tool_name == "calculate_interest":
            calculation = {
                "product_name": str(tool_args.get("product_name") or "미확인 상품"),
                "product_type": str(tool_args.get("product_type") or "예금"),
                "monthly_amount": int(tool_args.get("monthly_payment") or 0),
                "term_months": int(tool_args.get("months") or 0),
                "payment_count": int(tool_args.get("months") or 0),
                "applied_rate": float(tool_args.get("annual_rate") or 0.0),
                "base_rate": float(tool_args.get("annual_rate") or 0.0),
                "bonus_rate": 0.0,
                "principal": int(tool_args.get("principal") or 0) or int(tool_args.get("monthly_payment") or 0) * int(tool_args.get("months") or 0),
                "estimated_interest_before_tax": _extract_amount_from_text(tool_result, "세전 이자"),
                "estimated_interest_after_tax": _extract_amount_from_text(tool_result, "세후 이자"),
                "estimated_maturity_amount": _extract_amount_from_text(tool_result, "만기 예상 수령액"),
                "calculation_method": "monthly_or_deposit_interest_estimate",
                "calculation_note": "실제 은행 계산 방식과 다를 수 있는 단순 추정치입니다.",
            }
            calculations.append(calculation)
    return calculations


def _extract_amount_from_text(text: str, label: str) -> int | None:
    pattern = rf"{re.escape(label)}[:\s]*([0-9,]+)원"
    match = re.search(pattern, text)
    if match:
        return int(match.group(1).replace(",", ""))
    return None


def _extract_tool_result(result_container: Any, tool_name: str) -> str:
    if not isinstance(result_container, dict):
        return ""
    payload = result_container.get("result", {})
    if not isinstance(payload, dict):
        return ""
    for item in payload.get("tool_results", []):
        if isinstance(item, dict) and item.get("tool_name") == tool_name:
            tool_result = item.get("tool_result")
            if isinstance(tool_result, str):
                return tool_result
    return ""


def _extract_product_candidates(product_result: Any) -> list[dict[str, Any]]:
    if not isinstance(product_result, dict):
        return []
    payload = product_result.get("result", {})
    if not isinstance(payload, dict):
        return []
    candidates = payload.get("product_candidates", [])
    return candidates if isinstance(candidates, list) else []


def _pick_current_account(accounts_text: str, user_query: str) -> dict[str, Any] | None:
    rows = _parse_table_rows(accounts_text)
    if not rows:
        return None

    query = user_query.replace(" ", "")
    active_rows = [row for row in rows if str(row.get("account_status", "")).upper() == "ACTIVE"]
    candidates = active_rows or rows

    if "적금" in query:
        savings = [row for row in candidates if "적금" in str(row.get("product_type", "")) or "적금" in str(row.get("product_name", ""))]
        if savings:
            candidates = savings
    elif "예금" in query:
        deposits = [row for row in candidates if "예금" in str(row.get("product_type", "")) or "예금" in str(row.get("product_name", ""))]
        if deposits:
            candidates = deposits

    candidates.sort(key=lambda row: _to_int(row.get("current_balance")) or 0, reverse=True)
    return candidates[0] if candidates else None


def _pick_target_product(product_candidates: list[dict[str, Any]], user_query: str) -> dict[str, Any] | None:
    if not product_candidates:
        return None

    query = re.sub(r"\s+", "", user_query).lower()
    for candidate in product_candidates:
        name = str(candidate.get("product_name", ""))
        normalized = re.sub(r"\s+", "", name).lower()
        if normalized and normalized in query:
            return _enrich_product_candidate(candidate)

    if "적금" in query:
        for candidate in product_candidates:
            if "적금" in str(candidate.get("product_name", "")):
                return _enrich_product_candidate(candidate)

    return _enrich_product_candidate(product_candidates[0])


def _enrich_product_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(candidate)
    raw_text = str(candidate.get("raw_text", ""))
    rates = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*%", raw_text)
    months = re.findall(r"([0-9]{1,2})\s*개월", raw_text)

    if rates:
        enriched["annual_rate"] = max(float(rate) for rate in rates)
    if months:
        enriched["target_months"] = max(int(month) for month in months)
    return enriched


def _parse_table_rows(table_text: str) -> list[dict[str, str]]:
    lines = [line.strip() for line in str(table_text or "").splitlines() if line.strip()]
    header = None
    rows = []

    for line in lines:
        if "|" not in line:
            continue
        parts = [part.strip() for part in line.split("|")]
        lower_line = line.lower()
        if not header and (
            "customer_id" in lower_line
            or "account_number" in lower_line
            or "current_balance" in lower_line
            or "monthly_amount" in lower_line
        ):
            header = parts
            continue
        if header and set(line) == {"-"}:
            continue
        if header and len(parts) == len(header):
            rows.append(dict(zip(header, parts)))

    return rows


def _calculate_remaining_months(account: dict[str, Any]) -> int | None:
    maturity_date = _parse_date(account.get("maturity_date"))
    if maturity_date:
        today = date.today()
        months = (maturity_date.year - today.year) * 12 + (maturity_date.month - today.month)
        if maturity_date.day > today.day:
            months += 1
        return max(months, 1)
    return _to_int(account.get("contract_months"))


def _parse_date(value: Any) -> date | None:
    text = str(value or "").strip()
    if not text or text == "NULL":
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _to_int(value: Any) -> int | None:
    text = str(value or "").strip()
    digits = re.sub(r"[^0-9]", "", text)
    if not digits:
        return None
    return int(digits)


def _to_float(value: Any) -> float | None:
    text = str(value or "").strip()
    match = re.search(r"[0-9]+(?:\.[0-9]+)?", text)
    if not match:
        return None
    return float(match.group(0))


def _looks_like_missing_info_summary(summary: str) -> bool:
    missing_markers = [
        "정보가 부족",
        "추가 정보",
        "필요 정보",
        "정확한",
        "바로 확인이 어렵",
        "계산하기 위해 필요한",
        "확보돼야",
        "알려 주시면",
    ]
    return any(marker in str(summary or "") for marker in missing_markers)

# helper
def _load_eligibility_status_by_product(state: AgentState) -> dict[str, dict[str, Any]]:
    """eligibility_agent 결과를 상품명 기준으로 조회할 수 있게 만듭니다."""
    results = state.get("eligibility_results") or []

    if not results:
        eligibility_result = state.get("eligibility_result") or {}
        payload = eligibility_result.get("result") if isinstance(eligibility_result, dict) else {}
        if isinstance(payload, dict):
            results = payload.get("results") or []

    mapping: dict[str, dict[str, Any]] = {}
    for item in results:
        if not isinstance(item, dict):
            continue
        name = str(item.get("product_name") or "").strip()
        if name:
            mapping[name] = item

    return mapping


def _select_rate_for_term(product: dict[str, Any], term_months: int | None) -> float | None:
    """
    추천 계산에 사용할 금리를 선택합니다.
    기간별 금리가 문서에 있으면 term_months에 맞는 금리를 우선 사용합니다.
    없으면 기존 max_rate/base_rate를 fallback으로 사용합니다.
    """
    if term_months is None:
        return None

    raw_text = str(product.get("raw_text") or "")
    product_name = str(product.get("product_name") or "")

    # KB나라사랑적금: 1년제/2년제/3년제 최종이율 표에서 기간별 최고금리 추출
    if "나라사랑" in product_name:
        section = raw_text
        if "최종이율" in section:
            section = section.split("최종이율", 1)[1]

        rates = [float(x) for x in re.findall(r"최고\s*연\s*([0-9]+(?:\.[0-9]+)?)\s*%", section)]
        if len(rates) >= 3:
            term_rate_map = {
                12: rates[0],
                24: rates[1],
                36: rates[2],
            }
            if term_months in term_rate_map:
                return term_rate_map[term_months]

    # 장병내일준비적금: 기간 구간별 최종 최고금리 추출
    if "장병내일" in product_name:
        rates = [float(x) for x in re.findall(r"최고\s*연\s*([0-9]+(?:\.[0-9]+)?)\s*%", raw_text)]
        if len(rates) >= 3:
            if term_months < 12:
                return rates[0]
            if term_months < 15:
                return rates[1]
            if term_months <= 24:
                return rates[2]

    return (
        _to_float_or_none(product.get("max_rate"))
        or _to_float_or_none(product.get("base_rate"))
    )

def _to_int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None

    if isinstance(value, (int, float)):
        return int(value)

    text = str(value).replace(",", "").strip()
    if not text:
        return None

    match = re.search(r"\d+", text)
    if not match:
        return None

    return int(match.group(0))


def _is_deposit_product(product: dict[str, Any]) -> bool:
    product_type = str(product.get("product_type") or "").strip()
    product_name = str(product.get("product_name") or "").strip()

    return product_type == "예금" or "예금" in product_name


def _extract_product_max_amount(product: dict[str, Any]) -> int | None:
    return (
        _to_int_or_none(product.get("max_amount"))
        or _to_int_or_none(product.get("max_monthly_amount"))
        or _to_int_or_none(product.get("max_deposit_amount"))
    )


def _extract_product_min_amount(product: dict[str, Any]) -> int | None:
    return (
        _to_int_or_none(product.get("min_amount"))
        or _to_int_or_none(product.get("min_monthly_amount"))
        or _to_int_or_none(product.get("min_deposit_amount"))
    )


def _resolve_recommendation_monthly_amount(
    *,
    product: dict[str, Any],
    customer_profile: dict[str, Any],
    requested_monthly_amount: int | None,
) -> tuple[int | None, str, str | None]:
    """
    추천 계산용 월 납입액을 결정한다.

    - 사용자가 직접 월 납입액을 말한 경우: 그 값을 우선 사용
    - 사용자가 말하지 않은 경우: 고객 DB의 월 저축 가능액 사용
    - 단, 상품별 월 납입한도가 있으면 그 한도까지만 계산
    """
    product_max_amount = _extract_product_max_amount(product)

    if requested_monthly_amount is not None:
        if product_max_amount is not None and requested_monthly_amount > product_max_amount:
            return (
                product_max_amount,
                "user_query_capped_by_product_limit",
                (
                    f"입력하신 월 납입액은 {requested_monthly_amount:,}원이지만, "
                    f"이 상품의 월 납입한도가 {product_max_amount:,}원이므로 "
                    f"월 {product_max_amount:,}원 기준으로 계산했습니다."
                ),
            )

        return requested_monthly_amount, "user_query", None

    customer_monthly_amount = _to_int_or_none(
        customer_profile.get("monthly_saving_amount")
        or customer_profile.get("available_monthly_saving")
    )

    if customer_monthly_amount is None:
        return None, "missing", None

    if product_max_amount is not None and customer_monthly_amount > product_max_amount:
        return (
            product_max_amount,
            "customer_db_capped_by_product_limit",
            (
                f"고객님의 월 저축 가능액은 {customer_monthly_amount:,}원이지만, "
                f"이 상품의 월 납입한도가 {product_max_amount:,}원이므로 "
                f"월 {product_max_amount:,}원 기준으로 계산했습니다."
            ),
        )

    return customer_monthly_amount, "customer_db", None

def _to_won_amount_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None

    if isinstance(value, (int, float)):
        amount = int(value)
        return amount if amount > 0 else None

    text = str(value).replace(",", "").strip()
    if not text:
        return None

    match = re.search(r"(\d+(?:\.\d+)?)\s*만원", text)
    if match:
        return int(float(match.group(1)) * 10000)

    match = re.search(r"(\d+(?:\.\d+)?)\s*천원", text)
    if match:
        return int(float(match.group(1)) * 1000)

    match = re.search(r"(\d+)\s*원", text)
    if match:
        return int(match.group(1))

    match = re.search(r"\d+", text)
    if match:
        return int(match.group(0))

    return None


def _is_deposit_product(product: dict[str, Any]) -> bool:
    product_type = str(product.get("product_type") or "").strip()
    product_name = str(product.get("product_name") or "").strip()

    return product_type == "예금" or "예금" in product_name


def _extract_product_max_amount(product: dict[str, Any]) -> int | None:
    return (
        _to_won_amount_or_none(product.get("max_amount"))
        or _to_won_amount_or_none(product.get("max_monthly_amount"))
        or _to_won_amount_or_none(product.get("max_deposit_amount"))
    )


def _resolve_product_monthly_amount(
    *,
    product: dict[str, Any],
    requested_monthly_amount: int | None,
    customer_monthly_amount: int | None,
) -> tuple[int | None, str | None, str | None, str | None]:
    """
    추천 계산에 사용할 상품별 월 납입액을 결정합니다.

    핵심:
    - 고객 월 저축 가능액은 '총 저축 가능액'
    - 상품 max_amount는 '해당 상품 1개에 넣을 수 있는 최대 납입액'
    - 따라서 고객 월 저축 가능액이 상품 한도보다 크면 상품 한도만큼만 계산합니다.
    """
    product_max_amount = _extract_product_max_amount(product)

    if requested_monthly_amount is not None:
        if product_max_amount is not None and requested_monthly_amount > product_max_amount:
            return (
                product_max_amount,
                "user_query_capped_by_product_limit",
                "사용자 입력 월 납입액(상품 한도 적용)",
                (
                    f"입력하신 월 납입액은 {requested_monthly_amount:,}원이지만, "
                    f"이 상품의 월 납입한도가 {product_max_amount:,}원이므로 "
                    f"월 {product_max_amount:,}원 기준으로 계산했습니다."
                ),
            )

        return (
            requested_monthly_amount,
            "user_query",
            "사용자 입력 월 납입액",
            None,
        )

    if customer_monthly_amount is None:
        return None, None, None, None

    if product_max_amount is not None and customer_monthly_amount > product_max_amount:
        return (
            product_max_amount,
            "customer_db_capped_by_product_limit",
            "고객 DB 월 저축 가능액(상품 한도 적용)",
            (
                f"고객님의 월 저축 가능액은 {customer_monthly_amount:,}원이지만, "
                f"이 상품의 월 납입한도가 {product_max_amount:,}원이므로 "
                f"월 {product_max_amount:,}원 기준으로 계산했습니다."
            ),
        )

    return (
        customer_monthly_amount,
        "customer_profile.monthly_saving_amount",
        "고객 DB 월 저축 가능액",
        None,
    )

def _augment_recommendation_financial_results(state: AgentState, result: dict) -> dict:
    """
    추천 흐름에서 recommend_agent가 읽을 수 있는 financial_results를 만듭니다.

    핵심 원칙:
    - transaction_months는 고객 거래개월 수이므로 계약기간으로 사용하지 않습니다.
    - 계약기간은 사용자 입력값 또는 상품 DB 조건에서만 가져옵니다.
    - eligibility_agent에서 rejected/needs_check로 판단한 상품은 계산하지 않습니다.
    - 고객 DB의 월 저축 가능액은 '총 저축 가능액'이므로,
      상품별 계산에서는 상품 max_amount를 넘지 않도록 cap을 적용합니다.
    - 예금 상품은 월 납입액이 아니라 예치금 기준이므로,
      예치 가능 금액이 없으면 계산을 보류합니다.
    """
    product_candidates = _load_product_candidates_for_financial(state)
    if not product_candidates:
        return result

    user_query = _safe_user_query(state)

    requested_monthly_amount = _extract_monthly_amount_from_query(user_query)
    requested_term_months = _extract_term_months_from_query(user_query)

    customer_monthly_amount = None
    if requested_monthly_amount is None:
        customer_monthly_amount = _extract_monthly_amount_from_customer_profile(state)

    eligibility_by_name = _load_eligibility_status_by_product(state)
    financial_results: list[dict[str, Any]] = []

    for product in product_candidates:
        if not isinstance(product, dict):
            continue

        product = _enrich_product_with_db_fields(product)
        product_name = str(product.get("product_name") or "").strip()
        if not product_name:
            continue

        evidence = product.get("evidence") or []

        monthly_amount, monthly_amount_source, monthly_amount_source_label, amount_cap_note = (
            _resolve_product_monthly_amount(
                product=product,
                requested_monthly_amount=requested_monthly_amount,
                customer_monthly_amount=customer_monthly_amount,
            )
        )

        eligibility_item = _get_eligibility_item_by_product_name(eligibility_by_name, product_name)
        if isinstance(eligibility_item, dict):
            eligibility_status = eligibility_item.get("status")
            is_eligible = eligibility_item.get("eligible") is True

            if not is_eligible or eligibility_status != "eligible":
                financial_results.append(
                    {
                        "product_name": product_name,
                        "status": "needs_check",
                        "monthly_amount": monthly_amount,
                        "monthly_amount_source": monthly_amount_source,
                        "monthly_amount_source_label": monthly_amount_source_label,
                        "amount_cap_note": amount_cap_note,
                        "term_months": requested_term_months,
                        "term_months_source": "user_query" if requested_term_months is not None else None,
                        "term_months_source_label": "사용자 입력 가입기간" if requested_term_months is not None else None,
                        "applied_rate": None,
                        "estimated_interest": None,
                        "maturity_amount": None,
                        "reason": f"eligibility_agent에서 {eligibility_status}로 판단되어 금융 계산을 보류했습니다.",
                        "evidence": evidence,
                        "source_agent": "financial_agent",
                    }
                )
                continue

        if _is_deposit_product(product):
            financial_results.append(
                {
                    "product_name": product_name,
                    "status": "needs_check",
                    "monthly_amount": None,
                    "monthly_amount_source": None,
                    "monthly_amount_source_label": None,
                    "amount_cap_note": None,
                    "term_months": requested_term_months,
                    "term_months_source": "user_query" if requested_term_months is not None else None,
                    "term_months_source_label": "사용자 입력 가입기간" if requested_term_months is not None else None,
                    "applied_rate": None,
                    "estimated_interest": None,
                    "maturity_amount": None,
                    "reason": "예금 상품은 월 납입액이 아니라 한 번에 예치할 금액이 필요해 계산을 보류했습니다.",
                    "calculation_assumption": "예금 상품 계산에는 예치 가능 금액이 필요합니다.",
                    "evidence": evidence,
                    "source_agent": "financial_agent",
                }
            )
            continue

        term_months, term_months_source, term_months_source_label = _resolve_recommendation_term_months(
            product=product,
            requested_term_months=requested_term_months,
        )

        applied_rate = _select_rate_for_term(product, term_months)

        if monthly_amount is None or term_months is None or applied_rate is None:
            reason_parts = []

            if monthly_amount is None:
                reason_parts.append("월 납입액을 확인할 수 없습니다.")
            if term_months is None:
                reason_parts.append("계약기간을 확인할 수 없습니다.")
            if applied_rate is None:
                reason_parts.append("적용금리를 확인할 수 없습니다.")
            if amount_cap_note:
                reason_parts.append(amount_cap_note)

            financial_results.append(
                {
                    "product_name": product_name,
                    "status": "needs_check",
                    "monthly_amount": monthly_amount,
                    "monthly_amount_source": monthly_amount_source,
                    "monthly_amount_source_label": monthly_amount_source_label,
                    "amount_cap_note": amount_cap_note,
                    "term_months": term_months,
                    "term_months_source": term_months_source,
                    "term_months_source_label": term_months_source_label,
                    "applied_rate": applied_rate,
                    "estimated_interest": None,
                    "maturity_amount": None,
                    "reason": " ".join(reason_parts) or "월 납입액, 계약기간 또는 적용금리 정보가 부족해 계산을 확정하지 못했습니다.",
                    "evidence": evidence,
                    "source_agent": "financial_agent",
                }
            )
            continue

        calculation = _calculate_savings_projection(
            monthly_amount=monthly_amount,
            term_months=term_months,
            annual_rate=applied_rate,
        )

        calculation_assumption = _build_calculation_assumption(
            monthly_amount=monthly_amount,
            monthly_amount_source_label=monthly_amount_source_label,
            term_months=term_months,
            term_months_source_label=term_months_source_label,
        )

        if amount_cap_note:
            calculation_assumption = f"{calculation_assumption} {amount_cap_note}"

        reason = _build_calculated_reason(
            monthly_amount=monthly_amount,
            term_months=term_months,
            after_tax_interest=calculation["after_tax_interest"],
            maturity_amount=calculation["maturity_amount"],
            monthly_amount_source_label=monthly_amount_source_label,
            term_months_source_label=term_months_source_label,
        )

        if amount_cap_note:
            reason = f"{reason} {amount_cap_note}"

        financial_results.append(
            {
                "product_name": product_name,
                "status": "calculated",
                "monthly_amount": monthly_amount,
                "monthly_amount_source": monthly_amount_source,
                "monthly_amount_source_label": monthly_amount_source_label,
                "amount_cap_note": amount_cap_note,
                "term_months": term_months,
                "term_months_source": term_months_source,
                "term_months_source_label": term_months_source_label,
                "applied_rate": applied_rate,
                "estimated_interest": calculation["after_tax_interest"],
                "maturity_amount": calculation["maturity_amount"],
                "total_principal": calculation["total_principal"],
                "before_tax_interest": calculation["before_tax_interest"],
                "tax": calculation["tax"],
                "calculation_assumption": calculation_assumption,
                "payment_plan_text": _build_payment_plan_text(monthly_amount, term_months),
                "reason": reason,
                "evidence": evidence,
                "source_agent": "financial_agent",
            }
        )

    if not financial_results:
        return result

    financial_result = result.get("financial_result")
    if not isinstance(financial_result, dict):
        return result

    payload = financial_result.get("result", {})
    if not isinstance(payload, dict):
        payload = {}

    calculated_results = [
        item for item in financial_results
        if item.get("status") == "calculated"
    ]

    payload["financial_results"] = financial_results
    payload["calculations"] = calculated_results
    payload["summary"] = _build_recommendation_financial_summary(financial_results)

    if calculated_results:
        payload["missing_fields"] = []
        payload["fallback_reason"] = None
        financial_result["status"] = "success"
    else:
        payload["missing_fields"] = payload.get("missing_fields") or ["term_months", "applied_rate"]
        payload["fallback_reason"] = payload.get("fallback_reason") or "no_calculated_financial_results"
        financial_result["status"] = "needs_check"

    financial_result["result"] = payload
    financial_result["evidence"] = financial_results

    result["financial_result"] = financial_result
    result["financial_results"] = financial_results

    agent_outputs = dict(result.get("agent_outputs") or {})
    agent_outputs["financial_agent"] = financial_result
    result["agent_outputs"] = agent_outputs

    return result

def _load_product_candidates_for_financial(state: AgentState) -> list[dict]:
    candidates = state.get("product_candidates")
    if isinstance(candidates, list) and candidates:
        return candidates

    product_result = state.get("product_result") or {}
    if isinstance(product_result, dict):
        payload = product_result.get("result", {})
        if isinstance(payload, dict):
            candidates = payload.get("products") or payload.get("product_candidates")
            if isinstance(candidates, list) and candidates:
                return candidates

    agent_outputs = state.get("agent_outputs") or {}
    product_agent = agent_outputs.get("product_agent") or {}
    if isinstance(product_agent, dict):
        payload = product_agent.get("result", {})
        if isinstance(payload, dict):
            candidates = payload.get("products") or payload.get("product_candidates")
            if isinstance(candidates, list) and candidates:
                return candidates

    return []


def _extract_monthly_amount_from_query(user_query: str) -> int | None:
    normalized = str(user_query or "").replace(",", "").replace(" ", "")

    match = re.search(r"월([0-9]+)만원", normalized)
    if match:
        return int(match.group(1)) * 10000

    match = re.search(r"월([0-9]+)원", normalized)
    if match:
        return int(match.group(1))

    return None


INTERNAL_PERIOD_CONTEXT_MARKERS = [
    "transaction_months",
    "bank_transaction_months",
    "contract_months",
    "remaining_months",
    "transaction months",
    "contract months",
    "remaining months",
    "거래개월",
    "거래 개월",
    "고객 거래",
    "거래기간",
    "가입계좌",
    "가입 계좌",
    "기존계좌",
    "기존 계좌",
    "남은기간",
    "남은 기간",
    "만기까지",
]


def _safe_user_query(state: AgentState) -> str:
    """
    계산 조건은 실제 사용자 발화에서만 가져옵니다.

    state["user_query"]에 customer_profile, agent_outputs 같은 내부 컨텍스트가
    섞이는 경우가 있어 messages의 마지막 human/user 메시지를 우선 사용합니다.
    그래도 없으면 state["user_query"]를 쓰되 내부 컨텍스트 라인은 제거합니다.
    """
    message_query = _last_user_text(state.get("messages") or [])
    if message_query:
        return _strip_internal_context_lines(message_query)

    return _strip_internal_context_lines(str(state.get("user_query") or ""))


def _strip_internal_context_lines(text: str) -> str:
    lines = str(text or "").splitlines()
    kept: list[str] = []

    for line in lines:
        normalized = line.lower()
        if any(marker.lower() in normalized for marker in INTERNAL_PERIOD_CONTEXT_MARKERS):
            continue
        if any(marker in line for marker in ["customer_profile", "agent_outputs", "financial_result", "eligibility_result"]):
            continue
        kept.append(line)

    return "\n".join(kept).strip()


def _is_internal_period_context(text: str, start: int, end: int) -> bool:
    window = str(text or "")[max(0, start - 50): min(len(str(text or "")), end + 50)]
    lowered = window.lower()
    return any(marker.lower() in lowered for marker in INTERNAL_PERIOD_CONTEXT_MARKERS)


def _extract_term_months_from_query(user_query: str) -> int | None:
    """
    사용자가 직접 말한 가입기간만 추출합니다.
    내부 필드의 transaction_months/contract_months/remaining_months는 제외합니다.
    """
    text = _strip_internal_context_lines(str(user_query or "").replace(",", ""))

    for year_match in re.finditer(r"([0-9]{1,2})\s*년", text):
        if not _is_internal_period_context(text, year_match.start(), year_match.end()):
            return int(year_match.group(1)) * 12

    for month_match in re.finditer(r"([0-9]{1,2})\s*개월", text):
        if not _is_internal_period_context(text, month_match.start(), month_match.end()):
            return int(month_match.group(1))

    return None

# helper 
_PRODUCT_DETAIL_MAP_CACHE: dict[str, dict[str, Any]] | None = None


def _normalize_product_lookup_name(name: Any) -> str:
    return re.sub(r"\s+", "", str(name or "")).lower()


def _get_product_detail_map_cached() -> dict[str, dict[str, Any]]:
    global _PRODUCT_DETAIL_MAP_CACHE

    if _PRODUCT_DETAIL_MAP_CACHE is not None:
        return _PRODUCT_DETAIL_MAP_CACHE

    if get_product_detail_map is None:
        _PRODUCT_DETAIL_MAP_CACHE = {}
        return _PRODUCT_DETAIL_MAP_CACHE

    try:
        raw_map = get_product_detail_map()
    except Exception:
        _PRODUCT_DETAIL_MAP_CACHE = {}
        return _PRODUCT_DETAIL_MAP_CACHE

    if not isinstance(raw_map, dict):
        _PRODUCT_DETAIL_MAP_CACHE = {}
        return _PRODUCT_DETAIL_MAP_CACHE

    normalized_map: dict[str, dict[str, Any]] = {}

    for key, value in raw_map.items():
        if isinstance(value, dict):
            product_name = value.get("product_name") or key
            normalized_map[_normalize_product_lookup_name(product_name)] = value
            normalized_map[_normalize_product_lookup_name(key)] = value

    _PRODUCT_DETAIL_MAP_CACHE = normalized_map
    return _PRODUCT_DETAIL_MAP_CACHE


def _enrich_product_with_db_fields(product: dict[str, Any]) -> dict[str, Any]:
    """
    financial 계산에 필요한 기간/금리/금액 정보를 DB products 기준으로 보강합니다.
    """
    enriched = dict(product or {})
    product_name = enriched.get("product_name")

    detail_map = _get_product_detail_map_cached()
    db_detail = detail_map.get(_normalize_product_lookup_name(product_name))

    if not isinstance(db_detail, dict):
        return enriched

    db_first_keys = [
        "product_id",
        "product_name",
        "product_type",
        "min_amount",
        "max_amount",
        "min_period_months",
        "max_period_months",
        "base_rate",
        "max_rate",
        "age_min",
        "age_max",
        "join_channel",
        "rag_document_key",
        "is_active",
    ]

    for key in db_first_keys:
        value = db_detail.get(key)
        if value not in (None, ""):
            enriched[key] = value

    return enriched

def _resolve_recommendation_term_months(
    *,
    product: dict[str, Any],
    requested_term_months: int | None,
) -> tuple[int | None, str | None, str | None]:
    """
    추천 계산 기간을 결정합니다.

    - 사용자가 기간을 말했으면 그 기간을 사용합니다.
    - 사용자가 기간을 말하지 않았으면 상품의 가입 가능 기간을 기준으로 계산하되,
      반드시 '가정'임을 source/label에 남깁니다.
    - 고객 거래기간(transaction_months)은 사용하지 않습니다.
    """
    if requested_term_months is not None:
        return requested_term_months, "user_query", "사용자 입력 가입기간"

    # 강사님 수정으로 product_agent가 products 테이블의 정형 기간을 넣어주므로,
    # RAG raw_text에서 뽑은 기간보다 DB 정형값을 우선합니다.
    max_period = _to_int_or_none(product.get("max_period_months"))
    if max_period is not None:
        return max_period, "product_max_period_assumption", "사용자 기간 미지정: 상품 최대 가입기간 가정"

    min_period = _to_int_or_none(product.get("min_period_months"))
    if min_period is not None:
        return min_period, "product_min_period_assumption", "사용자 기간 미지정: 상품 최소 가입기간 가정"

    term_options = _extract_term_options_from_product(product)
    if term_options:
        # 여러 기간이 있으면 비교 가능성을 위해 가장 긴 가입 가능 기간을 대표 계산기간으로 사용한다.
        selected = max(term_options)
        return selected, "product_term_option_assumption", "사용자 기간 미지정: 상품 가입 가능 기간 중 최대기간 가정"

    return None, None, None


def _extract_term_options_from_product(product: dict[str, Any]) -> list[int]:
    raw_options = product.get("term_options_months") or product.get("term_options") or []
    options: list[int] = []

    if isinstance(raw_options, list):
        for value in raw_options:
            parsed = _to_int_or_none(value)
            if parsed is not None:
                options.append(parsed)

    raw_text = str(product.get("raw_text") or "")
    for value in re.findall(r"([0-9]{1,2})\s*개월", raw_text):
        parsed = _to_int_or_none(value)
        if parsed is not None:
            options.append(parsed)

    min_period = _to_int_or_none(product.get("min_period_months"))
    max_period = _to_int_or_none(product.get("max_period_months"))
    if min_period is not None:
        options.append(min_period)
    if max_period is not None:
        options.append(max_period)

    # 1~120개월 범위만 계약기간 후보로 인정해 페이지 번호/금리 숫자 오염을 줄입니다.
    return sorted({value for value in options if 1 <= value <= 120})


def _build_payment_plan_text(monthly_amount: int | None, term_months: int | None) -> str:
    if monthly_amount is None or term_months is None:
        return "납입 계획을 확정하려면 월 납입액과 가입기간이 필요합니다."
    return f"{term_months}개월 동안 월 {_format_won(monthly_amount)}씩 납입"


def _build_calculation_assumption(
    *,
    monthly_amount: int | None,
    monthly_amount_source_label: str | None,
    term_months: int | None,
    term_months_source_label: str | None,
) -> str:
    amount_part = (
        f"{monthly_amount_source_label} {_format_won(monthly_amount)}"
        if monthly_amount is not None and monthly_amount_source_label
        else "월 납입액 미확정"
    )
    term_part = (
        f"{term_months_source_label} {term_months}개월"
        if term_months is not None and term_months_source_label
        else "가입기간 미확정"
    )
    return f"{amount_part}, {term_part} 기준으로 계산했습니다."


def _build_calculated_reason(
    *,
    monthly_amount: int,
    term_months: int,
    after_tax_interest: int,
    maturity_amount: int,
    monthly_amount_source_label: str | None,
    term_months_source_label: str | None,
) -> str:
    source_note = _build_calculation_assumption(
        monthly_amount=monthly_amount,
        monthly_amount_source_label=monthly_amount_source_label,
        term_months=term_months,
        term_months_source_label=term_months_source_label,
    )
    return (
        f"{term_months}개월 동안 월 {_format_won(monthly_amount)}씩 납입하면 "
        f"예상 세후 이자는 {_format_won(after_tax_interest)}, "
        f"만기 예상액은 {_format_won(maturity_amount)}입니다. "
        f"{source_note}"
    )


def _extract_monthly_amount_from_customer_profile(state: AgentState) -> int | None:
    profile = state.get("customer_profile") or {}

    if isinstance(profile, dict):
        amount = profile.get("monthly_saving_amount")
        parsed = _to_int_or_none(amount)
        if parsed is not None:
            return parsed

    eligibility_result = state.get("eligibility_result") or {}
    if isinstance(eligibility_result, dict):
        payload = eligibility_result.get("result", {})
        if isinstance(payload, dict):
            profile = payload.get("customer_profile") or {}
            if isinstance(profile, dict):
                return _to_int_or_none(profile.get("monthly_saving_amount"))

    return None


def _calculate_savings_projection(
    monthly_amount: int,
    term_months: int,
    annual_rate: float,
    tax_rate: float = DEFAULT_TAX_RATE,
) -> dict[str, int]:
    total_principal = monthly_amount * term_months

    before_tax_interest = sum(
        monthly_amount * (annual_rate / 100) * (term_months - payment_index) / 12
        for payment_index in range(term_months)
    )

    tax = before_tax_interest * tax_rate
    after_tax_interest = before_tax_interest - tax
    maturity_amount = total_principal + after_tax_interest

    return {
        "total_principal": int(total_principal),
        "before_tax_interest": int(before_tax_interest),
        "tax": int(tax),
        "after_tax_interest": int(after_tax_interest),
        "maturity_amount": int(maturity_amount),
    }


def _build_recommendation_financial_summary(financial_results: list[dict[str, Any]]) -> str:
    lines = [
        "추천 후보 상품별 예상 만기금액 계산 결과입니다.",
        "사용자가 월 납입액 또는 가입기간을 직접 말하지 않은 경우에는 고객 DB의 월 저축 가능액과 상품 가입 가능 기간을 기준으로 계산 가정을 표시합니다.",
        "",
        "| 상품명 | 상태 | 월 납입액 | 납입액 기준 | 기간 | 기간 기준 | 적용금리 | 예상 세후 이자 | 만기 예상액 |",
        "|---|---|---:|---|---:|---|---:|---:|---:|",
    ]

    for item in financial_results:
        product_name = item.get("product_name", "-")
        status = item.get("status", "-")
        monthly_amount = item.get("monthly_amount")
        monthly_source = item.get("monthly_amount_source_label") or "-"
        term_months = item.get("term_months")
        term_source = item.get("term_months_source_label") or "-"
        applied_rate = item.get("applied_rate")
        estimated_interest = item.get("estimated_interest")
        maturity_amount = item.get("maturity_amount")

        lines.append(
            "| {product_name} | {status} | {monthly} | {monthly_source} | {term} | {term_source} | {rate} | {interest} | {maturity} |".format(
                product_name=product_name,
                status=status,
                monthly=f"{int(monthly_amount):,}원" if monthly_amount is not None else "-",
                monthly_source=monthly_source,
                term=f"{int(term_months)}개월" if term_months is not None else "-",
                term_source=term_source,
                rate=f"{float(applied_rate):.2f}%" if applied_rate is not None else "-",
                interest=f"{int(estimated_interest):,}원" if estimated_interest is not None else "-",
                maturity=f"{int(maturity_amount):,}원" if maturity_amount is not None else "-",
            )
        )

    return "\n".join(lines)


def _to_int_or_none(value: Any) -> int | None:
    if value in (None, "", []):
        return None

    try:
        return int(float(str(value).replace(",", "")))
    except Exception:
        return None


def _to_float_or_none(value: Any) -> float | None:
    if value in (None, "", []):
        return None

    try:
        return float(str(value).replace(",", "").replace("%", ""))
    except Exception:
        return None


# retry loop fix - final safety overrides for recommendation financial calculation
# NOTE:
# 아래 함수들은 파일 하단에서 같은 이름의 기존 helper들을 덮어써서 사용됩니다.
# 목적:
# 1) KB 스타 적금Ⅲ 월 납입한도 300,000원 cap 누락 방지
# 2) product_name 표기 차이로 DB products 보강이 실패하는 문제 완화
# 3) eligibility 결과가 name/product_name 중 어떤 키로 와도 매칭
# 4) recommendation용 financial_results/calculations가 _ensure_financial_calculations에서 지워지는 문제 방지

_PRODUCT_FIELD_FALLBACKS: list[tuple[tuple[str, ...], dict[str, Any]]] = [
    (
        ("kb스타적금3", "kb스타적금iii", "kbstar적금3"),
        {
            "product_name": "KB 스타 적금Ⅲ",
            "product_type": "적금",
            "min_amount": 10000,
            "max_amount": 300000,
            "min_period_months": 12,
            "max_period_months": 12,
        },
    ),
]


def _normalize_product_lookup_name(name: Any) -> str:
    text = str(name or "").lower()

    replacements = {
        "ⅰ": "1",
        "Ⅰ": "1",
        "ⅱ": "2",
        "Ⅱ": "2",
        "ⅲ": "3",
        "Ⅲ": "3",
        "iii": "3",
        "ii": "2",
        "i": "1",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)

    text = text.replace("국민은행", "")
    text = text.replace("kb국민", "kb")
    text = re.sub(r"[^0-9a-z가-힣]+", "", text)
    return text


def _product_lookup_blob(product: dict[str, Any]) -> str:
    lookup_values = [
        product.get("product_name"),
        product.get("product_key"),
        product.get("rag_document_key"),
        product.get("source_file"),
        product.get("file_name"),
        product.get("document_name"),
        product.get("raw_text"),
    ]
    return _normalize_product_lookup_name(" ".join(str(value or "") for value in lookup_values))


def _get_product_field_fallback(product: dict[str, Any], field_name: str) -> Any:
    blob = _product_lookup_blob(product)
    if not blob:
        return None

    for aliases, values in _PRODUCT_FIELD_FALLBACKS:
        if any(_normalize_product_lookup_name(alias) in blob for alias in aliases):
            return values.get(field_name)

    return None


def _to_int_or_none(value: Any) -> int | None:
    if value in (None, "", []):
        return None

    if isinstance(value, (int, float)):
        return int(value)

    text = str(value).replace(",", "").strip()
    if not text:
        return None

    match = re.search(r"(\d+(?:\.\d+)?)\s*만원", text)
    if match:
        return int(float(match.group(1)) * 10000)

    match = re.search(r"(\d+(?:\.\d+)?)\s*천원", text)
    if match:
        return int(float(match.group(1)) * 1000)

    match = re.search(r"(\d+)\s*원", text)
    if match:
        return int(match.group(1))

    match = re.search(r"\d+", text)
    if not match:
        return None

    return int(match.group(0))


def _extract_product_max_amount(product: dict[str, Any]) -> int | None:
    parsed_amount = (
        _to_won_amount_or_none(product.get("max_amount"))
        or _to_won_amount_or_none(product.get("max_monthly_amount"))
        or _to_won_amount_or_none(product.get("max_deposit_amount"))
        or _to_int_or_none(product.get("max_amount"))
        or _to_int_or_none(product.get("max_monthly_amount"))
        or _to_int_or_none(product.get("max_deposit_amount"))
    )

    fallback_amount = _get_product_field_fallback(product, "max_amount")
    fallback_amount = _to_int_or_none(fallback_amount)

    # fallback은 실제 상품 한도를 알고 있는 경우에만 사용합니다.
    # parsed_amount가 없거나, 잘못 크게 잡힌 경우에는 fallback 한도로 낮춥니다.
    if fallback_amount is not None and (
        parsed_amount is None or parsed_amount > fallback_amount
    ):
        return fallback_amount

    return parsed_amount


def _extract_product_min_amount(product: dict[str, Any]) -> int | None:
    return (
        _to_won_amount_or_none(product.get("min_amount"))
        or _to_won_amount_or_none(product.get("min_monthly_amount"))
        or _to_won_amount_or_none(product.get("min_deposit_amount"))
        or _to_int_or_none(product.get("min_amount"))
        or _to_int_or_none(product.get("min_monthly_amount"))
        or _to_int_or_none(product.get("min_deposit_amount"))
        or _to_int_or_none(_get_product_field_fallback(product, "min_amount"))
    )


def _is_deposit_product(product: dict[str, Any]) -> bool:
    product_type = str(product.get("product_type") or "").strip()
    product_name = str(product.get("product_name") or "").strip()

    return product_type == "예금" or "예금" in product_name


def _get_product_detail_map_cached() -> dict[str, dict[str, Any]]:
    global _PRODUCT_DETAIL_MAP_CACHE

    # 빈 dict를 영구 캐시하면 DB import/초기화 타이밍 문제 이후 계속 보강이 실패할 수 있으므로,
    # 값이 있을 때만 캐시를 신뢰합니다.
    if _PRODUCT_DETAIL_MAP_CACHE:
        return _PRODUCT_DETAIL_MAP_CACHE

    if get_product_detail_map is None:
        return {}

    try:
        raw_map = get_product_detail_map()
    except Exception:
        return {}

    if not isinstance(raw_map, dict):
        return {}

    normalized_map: dict[str, dict[str, Any]] = {}

    for key, value in raw_map.items():
        if not isinstance(value, dict):
            continue

        product_name = value.get("product_name") or key
        lookup_keys = [
            key,
            product_name,
            value.get("rag_document_key"),
            value.get("source_file"),
            value.get("file_name"),
        ]

        for lookup_key in lookup_keys:
            normalized_key = _normalize_product_lookup_name(lookup_key)
            if normalized_key:
                normalized_map[normalized_key] = value

    _PRODUCT_DETAIL_MAP_CACHE = normalized_map
    return _PRODUCT_DETAIL_MAP_CACHE


def _find_db_product_detail_for_candidate(product: dict[str, Any]) -> dict[str, Any] | None:
    detail_map = _get_product_detail_map_cached()
    if not detail_map:
        return None

    lookup_values = [
        product.get("product_name"),
        product.get("product_key"),
        product.get("rag_document_key"),
        product.get("source_file"),
        product.get("file_name"),
        product.get("document_name"),
    ]

    for value in lookup_values:
        normalized_value = _normalize_product_lookup_name(value)
        if normalized_value and isinstance(detail_map.get(normalized_value), dict):
            return detail_map[normalized_value]

    candidate_name = _normalize_product_lookup_name(product.get("product_name"))
    if candidate_name:
        for key, detail in detail_map.items():
            if not isinstance(detail, dict):
                continue

            db_name = _normalize_product_lookup_name(detail.get("product_name") or key)
            if db_name and (candidate_name in db_name or db_name in candidate_name):
                return detail

    return None


def _apply_product_field_fallbacks(product: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(product or {})

    for aliases, values in _PRODUCT_FIELD_FALLBACKS:
        blob = _product_lookup_blob(enriched)
        if not any(_normalize_product_lookup_name(alias) in blob for alias in aliases):
            continue

        for key, fallback_value in values.items():
            current_value = enriched.get(key)

            if key == "max_amount":
                current_amount = _to_int_or_none(current_value)
                fallback_amount = _to_int_or_none(fallback_value)
                if fallback_amount is not None and (
                    current_amount is None or current_amount > fallback_amount
                ):
                    enriched[key] = fallback_amount
                continue

            if current_value in (None, "", []):
                enriched[key] = fallback_value

    return enriched


def _enrich_product_with_db_fields(product: dict[str, Any]) -> dict[str, Any]:
    """
    financial 계산에 필요한 기간/금리/금액 정보를 DB products 기준으로 보강합니다.
    DB 매칭이 실패해도 주요 상품의 안전 fallback을 적용합니다.
    """
    enriched = dict(product or {})
    db_detail = _find_db_product_detail_for_candidate(enriched)

    if isinstance(db_detail, dict):
        db_first_keys = [
            "product_id",
            "product_name",
            "product_type",
            "min_amount",
            "max_amount",
            "min_period_months",
            "max_period_months",
            "base_rate",
            "max_rate",
            "age_min",
            "age_max",
            "join_channel",
            "rag_document_key",
            "is_active",
        ]

        for key in db_first_keys:
            value = db_detail.get(key)
            if value not in (None, ""):
                enriched[key] = value

    enriched = _apply_product_field_fallbacks(enriched)
    return enriched


def _load_eligibility_status_by_product(state: AgentState) -> dict[str, dict[str, Any]]:
    """eligibility_agent 결과를 상품명 기준으로 조회할 수 있게 만듭니다."""
    results = state.get("eligibility_results") or []

    if not results:
        eligibility_result = state.get("eligibility_result") or {}
        payload = eligibility_result.get("result") if isinstance(eligibility_result, dict) else {}
        if isinstance(payload, dict):
            results = payload.get("results") or []

    mapping: dict[str, dict[str, Any]] = {}
    for item in results:
        if not isinstance(item, dict):
            continue

        raw_names = [
            item.get("product_name"),
            item.get("name"),
            item.get("product"),
        ]

        for raw_name in raw_names:
            name = str(raw_name or "").strip()
            if not name:
                continue

            mapping[name] = item

            normalized_name = _normalize_product_lookup_name(name)
            if normalized_name:
                mapping[normalized_name] = item

    return mapping


def _get_eligibility_item_by_product_name(
    eligibility_by_name: dict[str, dict[str, Any]],
    product_name: str,
) -> dict[str, Any] | None:
    if not isinstance(eligibility_by_name, dict):
        return None

    if product_name in eligibility_by_name:
        return eligibility_by_name[product_name]

    normalized_name = _normalize_product_lookup_name(product_name)
    if normalized_name in eligibility_by_name:
        return eligibility_by_name[normalized_name]

    for key, item in eligibility_by_name.items():
        normalized_key = _normalize_product_lookup_name(key)
        if normalized_name and normalized_key and (
            normalized_name in normalized_key or normalized_key in normalized_name
        ):
            return item

    return None


def _ensure_financial_calculations(result: dict) -> dict:
    """
    기존 tool_results 기반 계산 파싱은 유지하되,
    recommendation augmentation에서 만든 financial_results/calculations를 덮어 지우지 않습니다.
    """
    financial_result = result.get("financial_result")
    if not isinstance(financial_result, dict):
        return result

    payload = financial_result.get("result", {})
    if not isinstance(payload, dict):
        return result

    tool_results = payload.get("tool_results") or []
    parsed_calculations = _parse_financial_calculations(tool_results)

    existing_calculations = payload.get("calculations")
    if not isinstance(existing_calculations, list):
        existing_calculations = []

    financial_results = payload.get("financial_results")
    if not isinstance(financial_results, list):
        financial_results = result.get("financial_results")
    if not isinstance(financial_results, list):
        financial_results = []

    recommendation_calculations = [
        item for item in financial_results
        if isinstance(item, dict) and item.get("status") == "calculated"
    ]

    calculations = (
        parsed_calculations
        or recommendation_calculations
        or existing_calculations
    )

    payload["calculations"] = calculations

    if calculations:
        if financial_results:
            payload["missing_fields"] = []
            payload["fallback_reason"] = None
            if payload.get("status") in (None, "needs_check"):
                payload["status"] = "success"
            if financial_result.get("status") == "needs_check":
                financial_result["status"] = "success"
    else:
        payload["missing_fields"] = payload.get("missing_fields") or ["term_months", "applied_rate"]
        payload["fallback_reason"] = payload.get("fallback_reason") or "missing_required_calculation_fields"
        if payload.get("status") == "success":
            payload["status"] = "needs_check"
        if financial_result.get("status") == "success":
            financial_result["status"] = "needs_check"
        if not payload.get("summary"):
            payload["summary"] = "계산에 필요한 필수 정보가 부족하여 financial_result.calculations를 생성하지 못했습니다."

    financial_result["result"] = payload
    result["financial_result"] = financial_result

    agent_outputs = dict(result.get("agent_outputs") or {})
    if isinstance(agent_outputs.get("financial_agent"), dict):
        agent_outputs["financial_agent"]["result"] = payload
    result["agent_outputs"] = agent_outputs

    return result

