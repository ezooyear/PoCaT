"""
Eligibility agent.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import AIMessage

from agents.base import make_agent_result
from agents.eligibility.prompts import ELIGIBILITY_SYSTEM_PROMPT
from agents.eligibility.tools import (
    build_eligibility_summary,
    evaluate_product_eligibility,
    extract_product_candidates,
    parse_customer_accounts,
    parse_customer_profile,
)
from graph.state import AgentState
from observability.langfuse import langfuse_observation, update_observation


REQUIRED_PROFILE_FIELDS = {
    "age": "age",
    "job": "job",
    "monthly_saving_amount": "monthly_saving_amount",
    "income": "income",
    "salary_transfer": "salary_transfer",
    "auto_transfer": "auto_transfer",
}

INVALID_PRODUCT_MARKERS = [
    "적용조건",
    "우대이율",
    "우대금리",
    "신규가입일",
    "영업점",
    "가입방법",
    "유의사항",
    "판매기간",
]


def eligibility_agent_node(state: AgentState) -> dict:
    with langfuse_observation(
        name="eligibility_agent",
        as_type="span",
        input=_build_trace_input(state, "eligibility_agent"),
        metadata={"agent": "eligibility_agent"},
    ) as observation:
        try:
            agent_outputs = dict(state.get("agent_outputs") or {})
            customer_result = state.get("customer_result") or {}
            product_result = state.get("product_result") or {}

            customer_profile = _load_customer_profile(state, customer_result, agent_outputs)
            customer_accounts = _load_customer_accounts(state, customer_result, agent_outputs)
            product_candidates = _load_product_candidates(state, product_result, agent_outputs)
            user_constraints = _extract_user_constraints(state)

            with langfuse_observation(
                name="eligibility_agent.prepare_inputs",
                as_type="span",
                input={
                    "customer_profile": customer_profile,
                    "customer_accounts_count": len(customer_accounts or []),
                    "product_candidate_count": len(product_candidates or []),
                    "user_constraints": user_constraints,
                },
                metadata={"agent": "eligibility_agent", "step": "prepare_inputs"},
            ) as step_observation:
                update_observation(
                    step_observation,
                    output={
                        "customer_profile": customer_profile,
                        "customer_accounts_preview": customer_accounts[:5] if isinstance(customer_accounts, list) else customer_accounts,
                        "product_candidates_preview": product_candidates[:5] if isinstance(product_candidates, list) else product_candidates,
                    },
                    metadata={"agent": "eligibility_agent"},
                )

            with langfuse_observation(
                name="eligibility_agent.evaluate",
                as_type="span",
                input={"product_candidate_count": len(product_candidates or [])},
                metadata={"agent": "eligibility_agent", "step": "evaluate"},
            ) as evaluation_observation:
                results, fallback_notes = _build_guarded_eligibility_results(
                    customer_profile=customer_profile,
                    customer_accounts=customer_accounts,
                    product_candidates=product_candidates,
                    user_constraints=user_constraints,
                )
                update_observation(
                    evaluation_observation,
                    output={
                        "result_count": len(results),
                        "fallback_notes": fallback_notes,
                        "product_names": [
                            item.get("product_name")
                            for item in results[:10]
                            if isinstance(item, dict)
                        ],
                    },
                    metadata={"agent": "eligibility_agent"},
                )

            summary = _build_guarded_summary(results, fallback_notes)
            grouped = _group_eligibility_results(results)
            overall_status = _resolve_eligibility_overall_status(results, fallback_notes)
            fallback_reason = "; ".join(fallback_notes) if fallback_notes else None
            missing_fields = _collect_unique_items(results, "missing_fields")
            invalid_fields = _collect_invalid_fields(results)

            eligibility_result = make_agent_result(
                status="success",
                result={
                    "status": overall_status,
                    "summary": summary,
                    "results": results,
                    "eligible_products": grouped["eligible"],
                    "needs_check_products": grouped["needs_check"],
                    "rejected_products": grouped["rejected"],
                    "invalid_products": grouped["invalid_product"],
                    "result_count": len(results),
                    "recommendable_count": len(grouped["eligible"]),
                    "needs_check_count": len(grouped["needs_check"]),
                    "rejected_count": len(grouped["rejected"]),
                    "invalid_product_count": len(grouped["invalid_product"]),
                    "customer_profile": customer_profile,
                    "customer_accounts": customer_accounts,
                    "product_candidates": product_candidates,
                    "fallback_reason": fallback_reason,
                    "missing_fields": missing_fields,
                    "invalid_fields": invalid_fields,
                    "source_agent": "eligibility_agent",
                    "user_constraints": user_constraints,
                },
                evidence=results,
                error=None,
            )
            agent_outputs["eligibility_agent"] = eligibility_result

            update_observation(
                observation,
                output=_build_trace_output(
                    eligibility_result,
                    extra_output={
                        "fallback_reason": fallback_reason,
                        "missing_fields": missing_fields,
                        "invalid_fields": invalid_fields,
                        "eligible_product_names": [
                            item.get("product_name")
                            for item in grouped["eligible"][:10]
                            if isinstance(item, dict)
                        ],
                        "needs_check_product_names": [
                            item.get("product_name")
                            for item in grouped["needs_check"][:10]
                            if isinstance(item, dict)
                        ],
                        "rejected_product_names": [
                            item.get("product_name")
                            for item in grouped["rejected"][:10]
                            if isinstance(item, dict)
                        ],
                        "invalid_product_names": [
                            item.get("product_name")
                            for item in grouped["invalid_product"][:10]
                            if isinstance(item, dict)
                        ],
                    },
                ),
                metadata={
                    "agent": "eligibility_agent",
                    "status": overall_status,
                    "fallback_reason": fallback_reason,
                    "missing_fields": ",".join(missing_fields[:10]) if missing_fields else None,
                    "invalid_fields": ",".join(invalid_fields[:10]) if invalid_fields else None,
                },
            )

            completed_agents = list(state.get("completed_agents") or [])
            if "eligibility_agent" not in completed_agents:
                completed_agents.append("eligibility_agent")

            return {
                "messages": [AIMessage(content=summary)],
                "agent_outputs": agent_outputs,
                "current_step": (state.get("current_step") or 0) + 1,
                "current_agent": "eligibility_agent",
                "completed_agents": completed_agents,
                "customer_profile": customer_profile,
                "customer_accounts": customer_accounts,
                "product_candidates": product_candidates,
                "eligibility_results": results,
                "eligibility_result": eligibility_result,
                "context": {
                    **(state.get("context") or {}),
                    "eligibility_prompt": ELIGIBILITY_SYSTEM_PROMPT,
                },
            }
        except Exception as error:
            fallback = _build_eligibility_exception_fallback(state, str(error))
            update_observation(
                observation,
                output=_build_trace_output(
                    fallback["eligibility_result"],
                    extra_output={
                        "fallback_reason": fallback["eligibility_result"]["result"].get("fallback_reason"),
                        "missing_fields": fallback["eligibility_result"]["result"].get("missing_fields"),
                        "invalid_fields": fallback["eligibility_result"]["result"].get("invalid_fields"),
                        "error_message": str(error),
                    },
                ),
                metadata={
                    "agent": "eligibility_agent",
                    "status": fallback["eligibility_result"]["result"].get("status"),
                    "fallback_reason": fallback["eligibility_result"]["result"].get("fallback_reason"),
                    "error_message": str(error),
                },
            )
            return fallback


def _load_customer_profile(state: AgentState, customer_result: dict, agent_outputs: dict) -> dict:
    customer_profile = state.get("customer_profile")
    if customer_profile:
        return _enrich_customer_profile(customer_profile)

    customer_profile_raw = _extract_tool_result(customer_result, "get_customer_profile")
    if not customer_profile_raw:
        customer_profile_raw = _extract_summary(agent_outputs.get("customer_agent"))
    return _enrich_customer_profile(parse_customer_profile(customer_profile_raw))


def _load_customer_accounts(state: AgentState, customer_result: dict, agent_outputs: dict) -> list[dict]:
    customer_accounts = state.get("customer_accounts")
    if customer_accounts is None:
        customer_accounts = []
    if customer_accounts:
        return customer_accounts

    customer_accounts_raw = _extract_tool_result(customer_result, "get_customer_accounts")
    if not customer_accounts_raw:
        customer_accounts_raw = _extract_summary(agent_outputs.get("customer_agent"))
    return parse_customer_accounts(customer_accounts_raw)


def _load_product_candidates(state: AgentState, product_result: dict, agent_outputs: dict) -> list[dict]:
    product_candidates = state.get("product_candidates")
    if not product_candidates:
        product_candidates = _extract_product_candidates_from_result(product_result)
    if not product_candidates:
        product_candidates = extract_product_candidates(_extract_summary(agent_outputs.get("product_agent")))
    else:
        product_candidates = extract_product_candidates(product_candidates)
    return product_candidates


def _build_guarded_eligibility_results(
    *,
    customer_profile: dict,
    customer_accounts: list[dict],
    product_candidates: list[dict],
    user_constraints: dict[str, Any],
) -> tuple[list[dict], list[str]]:
    """
    입력이 부실한 경우에도 agent가 죽지 않도록 방어형 eligibility 결과를 만든다.

    과한 확정 판단을 피하고, 필요한 경우 needs_check 또는 invalid_product로 낮춰서 반환한다.
    """

    results: list[dict] = []
    fallback_notes: list[str] = []
    profile_missing_fields = _find_missing_profile_fields(customer_profile)

    if not product_candidates:
        fallback_notes.append("product_candidates_missing")
        results.append(
            _make_guarded_result(
                product_name="상품 정보 확인 필요",
                status="needs_check",
                eligible=False,
                reasons=["상품 후보 정보가 없어 가입 가능 여부를 판단할 수 없습니다."],
                missing_fields=["product_candidates"],
                source_agent="eligibility_agent",
            )
        )
        return results, fallback_notes

    for product in product_candidates:
        product_name = str(product.get("product_name") or "미확인 상품").strip()
        invalid_reasons = _validate_product_name(product_name)

        # 문서 본문이 상품명으로 잘못 들어오면 이후 추천까지 오염될 수 있으므로
        # 초기에 invalid_product로 막아서 downstream 오류를 줄입니다.
        if invalid_reasons:
            fallback_notes.append("invalid_product_name")
            results.append(
                _make_guarded_result(
                    product_name=product_name,
                    status="invalid_product",
                    eligible=False,
                    reasons=invalid_reasons,
                    missing_fields=[],
                    invalid_fields=["product_name"],
                    source_agent="eligibility_agent",
                )
            )
            continue

        base_result = evaluate_product_eligibility(customer_profile, customer_accounts, product)
        guarded_result = _normalize_eligibility_result(base_result)
        guarded_result["source_agent"] = "eligibility_agent"

        if profile_missing_fields and guarded_result["status"] == "eligible":
            fallback_notes.append("customer_profile_incomplete")
            # 고객 핵심 정보가 없으면 eligible=true가 downstream 추천으로 이어질 수 있으므로
            # 확정 가능 대신 needs_check로 낮춰서 반환합니다.
            guarded_result["eligible"] = False
            guarded_result["status"] = "needs_check"
            guarded_result["missing_fields"] = _merge_unique_lists(
                guarded_result.get("missing_fields", []),
                profile_missing_fields,
            )
            guarded_result["reasons"] = _merge_unique_lists(
                guarded_result.get("reasons", []),
                ["고객 핵심 정보가 부족하여 가입 가능 여부를 확정할 수 없습니다."],
            )

        condition_notes = _apply_user_constraints(guarded_result, product, user_constraints)
        if condition_notes:
            fallback_notes.extend(condition_notes)

        requirement_missing_fields = _find_missing_product_requirements(product)
        if guarded_result["status"] == "eligible" and requirement_missing_fields:
            fallback_notes.append("product_requirement_missing")
            # 상품 가입조건을 확인할 수 없는데 eligible로 두면 잘못된 가입 가능 판단이 되므로
            # 확인 가능한 정보가 모일 때까지 needs_check로 유보합니다.
            guarded_result["eligible"] = False
            guarded_result["status"] = "needs_check"
            guarded_result["missing_fields"] = _merge_unique_lists(
                guarded_result.get("missing_fields", []),
                requirement_missing_fields,
            )
            guarded_result["reasons"] = _merge_unique_lists(
                guarded_result.get("reasons", []),
                ["상품 가입조건을 충분히 확인할 수 없어 추가 확인이 필요합니다."],
            )

        results.append(guarded_result)

    return results, _dedupe(fallback_notes)


def _normalize_eligibility_result(result: dict[str, Any]) -> dict[str, Any]:
    product_name = str(result.get("product_name") or "미확인 상품").strip()
    reasons = list(result.get("ineligibility_reasons") or [])
    check_required = list(result.get("check_required") or [])

    if reasons:
        status = "rejected"
        eligible = False
    elif check_required:
        status = "needs_check"
        eligible = False
    else:
        status = "eligible"
        eligible = bool(result.get("eligible"))

    return {
        "product_name": product_name,
        "eligible": eligible,
        "status": status,
        "reasons": reasons,
        "missing_fields": check_required,
        "invalid_fields": [],
        "source_agent": "eligibility_agent",
        "bonus_conditions_met": list(result.get("bonus_conditions_met") or []),
        "bonus_conditions_missing": list(result.get("bonus_conditions_missing") or []),
        "check_required": check_required,
        "ineligibility_reasons": reasons,
    }


def _extract_user_constraints(state: AgentState) -> dict[str, Any]:
    """
    사용자 발화에서 납입금액과 기간 같은 명시 조건을 추출한다.

    앞단 입력이 불완전하더라도 사용자가 직접 말한 조건이 있으면
    잘못된 eligible 판정을 줄이는 데 도움이 된다.
    """

    user_query = str(state.get("user_query") or _get_last_user_text(state.get("messages") or []))
    normalized = user_query.replace(",", "").replace(" ", "")

    monthly_amount = None
    amount_match = re.search(r"월([0-9]+)만원", normalized)
    if amount_match:
        monthly_amount = int(amount_match.group(1)) * 10000
    else:
        amount_match = re.search(r"월([0-9]+)원", normalized)
        if amount_match:
            monthly_amount = int(amount_match.group(1))

    period_months = None
    period_match = re.search(r"([0-9]{1,2})개월", normalized)
    if period_match:
        period_months = int(period_match.group(1))

    return {
        "monthly_amount": monthly_amount,
        "period_months": period_months,
        "raw_query": user_query,
    }


def _apply_user_constraints(result: dict[str, Any], product: dict, user_constraints: dict[str, Any]) -> list[str]:
    fallback_notes: list[str] = []
    monthly_amount = user_constraints.get("monthly_amount")
    period_months = user_constraints.get("period_months")

    min_amount, max_amount = _extract_amount_bounds(product)
    period_min, period_max = _extract_period_bounds(product)

    if monthly_amount is not None:
        if max_amount is not None and monthly_amount > max_amount:
            fallback_notes.append("monthly_amount_exceeds_limit")
            result["eligible"] = False
            result["status"] = "rejected"
            result["reasons"] = _merge_unique_lists(
                result.get("reasons", []),
                [f"사용자 희망 납입금액이 상품 최대 한도 {max_amount:,}원을 초과합니다."],
            )
        elif min_amount is not None and monthly_amount < min_amount:
            fallback_notes.append("monthly_amount_below_limit")
            result["eligible"] = False
            result["status"] = "needs_check"
            result["reasons"] = _merge_unique_lists(
                result.get("reasons", []),
                [f"사용자 희망 납입금액이 상품 최소 한도 {min_amount:,}원보다 낮을 수 있습니다."],
            )

    if period_months is not None:
        if period_max is not None and period_months > period_max:
            fallback_notes.append("period_mismatch")
            result["eligible"] = False
            result["status"] = "rejected"
            result["reasons"] = _merge_unique_lists(
                result.get("reasons", []),
                [f"사용자 희망 기간 {period_months}개월이 상품 허용 기간을 초과합니다."],
            )
        elif period_min is not None and period_months < period_min:
            fallback_notes.append("period_mismatch")
            result["eligible"] = False
            result["status"] = "needs_check"
            result["reasons"] = _merge_unique_lists(
                result.get("reasons", []),
                [f"사용자 희망 기간 {period_months}개월이 상품 기준과 맞지 않을 수 있습니다."],
            )

    return fallback_notes


def _find_missing_profile_fields(customer_profile: dict[str, Any]) -> list[str]:
    missing_fields = []
    raw_text = str(customer_profile.get("raw_text") or "")

    for field_name, label in REQUIRED_PROFILE_FIELDS.items():
        value = customer_profile.get(field_name)

        # 파싱되지 않은 핵심 고객 정보는 eligibility 오판의 가장 큰 원인이므로
        # False/None/빈값을 엄격하게 missing으로 간주합니다.
        if field_name == "income":
            income_value = customer_profile.get("income")
            if income_value in (None, "", 0):
                missing_fields.append(label)
            continue

        if isinstance(value, bool):
            if value is False and field_name not in raw_text:
                missing_fields.append(label)
            continue

        if value in (None, "", 0):
            missing_fields.append(label)

    return _dedupe(missing_fields)


def _enrich_customer_profile(customer_profile: dict[str, Any]) -> dict[str, Any]:
    """
    파서가 놓친 핵심 고객 정보를 raw_text에서 한 번 더 보완한다.

    기존 파서를 크게 바꾸지 않으면서 정상 입력이 불필요하게 needs_check로 내려가는 것을 줄이기 위한 보정 단계다.
    """

    profile = dict(customer_profile or {})
    raw_text = str(profile.get("raw_text") or "")

    if profile.get("income") in (None, "", 0):
        income_match = re.search(r"(소득|월소득|연소득)[:\s]*([0-9][0-9,]*)", raw_text)
        if income_match:
            profile["income"] = int(income_match.group(2).replace(",", ""))

    if profile.get("salary_transfer") is None:
        profile["salary_transfer"] = "급여이체" in raw_text

    if profile.get("auto_transfer") is None:
        profile["auto_transfer"] = "자동이체" in raw_text

    return profile


def _find_missing_product_requirements(product: dict[str, Any]) -> list[str]:
    raw_text = str(product.get("raw_text") or "")
    lowered = raw_text.lower()
    missing_fields = []

    if not any(token in raw_text for token in ["가입대상", "가입 대상", "군인", "직업", "전용"]):
        missing_fields.append("product_target")
    if not any(token in raw_text for token in ["만", "세", "연령", "나이"]):
        missing_fields.append("product_age_condition")
    if not any(token in lowered for token in ["직업", "군인", "대상", "전용", "가입 가능"]):
        missing_fields.append("product_job_condition")

    return _dedupe(missing_fields)


def _validate_product_name(product_name: str) -> list[str]:
    """
    상품명이 실제 상품명인지 간단히 검증한다.

    문서 본문 일부가 상품명으로 잘못 추출되는 경우를 방지하기 위한 함수.
    """

    stripped = str(product_name or "").strip()
    reasons = []

    if not stripped or stripped == "미확인 상품":
        reasons.append("상품명을 확인할 수 없습니다.")
        return reasons

    if len(stripped) > 40:
        reasons.append("상품명이 비정상적으로 길어 문서 본문이 섞였을 가능성이 있습니다.")

    if any(marker in stripped for marker in INVALID_PRODUCT_MARKERS):
        reasons.append("상품명 대신 문서 본문 또는 안내 문구가 추출된 것으로 보입니다.")

    if len(stripped.split()) > 6:
        reasons.append("상품명에 불필요한 설명 문장이 포함된 것으로 보입니다.")

    return reasons


def _extract_amount_bounds(product: dict[str, Any]) -> tuple[int | None, int | None]:
    min_amount = product.get("min_amount")
    max_amount = product.get("max_amount")
    raw_text = str(product.get("raw_text") or "")

    if min_amount is None:
        min_match = re.search(r"(최소|최저).{0,10}?([0-9][0-9,]*)", raw_text)
        if min_match:
            min_amount = int(min_match.group(2).replace(",", ""))

    if max_amount is None:
        max_match = re.search(r"(최대|최고).{0,10}?([0-9][0-9,]*)", raw_text)
        if max_match:
            max_amount = int(max_match.group(2).replace(",", ""))

    return min_amount, max_amount


def _extract_period_bounds(product: dict[str, Any]) -> tuple[int | None, int | None]:
    raw_text = str(product.get("raw_text") or "")
    months = [int(match) for match in re.findall(r"([0-9]{1,2})\s*개월", raw_text)]
    if not months:
        return None, None
    return min(months), max(months)


def _group_eligibility_results(results: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped = {
        "eligible": [],
        "needs_check": [],
        "rejected": [],
        "invalid_product": [],
    }

    for item in results:
        status = item.get("status", "needs_check")
        if status not in grouped:
            status = "needs_check"
        grouped[status].append(item)

    return grouped


def _resolve_eligibility_overall_status(results: list[dict[str, Any]], fallback_notes: list[str]) -> str:
    if not results:
        return "needs_check"
    if any(item.get("status") == "eligible" for item in results):
        return "eligible"
    if any(item.get("status") == "needs_check" for item in results):
        return "needs_check"
    if any(item.get("status") == "invalid_product" for item in results):
        return "invalid_product"
    if fallback_notes:
        return "needs_check"
    return "rejected"


def _build_guarded_summary(results: list[dict[str, Any]], fallback_notes: list[str]) -> str:
    base_summary = build_eligibility_summary(
        [
            {
                "product_name": item.get("product_name"),
                "eligible": item.get("status") == "eligible",
                "ineligibility_reasons": item.get("reasons", []),
                "bonus_conditions_met": item.get("bonus_conditions_met", []),
                "bonus_conditions_missing": item.get("bonus_conditions_missing", []),
                "check_required": item.get("missing_fields", []),
            }
            for item in results
        ]
    )

    if not fallback_notes:
        return base_summary

    return f"{base_summary}\n\n추가 확인 필요 사유: {', '.join(_dedupe(fallback_notes))}"


def _collect_unique_items(results: list[dict[str, Any]], key: str) -> list[str]:
    collected: list[str] = []
    for item in results:
        if isinstance(item.get(key), list):
            collected.extend(str(value) for value in item.get(key, []) if value)
    return _dedupe(collected)


def _collect_invalid_fields(results: list[dict[str, Any]]) -> list[str]:
    return _collect_unique_items(results, "invalid_fields")


def _build_eligibility_exception_fallback(state: AgentState, error_message: str) -> dict[str, Any]:
    """
    예외가 발생해도 그래프 전체가 죽지 않도록 eligibility fallback 결과를 만든다.
    """

    fallback_result = make_agent_result(
        status="failed",
        result={
            "status": "needs_check",
            "summary": "가입 가능 여부를 계산하는 중 오류가 발생해 추가 확인이 필요합니다.",
            "results": [
                _make_guarded_result(
                    product_name="상품 정보 확인 필요",
                    status="needs_check",
                    eligible=False,
                    reasons=["Eligibility 계산 중 오류가 발생했습니다."],
                    missing_fields=["customer_profile", "product_candidates"],
                    source_agent="eligibility_agent",
                )
            ],
            "eligible_products": [],
            "needs_check_products": [],
            "rejected_products": [],
            "invalid_products": [],
            "result_count": 1,
            "recommendable_count": 0,
            "needs_check_count": 1,
            "rejected_count": 0,
            "invalid_product_count": 0,
            "customer_profile": state.get("customer_profile"),
            "customer_accounts": state.get("customer_accounts"),
            "product_candidates": state.get("product_candidates") or [],
            "fallback_reason": "exception_in_eligibility_agent",
            "missing_fields": ["customer_profile", "product_candidates"],
            "invalid_fields": [],
            "source_agent": "eligibility_agent",
            "user_constraints": _extract_user_constraints(state),
        },
        evidence=[],
        error=error_message,
    )

    agent_outputs = dict(state.get("agent_outputs") or {})
    agent_outputs["eligibility_agent"] = fallback_result

    completed_agents = list(state.get("completed_agents") or [])
    if "eligibility_agent" not in completed_agents:
        completed_agents.append("eligibility_agent")

    return {
        "messages": [AIMessage(content=fallback_result["result"]["summary"])],
        "agent_outputs": agent_outputs,
        "current_step": (state.get("current_step") or 0) + 1,
        "current_agent": "eligibility_agent",
        "completed_agents": completed_agents,
        "customer_profile": state.get("customer_profile"),
        "customer_accounts": state.get("customer_accounts") or [],
        "product_candidates": state.get("product_candidates") or [],
        "eligibility_results": fallback_result["result"]["results"],
        "eligibility_result": fallback_result,
        "context": {
            **(state.get("context") or {}),
            "eligibility_prompt": ELIGIBILITY_SYSTEM_PROMPT,
        },
    }


def _make_guarded_result(
    *,
    product_name: str,
    status: str,
    eligible: bool,
    reasons: list[str],
    missing_fields: list[str],
    source_agent: str,
    invalid_fields: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "product_name": product_name,
        "eligible": eligible,
        "status": status,
        "reasons": _dedupe(reasons),
        "missing_fields": _dedupe(missing_fields),
        "invalid_fields": _dedupe(invalid_fields or []),
        "source_agent": source_agent,
        "bonus_conditions_met": [],
        "bonus_conditions_missing": [],
        "check_required": _dedupe(missing_fields),
        "ineligibility_reasons": _dedupe(reasons),
    }


def _merge_unique_lists(existing: list[str], new_items: list[str]) -> list[str]:
    return _dedupe(list(existing or []) + list(new_items or []))


def _extract_summary(value) -> str:
    if isinstance(value, dict):
        result = value.get("result")
        if isinstance(result, dict) and isinstance(result.get("summary"), str):
            return result["summary"]
        if isinstance(value.get("summary"), str):
            return value["summary"]
    if isinstance(value, str):
        return value
    return ""


def _extract_tool_result(result_container, tool_name: str) -> str:
    if not isinstance(result_container, dict):
        return ""
    result = result_container.get("result", {})
    if not isinstance(result, dict):
        return ""
    tool_results = result.get("tool_results", [])
    if not isinstance(tool_results, list):
        return ""
    for item in tool_results:
        if not isinstance(item, dict):
            continue
        if item.get("tool_name") == tool_name and isinstance(item.get("tool_result"), str):
            return item["tool_result"]
    return ""


def _extract_product_candidates_from_result(product_result) -> list[dict]:
    if not isinstance(product_result, dict):
        return []
    result = product_result.get("result", {})
    if not isinstance(result, dict):
        return []
    product_candidates = result.get("product_candidates", [])
    if isinstance(product_candidates, list):
        return product_candidates
    return []


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    output = []
    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output


def _get_last_user_text(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, tuple) and len(msg) >= 2:
            role, content = msg[0], msg[1]
            if role in ["user", "human"]:
                return str(content)

        if hasattr(msg, "type") and hasattr(msg, "content"):
            if msg.type in ["human", "user"]:
                return str(msg.content)

    return ""


def _build_trace_input(state: AgentState, agent_name: str) -> dict[str, Any]:
    return {
        "agent_name": agent_name,
        "user_query": state.get("user_query"),
        "task_type": state.get("task_type"),
        "current_step": state.get("current_step"),
        "plan": state.get("plan"),
        "completed_agents": state.get("completed_agents"),
        "message_count": len(state.get("messages") or []),
    }


def _build_trace_output(
    agent_result: dict[str, Any],
    *,
    extra_output: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result_payload = agent_result.get("result", {}) if isinstance(agent_result, dict) else {}

    payload: dict[str, Any] = {
        "status": agent_result.get("status") if isinstance(agent_result, dict) else None,
        "error": agent_result.get("error") if isinstance(agent_result, dict) else None,
        "summary": result_payload.get("summary") if isinstance(result_payload, dict) else None,
        "result": result_payload,
        "evidence_preview": agent_result.get("evidence") if isinstance(agent_result, dict) else None,
    }

    if extra_output:
        payload.update(extra_output)

    return payload
