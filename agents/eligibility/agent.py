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
    _extract_age_from_birth_date,
    build_eligibility_summary,
    evaluate_product_eligibility,
    extract_product_candidates,
    _infer_is_soldier,
    parse_customer_accounts,
    parse_customer_profile,
)
from graph.state import AgentState
from observability.langfuse import langfuse_observation, update_observation

try:
    from agents.product.tools import get_product_detail_map
except Exception:
    get_product_detail_map = None

REQUIRED_PROFILE_FIELDS = {
    "age": "고객 나이 정보",
    "job": "군 복무/직업 정보",
    "monthly_saving_amount": "고객 월 가용 저축액",
    "income": "고객 소득 정보",
    "salary_transfer": "급여이체 상태",
    "auto_transfer": "자동이체 상태",
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
def _merge_user_constraints_into_customer_profile(
    customer_profile: dict[str, Any],
    user_constraints: dict[str, Any],
) -> dict[str, Any]:
    """
    사용자가 '월 30만원'처럼 직접 말한 금액은
    customer_profile.monthly_saving_amount가 비어 있을 때 보조값으로 사용합니다.
    단, transaction_months 같은 고객 거래개월 수는 계약기간으로 절대 사용하지 않습니다.
    """
    profile = dict(customer_profile or {})
    parsed_fields = set(profile.get("parsed_customer_fields") or [])
    parsed_values = dict(profile.get("parsed_customer_values") or {})

    monthly_amount = user_constraints.get("monthly_amount")
    if monthly_amount is not None and profile.get("monthly_saving_amount") in (None, "", 0):
        profile["monthly_saving_amount"] = monthly_amount
        parsed_values["monthly_saving_amount"] = monthly_amount

    profile["parsed_customer_fields"] = list(parsed_fields)
    profile["parsed_customer_values"] = parsed_values
    return profile


async def eligibility_agent_node(state: AgentState) -> dict:
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
            customer_profile = _merge_user_constraints_into_customer_profile(
                customer_profile,
                user_constraints,
            )

            
            customer_profile_source = customer_profile.get("customer_profile_source", "missing")
            parsed_customer_fields = customer_profile.get("parsed_customer_fields", [])
            parsed_customer_values = customer_profile.get("parsed_customer_values", {})

            with langfuse_observation(
                name="eligibility_agent.prepare_inputs",
                as_type="span",
                input={
                    "customer_profile": customer_profile,
                    "customer_accounts_count": len(customer_accounts or []),
                    "product_candidate_count": len(product_candidates or []),
                    "user_constraints": user_constraints,
                    "customer_profile_source": customer_profile_source,
                    "parsed_customer_values": parsed_customer_values,
                },
                metadata={"agent": "eligibility_agent", "step": "prepare_inputs"},
            ) as step_observation:
                update_observation(
                    step_observation,
                    output={
                        "customer_profile": customer_profile,
                        "customer_accounts_preview": customer_accounts[:5] if isinstance(customer_accounts, list) else customer_accounts,
                        "product_candidates_preview": product_candidates[:5] if isinstance(product_candidates, list) else product_candidates,
                        "customer_profile_source": customer_profile_source,
                        "parsed_customer_fields": parsed_customer_fields,
                        "parsed_customer_values": parsed_customer_values,
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
                    "customer_profile_source": customer_profile_source,
                    "parsed_customer_fields": parsed_customer_fields,
                    "parsed_customer_values": parsed_customer_values,
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
                        "customer_profile_source": customer_profile_source,
                        "parsed_customer_fields": parsed_customer_fields,
                        "parsed_customer_values": parsed_customer_values,
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
                    "customer_profile_source": customer_profile_source,
                    "parsed_customer_fields": ",".join(parsed_customer_fields[:10]) if parsed_customer_fields else None,
                    "parsed_customer_values": str(parsed_customer_values)[:500] if parsed_customer_values else None,
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
        return _enrich_customer_profile(customer_profile, source="structured_state")

    customer_result_profile = _extract_customer_profile_from_result(customer_result)
    if customer_result_profile:
        return _enrich_customer_profile(customer_result_profile, source="customer_result")

    customer_agent_payload = agent_outputs.get("customer_agent")
    if isinstance(customer_agent_payload, dict):
        agent_output_profile = _extract_customer_profile_from_agent_output(customer_agent_payload)
        if agent_output_profile:
            return _enrich_customer_profile(agent_output_profile, source="agent_outputs")

    customer_texts = _collect_customer_profile_texts(customer_result, agent_outputs)
    if customer_texts:
        merged_text = "\n".join(text for text in customer_texts if text)
        source = "customer_result" if _has_customer_result_text(customer_result) else "parsed_summary"
        return _enrich_customer_profile(parse_customer_profile(merged_text), source=source)

    return _enrich_customer_profile({}, source="missing")


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


def _extract_customer_profile_from_result(customer_result: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(customer_result, dict):
        return None

    payload = customer_result.get("result", {})
    if isinstance(payload, dict):
        structured_profile = payload.get("customer_profile")
        if isinstance(structured_profile, dict):
            return structured_profile
    return None


def _extract_customer_profile_from_agent_output(agent_output: dict[str, Any]) -> dict[str, Any] | None:
    payload = agent_output.get("result", {})
    if isinstance(payload, dict):
        structured_profile = payload.get("customer_profile")
        if isinstance(structured_profile, dict):
            return structured_profile
    return None


def _collect_customer_profile_texts(customer_result: dict[str, Any], agent_outputs: dict[str, Any]) -> list[str]:
    texts: list[str] = []

    customer_profile_raw = _extract_tool_result(customer_result, "get_customer_profile")
    if customer_profile_raw:
        texts.append(str(customer_profile_raw))

    if isinstance(customer_result, dict):
        payload = customer_result.get("result", {})
        if isinstance(payload, dict):
            summary = payload.get("summary")
            if summary:
                texts.append(str(summary))

    customer_summary = _extract_summary(agent_outputs.get("customer_agent"))
    if customer_summary:
        texts.append(str(customer_summary))

    return _dedupe(texts)


def _has_customer_result_text(customer_result: dict[str, Any]) -> bool:
    if _extract_tool_result(customer_result, "get_customer_profile"):
        return True
    if isinstance(customer_result, dict):
        payload = customer_result.get("result", {})
        if isinstance(payload, dict) and payload.get("summary"):
            return True
    return False


def _load_product_candidates(state: AgentState, product_result: dict, agent_outputs: dict) -> list[dict]:
    product_candidates = state.get("product_candidates") or []

    if not product_candidates:
        if isinstance(product_result, dict):
            payload = product_result.get("result", {})
            if isinstance(payload, dict):
                candidate_list = payload.get("products") or payload.get("product_candidates")
                if isinstance(candidate_list, list):
                    product_candidates = [dict(item) for item in candidate_list if isinstance(item, dict)]

    if not product_candidates:
        if isinstance(agent_outputs.get("product_agent"), dict):
            payload = agent_outputs["product_agent"].get("result", {})
            if isinstance(payload, dict):
                candidate_list = payload.get("products") or payload.get("product_candidates")
                if isinstance(candidate_list, list):
                    product_candidates = [dict(item) for item in candidate_list if isinstance(item, dict)]

    if not product_candidates:
        product_candidates = extract_product_candidates(_extract_summary(agent_outputs.get("product_agent")))

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

    if profile_missing_fields:
        fallback_notes.append("customer_profile_incomplete")

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
        if not isinstance(product, dict):
            continue

        product = _enrich_product_with_db_fields(product)
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

        evaluation_profile, amount_cap_note = _prepare_profile_for_product_eligibility(
            customer_profile,
            product,
            user_constraints,
        )

        base_result = evaluate_product_eligibility(evaluation_profile, customer_accounts, product)
        guarded_result = _normalize_eligibility_result(base_result)
        guarded_result["source_agent"] = "eligibility_agent"

        if amount_cap_note:
            guarded_result["calculation_notes"] = _merge_unique_lists(
                guarded_result.get("calculation_notes", []),
                [amount_cap_note],
            )
        job_constraint_notes = _apply_job_specific_constraints(
            guarded_result,
            product,
            customer_profile,
        )
        if job_constraint_notes:
            fallback_notes.extend(job_constraint_notes)

        if profile_missing_fields and guarded_result["status"] in {"eligible", "needs_check"}:
            fallback_notes.append("customer_profile_incomplete")
            # 고객 핵심 정보가 없으면 eligible=true가 downstream 추천으로 이어질 수 있으므로
            # 확정 가능 대신 needs_check로 낮춰서 반환합니다.
            if guarded_result["status"] == "eligible":
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
    사용자 발화에서 납입금액과 기간 같은 명시 조건만 추출합니다.

    주의:
    - transaction_months, bank_transaction_months는 고객 거래기간이므로 계약기간으로 쓰지 않습니다.
    - customer_accounts.contract_months는 기존 가입계좌의 계약기간이므로 신규 희망기간으로 쓰지 않습니다.
    - 기간은 사용자가 '2년', '24개월'처럼 직접 말한 경우에만 추출합니다.
    """
    user_query = _get_explicit_user_query(state)
    normalized = user_query.replace(",", "").replace(" ", "")

    monthly_amount = None

    amount_match = re.search(r"월([0-9]+)만원", normalized)
    if amount_match:
        monthly_amount = int(amount_match.group(1)) * 10000
    else:
        amount_match = re.search(r"월([0-9]+)원", normalized)
        if amount_match:
            monthly_amount = int(amount_match.group(1))

    period_months = _extract_period_months_from_query(user_query)

    return {
        "monthly_amount": monthly_amount,
        "period_months": period_months,
        "raw_query": user_query,
    }


def _get_explicit_user_query(state: AgentState) -> str:
    """
    eligibility 조건 추출에는 실제 사용자 발화만 사용한다.

    state["user_query"]에 customer_profile, agent_outputs, debug text가 섞이면
    transaction_months=39 같은 고객 거래기간이 '39개월 희망기간'으로 오해될 수 있다.
    따라서 마지막 HumanMessage를 우선 사용하고, 불가피하게 state["user_query"]를 쓰더라도
    agent/debug/profile 라인은 제거한다.
    """
    last_user_text = _get_last_user_text(state.get("messages") or [])
    state_user_query = str(state.get("user_query") or "")

    raw_query = last_user_text or state_user_query
    return _strip_non_user_context(raw_query)


def _strip_non_user_context(text: str) -> str:
    """
    사용자 질문 문자열에 섞인 프로필/디버그/에이전트 결과 라인을 제거한다.
    정상적인 짧은 사용자 질문은 그대로 반환한다.
    """
    source = _normalize_summary_text(str(text or "")).strip()
    if not source:
        return ""

    # user_query: "..." 또는 사용자 질문: ... 형태가 있으면 해당 라인을 우선 사용한다.
    explicit_query_patterns = [
        r"(?:user_query|사용자\s*질문|질문)\s*[:=]\s*[\"']?([^\n\"']+)",
    ]
    for pattern in explicit_query_patterns:
        match = re.search(pattern, source, flags=re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            if candidate:
                return candidate

    blocked_markers = [
        "transaction_months",
        "bank_transaction_months",
        "contract_months",
        "remaining_months",
        "term_months",
        "customer_profile",
        "customer_accounts",
        "product_candidates",
        "agent_outputs",
        "customer_result",
        "product_result",
        "financial_result",
        "eligibility_result",
        "거래개월",
        "거래 개월",
        "거래기간",
        "거래 기간",
        "기존 가입",
        "가입계좌",
        "가입 계좌",
        "만기일",
        "maturity_date",
    ]

    cleaned_lines: list[str] = []
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if any(marker.lower() in lowered for marker in blocked_markers):
            continue
        cleaned_lines.append(stripped)

    if cleaned_lines:
        return " ".join(cleaned_lines).strip()

    return source


def _extract_period_months_from_query(user_query: str) -> int | None:
    """
    사용자 질문에 명시된 신규 가입 희망기간만 추출한다.

    고객 거래기간(transaction_months), 기존 계좌 계약기간(contract_months),
    남은 기간(remaining_months) 주변에서 발견된 '39개월' 같은 숫자는 무시한다.
    """
    normalized = str(user_query or "").replace(",", "")
    normalized = re.sub(r"\s+", "", normalized)
    if not normalized:
        return None

    for match in re.finditer(r"([0-9]{1,2})년", normalized):
        if _is_customer_or_account_period_context(normalized, match.start(), match.end()):
            continue
        return int(match.group(1)) * 12

    for match in re.finditer(r"([0-9]{1,2})개월", normalized):
        if _is_customer_or_account_period_context(normalized, match.start(), match.end()):
            continue
        return int(match.group(1))

    return None


def _is_customer_or_account_period_context(text: str, start: int, end: int) -> bool:
    """
    기간 숫자 주변에 고객 거래기간/기존 계좌기간을 의미하는 단어가 있으면
    사용자 희망기간으로 보지 않는다.
    """
    window = text[max(0, start - 30): min(len(text), end + 30)].lower()
    blocked_contexts = [
        "transaction_months",
        "bank_transaction_months",
        "contract_months",
        "remaining_months",
        "term_months",
        "거래개월",
        "거래기간",
        "은행거래",
        "고객거래",
        "기존계좌",
        "가입계좌",
        "만기까지",
        "남은기간",
    ]
    return any(marker.lower() in window for marker in blocked_contexts)

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


def _prepare_profile_for_product_eligibility(
    customer_profile: dict[str, Any],
    product: dict[str, Any],
    user_constraints: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
    """
    추천 흐름에서 DB의 월 저축 가능액은 '총 저축 가능액'이다.
    따라서 이 값이 상품별 월 납입한도보다 커도 가입 불가가 아니라,
    해당 상품 한도만큼 납입하는 것으로 판단한다.

    단, 사용자가 직접 '월 70만원 넣겠다'처럼 명시한 금액은 희망 납입액이므로
    _apply_user_constraints에서 별도로 한도 초과 여부를 판단한다.
    """
    profile = dict(customer_profile or {})

    # 사용자가 직접 월 납입액을 말한 경우에는 cap 하지 않는다.
    # 예: "월 70만원씩 넣고 싶어" → 상품 한도 초과 안내가 필요함.
    if user_constraints.get("monthly_amount") is not None:
        return profile, None

    customer_monthly_amount = _to_int_or_none(
        profile.get("monthly_saving_amount")
        or profile.get("available_monthly_saving")
    )

    if customer_monthly_amount is None:
        return profile, None

    _, product_max_amount = _extract_amount_bounds(product)

    if product_max_amount is None:
        return profile, None

    if customer_monthly_amount <= product_max_amount:
        return profile, None

    profile["available_monthly_saving"] = customer_monthly_amount
    profile["monthly_saving_amount"] = product_max_amount
    profile["monthly_amount_cap_applied"] = True
    profile["monthly_amount_before_cap"] = customer_monthly_amount
    profile["monthly_amount_after_cap"] = product_max_amount

    product_name = str(product.get("product_name") or "해당 상품").strip()
    note = (
        f"고객님의 월 저축 가능액은 {customer_monthly_amount:,}원이지만, "
        f"{product_name}의 월 납입한도가 {product_max_amount:,}원이므로 "
        f"가입 가능 여부는 월 {product_max_amount:,}원 기준으로 판단했습니다."
    )

    return profile, note

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


def _apply_job_specific_constraints(
    result: dict[str, Any],
    product: dict[str, Any],
    customer_profile: dict[str, Any],
) -> list[str]:
    """
    군인/장병/간부 등 특정 직군 대상 상품은 고객 직업과 반드시 대조합니다.

    evaluate_product_eligibility가 일반 조건만 통과시켜도,
    KB나라사랑적금(직업군인용), KB장병내일준비적금, 장기간부 적금처럼
    가입대상이 제한된 상품은 여기서 한 번 더 hard gate로 차단합니다.
    """
    if not _is_military_only_product(product):
        return []

    job = _clean_text_value(customer_profile.get("job"))
    target_label = _military_product_target_label(product)

    if not job:
        result["eligible"] = False
        result["status"] = "needs_check"
        result["missing_fields"] = _merge_unique_lists(
            result.get("missing_fields", []),
            ["job"],
        )
        result["check_required"] = _merge_unique_lists(
            result.get("check_required", []),
            ["job"],
        )
        result["reasons"] = _merge_unique_lists(
            result.get("reasons", []),
            [f"{target_label} 상품으로 보이나 고객 직업 정보가 없어 가입 가능 여부를 확정할 수 없습니다."],
        )
        return ["job_condition_needs_check"]

    if not _customer_is_soldier(customer_profile):
        reason = f"고객 직업이 '{job}'이므로 {target_label} 상품 가입 대상이 아닙니다."
        result["eligible"] = False
        result["status"] = "rejected"
        result["reasons"] = _merge_unique_lists(result.get("reasons", []), [reason])
        result["ineligibility_reasons"] = _merge_unique_lists(
            result.get("ineligibility_reasons", []),
            [reason],
        )
        return ["job_mismatch"]

    return []


def _is_military_only_product(product: dict[str, Any]) -> bool:
    """군인/장병/간부 등 특정 직군 대상 상품 여부를 상품명·DB key·RAG 본문으로 판정합니다."""
    text = _product_constraint_text(product)
    normalized = re.sub(r"\s+", "", text).lower()

    # 상품명 또는 문서키만으로도 특정 군 관련 상품임이 명확한 경우.
    # 여기에는 가입대상 문구가 raw_text에 없어도 hard gate를 적용해야 합니다.
    hard_keywords = [
        "kb나라사랑적금",
        "나라사랑적금",
        "직업군인용",
        "직업군인전용",
        "군인용",
        "군인전용",
        "kb장병내일준비적금",
        "장병내일준비적금",
        "장병내일준비",
        "장기간부적금",
        "장기간부",
        "장기복무",
        "군간부전용",
        "군간부",
    ]
    if any(keyword in normalized for keyword in hard_keywords):
        return True

    # 상품 설명에 가입대상/가입조건 표현과 군 관련 표현이 함께 있으면 군 관련 제한 상품으로 본다.
    target_markers = ["가입대상", "가입조건", "대상고객", "가입가능", "전용", "대상"]
    soldier_markers = ["직업군인", "군인", "장병", "장교", "부사관", "간부", "군간부"]
    return any(marker in text for marker in target_markers) and any(marker in text for marker in soldier_markers)


def _military_product_target_label(product: dict[str, Any]) -> str:
    """고객에게 보여줄 군 관련 상품 대상 표현을 상품명 기준으로 조금 더 정확히 만든다."""
    normalized = re.sub(r"\s+", "", _product_constraint_text(product)).lower()

    if "장기간부" in normalized or "군간부" in normalized or "장기복무" in normalized:
        return "군 간부 등 특정 직군 대상"
    if "장병내일준비" in normalized or "장병" in normalized:
        return "장병 등 특정 직군 대상"
    if "나라사랑" in normalized or "직업군인" in normalized:
        return "직업군인 등 특정 직군 대상"
    return "군인·장병·간부 등 특정 직군 대상"


def _product_constraint_text(product: dict[str, Any]) -> str:
    return " ".join(
        str(product.get(key) or "")
        for key in [
            "product_name",
            "product_target",
            "join_target",
            "eligibility",
            "rag_document_key",
            "raw_text",
        ]
    )


def _customer_is_soldier(customer_profile: dict[str, Any]) -> bool:
    is_soldier = customer_profile.get("is_soldier")
    if isinstance(is_soldier, bool):
        return is_soldier

    job = _clean_text_value(customer_profile.get("job"))
    if not job:
        return False

    return bool(re.search(r"(직업군인|군인|장교|부사관|하사|중사|상사|원사|대위|소령|중령|대령)", job))


def _find_missing_profile_fields(customer_profile: dict[str, Any]) -> list[str]:
    missing_fields = []
    raw_text = str(customer_profile.get("raw_text") or "")
    parsed_customer_fields = set(customer_profile.get("parsed_customer_fields") or [])

    for field_name, label in REQUIRED_PROFILE_FIELDS.items():
        value = customer_profile.get(field_name)

        # 파싱되지 않은 핵심 고객 정보는 eligibility 오판의 가장 큰 원인이므로
        # False/None/빈값 또는 실제 파싱되지 않은 필드는 엄격하게 missing으로 간주합니다.
        if field_name == "income":
            income_value = customer_profile.get("income")
            if income_value in (None, "", 0) or "income" not in parsed_customer_fields:
                missing_fields.append(label)
            continue

        if isinstance(value, bool):
            if value is False and field_name not in parsed_customer_fields and not _has_explicit_boolean_text(raw_text, field_name):
                missing_fields.append(label)
            continue

        if value in (None, "", 0) or field_name not in parsed_customer_fields:
            missing_fields.append(label)

    return _dedupe(missing_fields)


def _enrich_customer_profile(customer_profile: dict[str, Any], *, source: str) -> dict[str, Any]:
    """
    파서가 놓친 핵심 고객 정보를 raw_text에서 한 번 더 보완한다.

    기존 파서를 크게 바꾸지 않으면서 정상 입력이 불필요하게 needs_check로 내려가는 것을 줄이기 위한 보정 단계다.
    """

    profile = dict(customer_profile or {})
    raw_text = _normalize_summary_text(str(profile.get("raw_text") or ""))
    parsed_customer_fields = set(profile.get("parsed_customer_fields") or [])
    parsed_customer_values = dict(profile.get("parsed_customer_values") or {})

    profile.setdefault("customer_profile_source", source)
    profile.setdefault("parsed_customer_fields", [])
    profile.setdefault("parsed_customer_values", {})

    # 이미 구조화된 값이 있으면 True/False 여부와 상관없이 확인된 정보로 취급해야
    # "아니오"가 missing으로 잘못 분류되는 문제를 줄일 수 있습니다.
    for field_name in [
        "age",
        "job",
        "income",
        "monthly_saving_amount",
        "salary_transfer",
        "auto_transfer",
        "card_usage",
        "main_bank",
    ]:
        if field_name in profile and profile.get(field_name) not in (None, "", []):
            parsed_customer_fields.add(field_name)
            parsed_customer_values[field_name] = profile.get(field_name)

    if profile.get("job"):
        cleaned_job = _clean_text_value(profile.get("job"))
        profile["job"] = cleaned_job
        parsed_customer_fields.add("job")
        parsed_customer_values["job"] = cleaned_job

    # customer_agent가 자연어 summary만 넘겨도 eligibility가 핵심 값을 읽을 수 있어야 하므로
    # 나이, 소득, 월 저축 가능액 같은 필수 필드를 summary 표현에서 다시 보강합니다.
    if profile.get("age") in (None, "", 0):
        age_value = _extract_age_from_summary(raw_text)
        if age_value is None:
            birth_date = _extract_birth_date_from_summary(raw_text)
            if birth_date:
                # 생년월일만 있어도 현재 날짜 기준으로 만 나이를 계산해
                # 연령 제한 상품에서 불필요한 missing 처리를 줄입니다.
                age_value = _extract_age_from_birth_date(birth_date)
        if age_value is not None:
            profile["age"] = age_value
            parsed_customer_fields.add("age")
            parsed_customer_values["age"] = age_value

    if profile.get("income") in (None, "", 0):
        income_value = _extract_money_from_summary(
            raw_text,
            labels=["연간 소득", "연소득", "소득", "annual_income"],
        )
        if income_value is not None:
            profile["income"] = income_value
            parsed_customer_fields.add("income")
            parsed_customer_values["income"] = income_value

    if profile.get("monthly_saving_amount") in (None, "", 0):
        saving_value = _extract_money_from_summary(
            raw_text,
            labels=["월 가용 저축액", "가용저축액", "월 납입 가능 금액", "월 저축 가능 금액", "월 저축 가능액", "월 저축액"],
        )
        if saving_value is not None:
            profile["monthly_saving_amount"] = saving_value
            parsed_customer_fields.add("monthly_saving_amount")
            parsed_customer_values["monthly_saving_amount"] = saving_value

    if not profile.get("job"):
        job_match = re.search(r"(직업|customer_job)[:\s|*]*([^\n|]+)", raw_text)
        if job_match:
            profile["job"] = _clean_text_value(job_match.group(2))
            parsed_customer_fields.add("job")
            parsed_customer_values["job"] = profile["job"]

    # 여부 항목은 "아니오"도 유효한 정보이므로
    # 명시 표현이 있으면 기존 기본값보다 summary 해석 결과를 우선합니다.
    parsed_value = _extract_boolean_from_summary(raw_text, "급여이체")
    if parsed_value is not None:
        profile["salary_transfer"] = parsed_value
        parsed_customer_fields.add("salary_transfer")
        parsed_customer_values["salary_transfer"] = parsed_value

    parsed_value = _extract_boolean_from_summary(raw_text, "자동이체")
    if parsed_value is not None:
        profile["auto_transfer"] = parsed_value
        parsed_customer_fields.add("auto_transfer")
        parsed_customer_values["auto_transfer"] = parsed_value

    parsed_value = _extract_boolean_from_summary(raw_text, "카드 사용")
    if parsed_value is not None:
        profile["card_usage"] = parsed_value
        parsed_customer_fields.add("card_usage")
        parsed_customer_values["card_usage"] = parsed_value

    if profile.get("main_bank") is None:
        parsed_value = _extract_boolean_from_summary(raw_text, "주거래")
        if parsed_value is not None:
            profile["main_bank"] = parsed_value
            parsed_customer_fields.add("main_bank")
            parsed_customer_values["main_bank"] = parsed_value

    profile["is_soldier"] = _infer_is_soldier(profile.get("job"), raw_text)
    profile["parsed_customer_fields"] = sorted(parsed_customer_fields)
    profile["parsed_customer_values"] = parsed_customer_values

    return profile

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
    RAG에서 넘어온 상품 후보에 DB products 정형값을 보강합니다.
    추천/가입가능성 판단은 raw_text보다 DB 정형값을 우선 사용합니다.
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

def _find_missing_product_requirements(product: dict[str, Any]) -> list[str]:
    """
    상품 가입조건 누락 여부를 확인합니다.

    원칙:
    - 일반 예금/적금은 DB 정형값으로 금액, 기간, 연령 조건을 확인할 수 있으면
      product_target 문구가 없다는 이유만으로 needs_check 처리하지 않습니다.
    - 군인/장병/간부/직업군인 전용처럼 특정 대상 상품만 가입대상/직업조건 근거를 엄격히 봅니다.
    """
    missing_fields: list[str] = []

    if not _is_special_target_product(product):
        return []

    raw_text = str(product.get("raw_text") or "")
    target_text = " ".join(
        str(product.get(key) or "")
        for key in [
            "product_name",
            "product_target",
            "join_target",
            "eligibility",
            "rag_document_key",
            "raw_text",
        ]
    )

    if not any(
        token in target_text
        for token in ["가입대상", "가입 대상", "대상고객", "군인", "장병", "간부", "직업군인", "전용"]
    ):
        missing_fields.append("product_target")

    if _is_job_sensitive_product(product) and not any(
        token in target_text
        for token in ["직업", "군인", "장병", "간부", "직업군인", "전용", "가입 가능"]
    ):
        missing_fields.append("product_job_condition")

    return _dedupe(missing_fields)


def _is_special_target_product(product: dict[str, Any]) -> bool:
    text = " ".join(
        str(product.get(key) or "")
        for key in [
            "product_name",
            "product_target",
            "join_target",
            "eligibility",
            "rag_document_key",
            "raw_text",
        ]
    )
    normalized = re.sub(r"\s+", "", text).lower()

    special_keywords = [
        "군인",
        "직업군인",
        "장병",
        "장병내일준비",
        "나라사랑",
        "장기간부",
        "군간부",
        "간부",
        "청년",
        "청년도약",
        "소상공인",
        "사업자",
    ]

    return any(keyword in normalized for keyword in special_keywords)


def _is_job_sensitive_product(product: dict[str, Any]) -> bool:
    text = " ".join(
        str(product.get(key) or "")
        for key in [
            "product_name",
            "product_target",
            "join_target",
            "eligibility",
            "rag_document_key",
            "raw_text",
        ]
    )
    normalized = re.sub(r"\s+", "", text).lower()

    job_keywords = [
        "군인",
        "직업군인",
        "장병",
        "나라사랑",
        "장기간부",
        "군간부",
        "간부",
        "소상공인",
        "사업자",
    ]

    return any(keyword in normalized for keyword in job_keywords)


def _extract_age_from_summary(raw_text: str) -> int | None:
    patterns = [
        r"연령[:\s]*([0-9]{1,2})(?!\d)",
        r"나이[:\s]*([0-9]{1,2})(?!\d)",
        r"만\s*([0-9]{1,2})세",
        r"([0-9]{1,2})\s*세",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw_text)
        if match:
            return int(match.group(1))
    return None


def _extract_birth_date_from_summary(raw_text: str) -> str | None:
    match = re.search(r"생년월일[:\s|]*([0-9]{4}-[0-9]{2}-[0-9]{2})", raw_text)
    if match:
        return match.group(1)
    match = re.search(r"([0-9]{4}-[0-9]{2}-[0-9]{2})", raw_text)
    if match:
        return match.group(1)
    return None


def _extract_money_from_summary(raw_text: str, *, labels: list[str]) -> int | None:
    normalized_text = _normalize_summary_text(raw_text)

    for label in labels:
        pattern = rf"{re.escape(label)}[:\s|]*([0-9][0-9,\s]*)\s*(만원|원)?"
        match = re.search(pattern, normalized_text)
        if not match:
            continue
        value_text = match.group(1)
        unit = match.group(2) or "원"
        parsed_value = _parse_money_value(value_text, unit)
        if parsed_value is not None:
            return parsed_value

    # 같은 줄 안에 라벨과 금액이 함께 있으면 금액 부분만 한 번 더 느슨하게 찾습니다.
    for line in normalized_text.splitlines():
        if not any(label in line for label in labels):
            continue
        amount_match = re.search(r"([0-9][0-9,\s]*)\s*(만원|원)", line)
        if not amount_match:
            continue
        parsed_value = _parse_money_value(amount_match.group(1), amount_match.group(2))
        if parsed_value is not None:
            return parsed_value
    return None


def _parse_money_value(value_text: str, unit: str) -> int | None:
    digits = re.sub(r"[^0-9]", "", str(value_text or ""))
    if not digits:
        return None

    amount = int(digits)
    # "만원"은 숫자 × 10,000, "원"은 그대로 사용합니다.
    if unit == "만원":
        return amount * 10000
    return amount


def _normalize_summary_text(raw_text: str) -> str:
    # 요약 문장에 들어가는 굵게 표시나 특수 공백 때문에 숫자 파싱이 흔들리지 않도록
    # 먼저 텍스트를 단순한 형태로 정리합니다.
    text = str(raw_text or "")
    text = text.replace("**", "")
    text = text.replace("\u00a0", " ")
    text = text.replace("\u202f", " ")
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return text


def _clean_text_value(value: Any) -> str:
    text = _normalize_summary_text(str(value or ""))
    text = text.replace("*", "")
    text = text.strip(" :-")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_boolean_from_summary(raw_text: str, label: str) -> bool | None:
    patterns = [
        rf"{re.escape(label)}\s*여부[:\s|]*(예|아니오|yes|no|true|false)",
        rf"{re.escape(label)}[:\s|]*(예|아니오|yes|no|true|false)",
    ]

    for pattern in patterns:
        match = re.search(pattern, raw_text, flags=re.IGNORECASE)
        if not match:
            continue
        value = match.group(1).strip().lower()
        if value in {"예", "yes", "true"}:
            return True
        if value in {"아니오", "no", "false"}:
            return False
    return None


def _has_explicit_boolean_text(raw_text: str, field_name: str) -> bool:
    label_map = {
        "salary_transfer": "급여이체",
        "auto_transfer": "자동이체",
        "card_usage": "카드 사용",
        "main_bank": "주거래",
        "marketing_agree": "마케팅",
    }
    label = label_map.get(field_name)
    if not label:
        return False
    return _extract_boolean_from_summary(raw_text, label) is not None


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


def _normalize_amount_for_eligibility(value: Any) -> int | None:
    """
    가입 금액/월 납입 한도만 원 단위로 변환합니다.
    금리, 퍼센트, 페이지 번호, 출처 번호처럼 금액이 아닌 숫자는 제외합니다.
    """
    if value is None:
        return None

    if isinstance(value, (int, float)):
        amount = int(value)
        return amount if amount >= 1000 else None

    text = str(value).replace(",", "").strip()
    if not text:
        return None

    lowered = text.lower()

    # 금리/이율/퍼센트는 금액 한도가 아님
    if "%" in text or "금리" in text or "이율" in text or "우대" in text:
        return None

    # 출처/페이지 번호는 금액 한도가 아님
    if "출처" in text or "page" in lowered or "p." in lowered:
        return None

    match = re.search(r"(\d+(?:\.\d+)?)\s*만원", text)
    if match:
        return int(float(match.group(1)) * 10000)

    match = re.search(r"(\d+(?:\.\d+)?)\s*천원", text)
    if match:
        return int(float(match.group(1)) * 1000)

    match = re.search(r"(\d+)\s*원", text)
    if match:
        amount = int(match.group(1))
        return amount if amount >= 1000 else None

    if text.isdigit():
        amount = int(text)
        return amount if amount >= 1000 else None

    return None


def _extract_amount_from_raw_text(raw_text: str, patterns: list[str]) -> int | None:
    source = str(raw_text or "")

    for pattern in patterns:
        for match in re.finditer(pattern, source):
            value = match.group(1) if match.groups() else match.group(0)
            amount = _normalize_amount_for_eligibility(value)
            if amount is not None:
                return amount

    return None


def _extract_amount_bounds(product: dict[str, Any]) -> tuple[int | None, int | None]:
    """
    상품의 최소/최대 납입 한도를 추출합니다.
    product_agent는 min_monthly_amount/max_monthly_amount로 넘기기도 하므로
    해당 키를 우선 확인합니다.
    """
    min_amount = (
        _normalize_amount_for_eligibility(product.get("min_amount"))
        or _normalize_amount_for_eligibility(product.get("min_monthly_amount"))
        or _normalize_amount_for_eligibility(product.get("min_deposit_amount"))
    )

    max_amount = (
        _normalize_amount_for_eligibility(product.get("max_amount"))
        or _normalize_amount_for_eligibility(product.get("max_monthly_amount"))
        or _normalize_amount_for_eligibility(product.get("max_deposit_amount"))
    )

    raw_text = str(product.get("raw_text") or "")

    if min_amount is None:
        min_amount = _extract_amount_from_raw_text(
            raw_text,
            [
                r"(?:최소|최저).{0,30}?([0-9,]+(?:\.\d+)?\s*(?:만원|천원|원))",
                r"([0-9,]+(?:\.\d+)?\s*(?:만원|천원|원))\s*이상",
            ],
        )

    if max_amount is None:
        max_amount = _extract_amount_from_raw_text(
            raw_text,
            [
                r"(?:최대|최고|납입한도|한도).{0,30}?([0-9,]+(?:\.\d+)?\s*(?:만원|천원|원))\s*이하",
                r"(?:월|매월).{0,30}?([0-9,]+(?:\.\d+)?\s*(?:만원|천원|원))\s*이하",
                r"([0-9,]+(?:\.\d+)?\s*(?:만원|천원|원))\s*이하",
            ],
        )

    # 3원, 9원 같은 비정상 값은 금리/출처 숫자를 잘못 파싱한 것으로 보고 무시
    if min_amount is not None and min_amount < 1000:
        min_amount = None
    if max_amount is not None and max_amount < 1000:
        max_amount = None

    return min_amount, max_amount

def _extract_period_bounds(product: dict[str, Any]) -> tuple[int | None, int | None]:
    period_min = (
        _to_int_or_none(product.get("min_period_months"))
        or _to_int_or_none(product.get("min_period"))
        or _to_int_or_none(product.get("min_contract_months"))
    )

    period_max = (
        _to_int_or_none(product.get("max_period_months"))
        or _to_int_or_none(product.get("max_period"))
        or _to_int_or_none(product.get("max_contract_months"))
    )

    if period_min is not None or period_max is not None:
        return period_min, period_max

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
