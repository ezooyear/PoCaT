"""
Recommend agent.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from agents.base import get_financial_calculations, get_product_candidates, make_agent_result
from agents.recommend.prompts import RECOMMEND_SYSTEM_PROMPT
from agents.recommend.tools import (
    build_recommendation_summary,
    build_recommendations,
    parse_financial_results,
)
from graph.state import AgentState
from observability.langfuse import langfuse_observation, update_observation


ALLOWED_ELIGIBILITY_STATUSES = {"eligible"}
EXCLUDED_ELIGIBILITY_STATUSES = {"needs_check", "rejected", "invalid_product"}


async def recommend_agent_node(state: AgentState) -> dict:
    with langfuse_observation(
        name="recommend_agent",
        as_type="span",
        input=_build_trace_input(state, "recommend_agent"),
        metadata={"agent": "recommend_agent"},
    ) as observation:
        try:
            agent_outputs = dict(state.get("agent_outputs") or {})

            eligibility_results = _load_eligibility_results(state)
            product_candidates = _load_product_candidates(state, agent_outputs)
            financial_results = _load_financial_results(state, agent_outputs)

            with langfuse_observation(
                name="recommend_agent.prepare_context",
                as_type="span",
                input={
                    "eligibility_result_count": len(eligibility_results or []),
                    "product_candidate_count": len(product_candidates or []),
                    "financial_result_count": len(financial_results or []),
                },
                metadata={"agent": "recommend_agent", "step": "prepare_context"},
            ) as step_observation:
                update_observation(
                    step_observation,
                    output={
                        "eligibility_preview": eligibility_results[:5] if isinstance(eligibility_results, list) else eligibility_results,
                        "product_candidates_preview": product_candidates[:5] if isinstance(product_candidates, list) else product_candidates,
                        "financial_preview": financial_results[:5] if isinstance(financial_results, list) else financial_results,
                    },
                    metadata={"agent": "recommend_agent"},
                )

            normalized_results = _normalize_eligibility_results(eligibility_results)
            # 디버깅
            print("\n[RECOMMEND DEBUG]")
            print("product_candidates =", [
                p.get("product_name")
                for p in product_candidates
                if isinstance(p, dict)
            ])
            print("eligibility_results =", [
                {
                    "name": e.get("product_name"),
                    "status": e.get("status"),
                    "eligible": e.get("eligible"),
                    "reasons": e.get("reasons") or e.get("ineligibility_reasons"),
                    "missing": e.get("missing_fields") or e.get("check_required"),
                }
                for e in eligibility_results
                if isinstance(e, dict)
            ])
            print("financial_results =", [
                {
                    "name": f.get("product_name"),
                    "status": f.get("status"),
                    "monthly_amount": f.get("monthly_amount"),
                    "term_months": f.get("term_months"),
                    "estimated_interest": f.get("estimated_interest"),
                    "maturity_amount": f.get("maturity_amount"),
                }
                for f in financial_results
                if isinstance(f, dict)
            ])
            print("[/RECOMMEND DEBUG]\n")
            recommendations, excluded_products, status, fallback_reason, required_next_steps = _build_guarded_recommendations(
                normalized_results=normalized_results,
                product_candidates=product_candidates,
                financial_results=financial_results,
            )

            summary = _build_guarded_recommendation_summary(
                recommendations=recommendations,
                excluded_products=excluded_products,
                status=status,
                fallback_reason=fallback_reason,
            )

            normalized_recommendations = []
            for item in recommendations:
                normalized_recommendations.append(
                    {
                        "rank": item.get("rank"),
                        "product_name": item.get("product_name"),
                        "product_type": item.get("product_type") or None,
                        "recommendation_status": "recommended" if status == "recommended" else status,
                        "reason": item.get("reason") or "추천 사유가 생성되지 않았습니다.",
                        "eligibility_status": item.get("status") or "eligible",
                        "applied_rate": item.get("applied_rate") or item.get("base_rate"),
                        "monthly_amount": item.get("monthly_amount"),
                        "term_months": item.get("term_months"),
                        "total_principal": item.get("total_principal"),
                        "payment_plan_text": item.get("payment_plan_text"),
                        "calculation_assumption": item.get("calculation_assumption"),
                        "monthly_amount_source": item.get("monthly_amount_source"),
                        "monthly_amount_source_label": item.get("monthly_amount_source_label"),
                        "term_months_source": item.get("term_months_source"),
                        "term_months_source_label": item.get("term_months_source_label"),
                        "estimated_maturity_amount": item.get("maturity_amount"),
                        "estimated_interest_after_tax": item.get("estimated_interest"),
                        "estimated_interest_before_tax": item.get("before_tax_interest"),
                        "warnings": item.get("warnings") or [],
                        "source_product_name": item.get("product_name"),
                        **{
                            k: v
                            for k, v in item.items()
                            if k
                            not in {
                                "rank",
                                "product_name",
                                "product_type",
                                "status",
                                "reason",
                                "applied_rate",
                                "maturity_amount",
                                "estimated_interest",
                                "before_tax_interest",
                                "warnings",
                            }
                        },
                    }
                )

            product_names = [_normalize_name(item.get("product_name")) for item in product_candidates if item.get("product_name")]
            financial_names = [_normalize_name(item.get("product_name")) for item in financial_results if item.get("product_name")]
            matched_products = [
                {
                    "product_name": item.get("product_name"),
                    "has_product": True,
                    "has_eligibility": True,
                    "has_financial_calculation": _normalize_name(item.get("product_name", "")) in financial_names,
                }
                for item in normalized_recommendations
            ]
            unmatched_products = [
                {
                    "product_name": str(item.get("product_name")),
                    "has_product": _normalize_name(item.get("product_name", "")) in product_names,
                    "has_eligibility": False,
                    "has_financial_calculation": _normalize_name(item.get("product_name", "")) in financial_names,
                }
                for item in excluded_products
            ]

            recommend_result = make_agent_result(
                status="success",
                result={
                    "status": status,
                    "summary": summary,
                    "recommendations": normalized_recommendations,
                    "recommended_products": normalized_recommendations,
                    "recommendation_count": len(normalized_recommendations),
                    "matched_products": matched_products,
                    "unmatched_products": unmatched_products,
                    "excluded_products": excluded_products,
                    "fallback_reason": fallback_reason,
                    "required_next_steps": required_next_steps,
                    "financial_results": financial_results,
                    "source_agent": "recommend_agent",
                },
                evidence=normalized_recommendations,
                error=None,
            )
            agent_outputs["recommend_agent"] = recommend_result

            update_observation(
                observation,
                output=_build_trace_output(
                    recommend_result,
                    extra_output={
                        "fallback_reason": fallback_reason,
                        "required_next_steps": required_next_steps,
                        "excluded_products": excluded_products[:10],
                        "top_recommendations": [
                            {
                                "product_name": item.get("product_name"),
                                "score": item.get("score"),
                                "payment_plan_text": item.get("payment_plan_text"),
                                "reason": (item.get("reason") or "")[:200],
                            }
                            for item in recommendations[:5]
                            if isinstance(item, dict)
                        ],
                    },
                ),
                metadata={
                    "agent": "recommend_agent",
                    "task_type": state.get("task_type"),
                    "status": status,
                    "fallback_reason": fallback_reason,
                },
            )

            completed_agents = list(state.get("completed_agents") or [])
            if "recommend_agent" not in completed_agents:
                completed_agents.append("recommend_agent")

            return {
                "messages": [AIMessage(content=summary)],
                "agent_outputs": agent_outputs,
                "current_step": (state.get("current_step") or 0) + 1,
                "current_agent": "recommend_agent",
                "completed_agents": completed_agents,
                "financial_results": financial_results,
                "recommendation_results": recommendations,
                "recommend_result": recommend_result,
                "context": {
                    **(state.get("context") or {}),
                    "recommend_prompt": RECOMMEND_SYSTEM_PROMPT,
                },
            }
        except Exception as error:
            fallback = _build_recommend_exception_fallback(state, str(error))
            update_observation(
                observation,
                output=_build_trace_output(
                    fallback["recommend_result"],
                    extra_output={
                        "fallback_reason": fallback["recommend_result"]["result"].get("fallback_reason"),
                        "error_message": str(error),
                    },
                ),
                metadata={
                    "agent": "recommend_agent",
                    "status": fallback["recommend_result"]["result"].get("status"),
                    "fallback_reason": fallback["recommend_result"]["result"].get("fallback_reason"),
                    "error_message": str(error),
                },
            )
            return fallback


def _load_eligibility_results(state: AgentState) -> list[dict]:
    eligibility_results = state.get("eligibility_results") or []
    if eligibility_results:
        return eligibility_results

    eligibility_result = state.get("eligibility_result") or {}
    if isinstance(eligibility_result, dict):
        payload = eligibility_result.get("result", {})
        if isinstance(payload, dict):
            results = payload.get("results") or []
            if isinstance(results, list):
                return results
    return []


def _load_product_candidates(state: AgentState, agent_outputs: dict) -> list[dict]:
    candidates = state.get("product_candidates")
    if isinstance(candidates, list) and candidates:
        return candidates

    products = get_product_candidates(state)
    if isinstance(products.get("data"), list) and products.get("data"):
        return products["data"]

    product_result = state.get("product_result") or {}
    if isinstance(product_result, dict):
        payload = product_result.get("result", {})
        if isinstance(payload, dict) and isinstance(payload.get("product_candidates"), list):
            return payload.get("product_candidates") or []

    product_agent_output = agent_outputs.get("product_agent") or {}
    if isinstance(product_agent_output, dict):
        payload = product_agent_output.get("result", {})
        if isinstance(payload, dict) and isinstance(payload.get("product_candidates"), list):
            return payload.get("product_candidates") or []

    return []


def _load_financial_results(state: AgentState, agent_outputs: dict) -> list[dict]:
    financial_results = state.get("financial_results")
    if isinstance(financial_results, list) and financial_results:
        return financial_results

    financial_path = get_financial_calculations(state)
    if isinstance(financial_path.get("data"), list) and financial_path.get("data"):
        return financial_path["data"]

    financial_result = state.get("financial_result")
    if financial_result:
        extracted = _extract_financial_results_from_container(financial_result)
        if extracted:
            return extracted

    financial_agent_output = agent_outputs.get("financial_agent", "")
    return _extract_financial_results_from_container(financial_agent_output)


def _extract_financial_results_from_container(container: Any) -> list[dict]:
    if isinstance(container, dict):
        payload = container.get("result", {})
        if isinstance(payload, dict):
            # financial_agent가 추천 흐름에서 만든 구조화 결과를 가장 먼저 사용합니다.
            # 이 경로에 monthly_amount, term_months, total_principal, source label 등이 들어 있습니다.
            financial_results = payload.get("financial_results")
            if isinstance(financial_results, list) and any(isinstance(item, dict) for item in financial_results):
                return [dict(item) for item in financial_results if isinstance(item, dict)]

            calculations = payload.get("calculations")
            if isinstance(calculations, list) and any(isinstance(item, dict) for item in calculations):
                return [dict(item) for item in calculations if isinstance(item, dict)]

            tool_results = payload.get("tool_results", [])
        else:
            tool_results = []

        extracted_results = []
        for item in tool_results:
            if not isinstance(item, dict):
                continue
            tool_result = item.get("tool_result")
            if isinstance(tool_result, str):
                extracted_results.extend(parse_financial_results(tool_result))
        return extracted_results

    parsed = parse_financial_results(container)
    return parsed if isinstance(parsed, list) else []


def _normalize_eligibility_results(eligibility_results: list[dict]) -> list[dict]:
    normalized = []
    for item in eligibility_results:
        if not isinstance(item, dict):
            normalized.append(
                {
                    "product_name": str(item),
                    "eligible": False,
                    "status": "invalid_product",
                    "reasons": ["Eligibility 결과 형식이 올바르지 않습니다."],
                    "missing_fields": [],
                    "source_agent": "eligibility_agent",
                }
            )
            continue

        normalized.append(
            {
                "product_name": item.get("product_name", "미확인 상품"),
                "eligible": bool(item.get("eligible")) and item.get("status") == "eligible",
                "status": item.get("status") or ("eligible" if item.get("eligible") else "needs_check"),
                "reasons": list(item.get("reasons") or item.get("ineligibility_reasons") or []),
                "missing_fields": list(item.get("missing_fields") or item.get("check_required") or []),
                "source_agent": item.get("source_agent", "eligibility_agent"),
                "bonus_conditions_met": list(item.get("bonus_conditions_met") or []),
                "bonus_conditions_missing": list(item.get("bonus_conditions_missing") or []),
            }
        )
    return normalized

def _build_guarded_recommendations(
    *,
    normalized_results: list[dict],
    product_candidates: list[dict],
    financial_results: list[dict],
) -> tuple[list[dict], list[dict], str, str | None, list[str]]:
    """
    앞단 결과가 부실할 때 확정 추천을 막고 안전한 fallback 상태를 만든다.
    """

    excluded_products = _build_excluded_products(normalized_results)

    if not normalized_results:
        # eligibility 결과가 없는데 추천을 생성하면 가입 가능 여부를 recommend가 대신 판단하게 되므로
        # needs_more_info로 멈추고 앞단 정보 보완을 요구합니다.
        return [], excluded_products, "needs_more_info", "eligibility_results_missing", ["eligibility_results 확인"]

    if not product_candidates:
        return [], excluded_products, "needs_more_info", "product_candidates_missing", ["product_agent 결과 확인"]

    if not financial_results:
        # 금융 계산 근거가 없으면 금액 기반 확정 추천이 오해를 만들 수 있으므로
        # recommendation_deferred로 낮춰서 후속 계산을 기다립니다.
        return [], excluded_products, "recommendation_deferred", "financial_results_missing", ["financial_agent 결과 확인"]

    eligible_results = [
        item
        for item in normalized_results
        if item.get("status") in ALLOWED_ELIGIBILITY_STATUSES and item.get("eligible") is True
    ]

    if not eligible_results:
        return [], excluded_products, "no_eligible_product", "no_eligible_product", ["고객 조건 또는 상품 조건 재확인"]

    valid_candidate_names = {
        _normalize_name(candidate.get("product_name", ""))
        for candidate in product_candidates
        if _is_valid_product_name(candidate.get("product_name", ""))
    }

    filtered_eligible_results = []
    for item in eligible_results:
        normalized_name = _normalize_name(item.get("product_name", ""))

        # 상품 후보 목록에 없는 이름을 추천하면 product parsing 오류가 추천 결과로 이어질 수 있으므로
        # 실제 product_candidate와 매칭되는 이름만 추천 대상으로 남깁니다.
        if normalized_name not in valid_candidate_names:
            excluded_products.append(
                {
                    "product_name": item.get("product_name", "미확인 상품"),
                    "status": "invalid_product",
                    "reason": "상품 후보와 매칭되지 않는 이름이라 추천에서 제외했습니다.",
                    "source_agent": "recommend_agent",
                }
            )
            continue

        if not _is_valid_product_name(item.get("product_name", "")):
            excluded_products.append(
                {
                    "product_name": item.get("product_name", "미확인 상품"),
                    "status": "invalid_product",
                    "reason": "비정상 상품명으로 보여 추천에서 제외했습니다.",
                    "source_agent": "recommend_agent",
                }
            )
            continue

        filtered_eligible_results.append(item)

    if not filtered_eligible_results:
        return [], excluded_products, "no_eligible_product", "no_valid_eligible_product", ["product_agent 결과 정제 확인"]

    recommendations = build_recommendations(filtered_eligible_results, financial_results)

    if not recommendations:
        return [], excluded_products, "recommendation_deferred", "recommendation_build_failed", ["financial_result 파싱 결과 확인"]

    return recommendations, _dedupe_excluded_products(excluded_products), "recommended", None, []


def _build_excluded_products(normalized_results: list[dict]) -> list[dict]:
    excluded = []
    for item in normalized_results:
        status = item.get("status")
        if status not in EXCLUDED_ELIGIBILITY_STATUSES:
            continue
        excluded.append(
            {
                "product_name": item.get("product_name", "미확인 상품"),
                "status": status,
                "reason": ", ".join(item.get("reasons") or item.get("missing_fields") or []) or status,
                "source_agent": item.get("source_agent", "eligibility_agent"),
            }
        )
    return _dedupe_excluded_products(excluded)


def _build_guarded_recommendation_summary(
    *,
    recommendations: list[dict],
    excluded_products: list[dict],
    status: str,
    fallback_reason: str | None,
) -> str:
    if status == "recommended":
        rejected_items = [
            {
                "product_name": item.get("product_name"),
                "ineligibility_reasons": [item.get("reason", "")],
            }
            for item in excluded_products
            if item.get("status") in {"rejected", "invalid_product"}
        ]
        needs_check_items = [
            {
                "product_name": item.get("product_name"),
                "check_required": [item.get("reason", "")],
            }
            for item in excluded_products
            if item.get("status") == "needs_check"
        ]
        return build_recommendation_summary(
            recommendations,
            needs_check_items,
            rejected_items,
        )

    lines = ["현재 조건만으로 바로 추천드릴 수 있는 상품은 확인되지 않았습니다."]

    if fallback_reason == "eligibility_results_missing":
        lines.append("- 가입 가능 여부를 확인하지 못해 추천을 잠시 보류했습니다.")
    elif fallback_reason == "product_candidates_missing":
        lines.append("- 비교할 상품 후보를 찾지 못해 추천을 잠시 보류했습니다.")
    elif fallback_reason == "financial_results_missing":
        lines.append("- 예상 이자 계산 결과가 없어 금액 기준 추천을 보류했습니다.")
    elif fallback_reason in {"no_eligible_product", "no_valid_eligible_product"}:
        pass
    elif fallback_reason:
        lines.append(f"- 추가 확인 사유: {fallback_reason}")

    if excluded_products:
        lines.append("")
        lines.append("이번 추천에서 제외한 상품:")
        for item in excluded_products:
            if not isinstance(item, dict):
                continue
            lines.append(_format_excluded_product_for_customer(item))
    return "\n".join(lines)


def _build_recommend_exception_fallback(state: AgentState, error_message: str) -> dict[str, Any]:
    fallback_result = make_agent_result(
        status="failed",
        result={
            "status": "recommendation_deferred",
            "summary": "추천 생성 중 오류가 발생해 추천을 보류했습니다.",
            "recommendations": [],
            "recommended_products": [],
            "recommendation_count": 0,
            "excluded_products": [],
            "fallback_reason": "exception_in_recommend_agent",
            "required_next_steps": ["recommend_agent 예외 로그 확인"],
            "financial_results": state.get("financial_results") or [],
            "source_agent": "recommend_agent",
        },
        evidence=[],
        error=error_message,
    )

    agent_outputs = dict(state.get("agent_outputs") or {})
    agent_outputs["recommend_agent"] = fallback_result

    completed_agents = list(state.get("completed_agents") or [])
    if "recommend_agent" not in completed_agents:
        completed_agents.append("recommend_agent")

    return {
        "messages": [AIMessage(content=fallback_result["result"]["summary"])],
        "agent_outputs": agent_outputs,
        "current_step": (state.get("current_step") or 0) + 1,
        "current_agent": "recommend_agent",
        "completed_agents": completed_agents,
        "financial_results": state.get("financial_results") or [],
        "recommendation_results": [],
        "recommend_result": fallback_result,
        "context": {
            **(state.get("context") or {}),
            "recommend_prompt": RECOMMEND_SYSTEM_PROMPT,
        },
    }


def _dedupe_excluded_products(items: list[dict]) -> list[dict]:
    seen = set()
    deduped = []
    for item in items:
        key = (
            _normalize_name(item.get("product_name", "")),
            item.get("status"),
            item.get("reason"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _is_valid_product_name(product_name: Any) -> bool:
    text = str(product_name or "").strip()
    if not text or text == "미확인 상품":
        return False
    if len(text) > 40:
        return False
    invalid_markers = ["적용조건", "우대이율", "신규가입일", "영업점", "가입방법", "유의사항"]
    if any(marker in text for marker in invalid_markers):
        return False
    return True


def _normalize_name(name: str) -> str:
    return "".join(str(name or "").lower().split())


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


def _format_excluded_product_for_customer(item: dict) -> str:
    product_name = str(item.get("product_name") or "미확인 상품").strip()
    reason = str(item.get("reason") or "").strip()

    return f"- {product_name}: {_soften_excluded_reason(reason)}"


def _soften_excluded_reason(reason: str) -> str:
    text = str(reason or "").strip()

    if not text or text in {"rejected", "invalid_product", "needs_check"}:
        return "현재 확인된 조건과 맞지 않아 이번 추천에서 제외했습니다."

    if "군 간부" in text or "장기간부" in text or "간부" in text:
        return "군 간부 등 특정 직군 대상 상품으로 확인되어, 현재 고객님의 직업 조건과 맞지 않아 이번 추천에서 제외했습니다."

    if "직업군인" in text or "나라사랑" in text or "군인" in text:
        return "직업군인 등 특정 직군 대상 상품으로 확인되어, 현재 고객님의 직업 조건과 맞지 않아 이번 추천에서 제외했습니다."

    if "상품 가입조건을 충분히 확인할 수 없어" in text:
        return "가입 대상이나 세부 조건 확인이 필요해 이번에는 확정 추천에 포함하지 않았습니다."

    return text
