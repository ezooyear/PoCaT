"""
Recommend agent.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import AIMessage

from agents.base import get_first_matching_path, make_agent_result, mirror_result_fields
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


def recommend_agent_node(state: AgentState) -> dict:
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
            financial_results, financial_calculation_source = _load_financial_results(state, agent_outputs)

            with langfuse_observation(
                name="recommend_agent.prepare_context",
                as_type="span",
                input={
                    "eligibility_result_count": len(eligibility_results or []),
                    "product_candidate_count": len(product_candidates or []),
                    "financial_result_count": len(financial_results or []),
                    "financial_calculation_source": financial_calculation_source,
                },
                metadata={"agent": "recommend_agent", "step": "prepare_context"},
            ) as step_observation:
                update_observation(
                    step_observation,
                    output={
                        "eligibility_preview": eligibility_results[:5] if isinstance(eligibility_results, list) else eligibility_results,
                        "product_candidates_preview": product_candidates[:5] if isinstance(product_candidates, list) else product_candidates,
                        "financial_preview": financial_results[:5] if isinstance(financial_results, list) else financial_results,
                        "financial_calculation_source": financial_calculation_source,
                    },
                    metadata={"agent": "recommend_agent"},
                )

            normalized_results = _normalize_eligibility_results(eligibility_results)
            recommendations, excluded_products, status, fallback_reason, required_next_steps, matched_products = _build_guarded_recommendations(
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

            recommend_result = make_agent_result(
                status="success",
                result={
                    "status": status,
                    "summary": summary,
                    "recommendations": recommendations,
                    "recommended_products": recommendations,
                    "recommendation_count": len(recommendations),
                    "excluded_products": excluded_products,
                    "fallback_reason": fallback_reason,
                    "required_next_steps": required_next_steps,
                    "financial_results": financial_results,
                    "financial_calculation_source": financial_calculation_source,
                    "financial_calculation_count": len(financial_results),
                    "financial_calculation_product_names": [
                        item.get("product_name") for item in financial_results if isinstance(item, dict)
                    ],
                    "matched_products": matched_products,
                    "source_agent": "recommend_agent",
                },
                evidence=recommendations,
                error=None,
            )
            recommend_result = mirror_result_fields(
                recommend_result,
                field_names=[
                    "recommendations",
                    "recommended_products",
                    "recommendation_count",
                    "excluded_products",
                    "fallback_reason",
                    "required_next_steps",
                    "financial_results",
                    "financial_calculation_source",
                    "financial_calculation_count",
                    "financial_calculation_product_names",
                    "matched_products",
                ],
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
                    "result_key": "recommend_result",
                    "input_sources": f"eligibility_results;product_candidates;financial_results:{financial_calculation_source}",
                    "output_keys": "recommendations,matched_products,excluded_products",
                    "fallback_reason": fallback_reason,
                    "financial_calculation_source": financial_calculation_source,
                    "financial_calculation_count": len(financial_results),
                    "financial_calculation_product_names": ",".join(
                        item.get("product_name", "") for item in financial_results if isinstance(item, dict)
                    )[:500] or None,
                    "matched_product_count": len(matched_products),
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
    results, _ = get_first_matching_path(
        ("state.eligibility_results", state, ("eligibility_results",)),
        ("eligibility_result.results", state.get("eligibility_result"), ("results",)),
        ("eligibility_result.result.results", state.get("eligibility_result"), ("result", "results")),
        ("agent_outputs.eligibility_agent.results", state.get("agent_outputs"), ("eligibility_agent", "results")),
        ("agent_outputs.eligibility_agent.result.results", state.get("agent_outputs"), ("eligibility_agent", "result", "results")),
    )
    return list(results or []) if isinstance(results, list) else []


def _load_product_candidates(state: AgentState, agent_outputs: dict) -> list[dict]:
    product_candidates, _ = get_first_matching_path(
        ("state.product_candidates", state, ("product_candidates",)),
        ("product_result.products", state.get("product_result"), ("products",)),
        ("product_result.result.products", state.get("product_result"), ("result", "products")),
        ("product_result.product_candidates", state.get("product_result"), ("product_candidates",)),
        ("product_result.result.product_candidates", state.get("product_result"), ("result", "product_candidates")),
        ("agent_outputs.product_agent.products", agent_outputs, ("product_agent", "products")),
        ("agent_outputs.product_agent.result.products", agent_outputs, ("product_agent", "result", "products")),
        ("agent_outputs.product_agent.result.product_candidates", agent_outputs, ("product_agent", "result", "product_candidates")),
    )
    return list(product_candidates or []) if isinstance(product_candidates, list) else []


def _load_financial_results(state: AgentState, agent_outputs: dict) -> tuple[list[dict], str]:
    """financial 계산 결과를 로드하고 (results, source) 튜플 반환."""
    financial_results, source = get_first_matching_path(
        ("state.financial_results", state, ("financial_results",)),
        ("financial_result.calculations", state.get("financial_result"), ("calculations",)),
        ("financial_result.result.calculations", state.get("financial_result"), ("result", "calculations")),
        ("agent_outputs.financial_agent.calculations", agent_outputs, ("financial_agent", "calculations")),
        ("agent_outputs.financial_agent.result.calculations", agent_outputs, ("financial_agent", "result", "calculations")),
    )
    if isinstance(financial_results, list) and financial_results:
        return financial_results, source

    financial_result = state.get("financial_result")
    if financial_result:
        extracted = _extract_financial_results_from_container(financial_result)
        if extracted:
            return extracted, "financial_result.tool_results"

    financial_agent_output = agent_outputs.get("financial_agent") or {}
    calcs = _extract_calculations(financial_agent_output)
    if calcs:
        return calcs, "agent_outputs.financial_agent.calculations"
    extracted = _extract_financial_results_from_container(financial_agent_output)
    return extracted, "agent_outputs.financial_agent.tool_results" if extracted else "missing"


def _extract_calculations(container: Any) -> list[dict]:
    """financial_result 또는 agent_output에서 structured calculations를 추출한다."""
    if not isinstance(container, dict):
        return []
    calcs = container.get("calculations")
    if isinstance(calcs, list) and calcs:
        return calcs
    payload = container.get("result", {})
    if isinstance(payload, dict):
        calcs = payload.get("calculations")
        if isinstance(calcs, list) and calcs:
            return calcs
    return []


def _extract_financial_results_from_container(container: Any) -> list[dict]:
    if isinstance(container, dict):
        payload = container.get("result", {})
        tool_results = payload.get("tool_results", []) if isinstance(payload, dict) else []
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
) -> tuple[list[dict], list[dict], str, str | None, list[str], list[dict]]:
    """
    앞단 결과가 부실할 때 확정 추천을 막고 안전한 fallback 상태를 만든다.
    Returns: (recommendations, excluded_products, status, fallback_reason, required_next_steps, matched_products)
    """

    excluded_products = _build_excluded_products(normalized_results)
    matched_products: list[dict] = []

    if not normalized_results:
        return [], excluded_products, "needs_more_info", "eligibility_results_missing", ["eligibility_results 확인"], []

    if not product_candidates:
        return [], excluded_products, "needs_more_info", "product_candidates_missing", ["product_agent 결과 확인"], []

    if not financial_results:
        return [], excluded_products, "recommendation_deferred", "financial_results_missing", ["financial_agent 결과 확인"], []

    eligible_results = [
        item
        for item in normalized_results
        if item.get("status") in ALLOWED_ELIGIBILITY_STATUSES and item.get("eligible") is True
    ]

    if not eligible_results:
        return [], excluded_products, "no_eligible_product", "no_eligible_product", ["고객 조건 또는 상품 조건 재확인"], []

    # 상품 후보 이름 집합 (정규화 + alias)
    valid_candidate_name_map = _build_candidate_name_map(product_candidates)

    # financial 결과 이름 집합 (정규화 + alias)
    financial_name_map = _build_financial_name_map(financial_results)

    filtered_eligible_results = []
    for item in eligible_results:
        product_name = item.get("product_name", "미확인 상품")
        norm = _normalize_name(product_name)
        alias = _name_alias(product_name)

        has_candidate = norm in valid_candidate_name_map or alias in valid_candidate_name_map
        has_financial = norm in financial_name_map or alias in financial_name_map

        if not has_candidate:
            excluded_products.append({
                "product_name": product_name,
                "status": "invalid_product",
                "reason": "상품 후보와 매칭되지 않는 이름이라 추천에서 제외했습니다.",
                "source_agent": "recommend_agent",
            })
            continue

        if not _is_valid_product_name(product_name):
            excluded_products.append({
                "product_name": product_name,
                "status": "invalid_product",
                "reason": "비정상 상품명으로 보여 추천에서 제외했습니다.",
                "source_agent": "recommend_agent",
            })
            continue

        matched_products.append({
            "product_name": product_name,
            "eligibility_status": item.get("status"),
            "has_financial_calculation": has_financial,
        })
        filtered_eligible_results.append(item)

    if not filtered_eligible_results:
        return [], excluded_products, "no_eligible_product", "no_valid_eligible_product", ["product_agent 결과 정제 확인"], matched_products

    recommendations = build_recommendations(filtered_eligible_results, financial_results)

    if not recommendations:
        return [], excluded_products, "recommendation_deferred", "recommendation_build_failed", ["financial_result 파싱 결과 확인"], matched_products

    return recommendations, _dedupe_excluded_products(excluded_products), "recommended", None, [], matched_products


def _build_candidate_name_map(product_candidates: list[dict]) -> dict[str, str]:
    """product_name → original_name 매핑 (정규화 + alias 포함)"""
    result: dict[str, str] = {}
    for c in product_candidates:
        name = c.get("product_name", "")
        if not _is_valid_product_name(name):
            continue
        result[_normalize_name(name)] = name
        alias = _name_alias(name)
        if alias:
            result[alias] = name
    return result


def _build_financial_name_map(financial_results: list[dict]) -> dict[str, str]:
    """financial product_name → original_name 매핑 (정규화 + alias 포함)"""
    result: dict[str, str] = {}
    for f in financial_results:
        if not isinstance(f, dict):
            continue
        name = f.get("product_name", "")
        if not name:
            continue
        result[_normalize_name(name)] = name
        alias = _name_alias(name)
        if alias:
            result[alias] = name
    return result


def _name_alias(name: str) -> str:
    """괄호 안 설명 제거한 alias 생성"""
    alias = re.sub(r"\([^)]*\)", "", str(name or ""))
    return _normalize_name(alias)


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

    lines = ["확정 추천을 생성하지 않았습니다."]

    if fallback_reason == "eligibility_results_missing":
        lines.append("- 가입 가능 여부 결과가 없어 추천을 보류했습니다.")
    elif fallback_reason == "product_candidates_missing":
        lines.append("- 상품 후보 정보가 없어 추천을 보류했습니다.")
    elif fallback_reason == "financial_results_missing":
        lines.append("- 금융 계산 결과가 없어 금액 기반 추천을 보류했습니다.")
    elif fallback_reason in {"no_eligible_product", "no_valid_eligible_product"}:
        lines.append("- 추천 가능한 eligible 상품이 없어 억지 추천을 하지 않았습니다.")
    elif fallback_reason:
        lines.append(f"- fallback reason: {fallback_reason}")

    if excluded_products:
        lines.append("")
        lines.append("제외된 상품:")
        for item in excluded_products:
            lines.append(f"- {item.get('product_name', '미확인 상품')}: {item.get('status')} / {item.get('reason', '-')}")

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
    fallback_result = mirror_result_fields(
        fallback_result,
        field_names=[
            "recommendations",
            "recommended_products",
            "recommendation_count",
            "excluded_products",
            "fallback_reason",
            "required_next_steps",
            "financial_results",
        ],
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
