"""
Eligibility agent.
"""
from __future__ import annotations

import re

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


GENERIC_PRODUCT_NAMES = {
    "미확인 상품",
    "다음 단계",
    "최종 추천",
    "추천 상품",
    "추천 결과",
    "상품 유형",
    "유형",
    "정액적립식 예금",
    "자유적립식 예금",
    "전용 우대 적금",
    "군인 전용 우대 적금",
    "직업군인 우대 적금",
    "도약 적금",
    "입출금통장",
    "KB국민은행 입출금통장",
}

SUSPICIOUS_PRODUCT_TOKENS = [
    "추천 이유",
    "가입 가능",
    "기본 조건 충돌 없음",
    "우대조건 충족",
    "가입 신청",
    "자동이체 설정",
    "확인해 보세요",
    "배분하면",
    "고객님",
    "예상 이자",
    "만기 금액",
    "상품 유형",
    "유형 ▪",
    "보유",
    "제출",
    "예시",
    "경우",
]


def eligibility_agent_node(state: AgentState) -> dict:
    with langfuse_observation(
        name="eligibility_agent",
        as_type="span",
        input={"task_type": state.get("task_type")},
        metadata={"agent": "eligibility_agent"},
    ) as observation:
        agent_outputs = dict(state.get("agent_outputs") or {})
        customer_result = state.get("customer_result") or {}
        product_result = state.get("product_result") or {}

        customer_profile = _resolve_customer_profile(state, customer_result, agent_outputs)
        customer_accounts = _resolve_customer_accounts(state, customer_result, agent_outputs)
        product_candidates = _resolve_product_candidates(state, product_result, agent_outputs)

        results = [
            evaluate_product_eligibility(customer_profile, customer_accounts, product)
            for product in product_candidates
        ]
        summary = build_eligibility_summary(results)

        eligible_products = [
            item for item in results
            if item.get("eligible") is True and not item.get("check_required")
        ]
        needs_check_products = [
            item for item in results
            if item.get("eligible") is True and item.get("check_required")
        ]
        rejected_products = [
            item for item in results
            if item.get("eligible") is not True
        ]

        eligibility_result = make_agent_result(
            status="success",
            result={
                "summary": summary,
                "results": results,
                "eligible_products": eligible_products,
                "needs_check_products": needs_check_products,
                "rejected_products": rejected_products,
                "result_count": len(results),
                "recommendable_count": len(eligible_products),
                "needs_check_count": len(needs_check_products),
                "rejected_count": len(rejected_products),
                "customer_profile": customer_profile,
                "customer_accounts": customer_accounts,
                "product_candidates": product_candidates,
            },
            evidence=results,
            error=None,
        )
        agent_outputs["eligibility_agent"] = eligibility_result

        update_observation(
            observation,
            output={
                "summary_preview": summary[:500],
                "customer_job": customer_profile.get("job"),
                "customer_age": customer_profile.get("age"),
                "candidate_count": len(product_candidates),
                "candidate_names": [
                    item.get("product_name")
                    for item in product_candidates[:5]
                    if isinstance(item, dict)
                ],
                "eligible_count": len(eligible_products),
                "needs_check_count": len(needs_check_products),
                "rejected_count": len(rejected_products),
                "eligible_products": [item.get("product_name") for item in eligible_products[:5]],
                "needs_check_products": [item.get("product_name") for item in needs_check_products[:5]],
                "rejected_products": [item.get("product_name") for item in rejected_products[:5]],
            },
            metadata={
                "agent": "eligibility_agent",
                "task_type": state.get("task_type"),
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


def _resolve_customer_profile(state: AgentState, customer_result: dict, agent_outputs: dict) -> dict:
    customer_profile = state.get("customer_profile")
    if customer_profile and not _looks_like_invalid_customer_profile(customer_profile):
        return customer_profile

    customer_profile_raw = _extract_tool_result(customer_result, "get_customer_profile")
    if not customer_profile_raw:
        customer_profile_raw = _extract_summary(agent_outputs.get("customer_agent"))
    customer_profile = parse_customer_profile(customer_profile_raw)

    if _looks_like_invalid_customer_profile(customer_profile):
        summary_profile = parse_customer_profile(_extract_summary(agent_outputs.get("customer_agent")))
        if not _looks_like_invalid_customer_profile(summary_profile):
            customer_profile = summary_profile

    return customer_profile


def _resolve_customer_accounts(state: AgentState, customer_result: dict, agent_outputs: dict) -> list[dict]:
    customer_accounts = state.get("customer_accounts")
    if customer_accounts:
        return customer_accounts

    customer_accounts_raw = _extract_tool_result(customer_result, "get_customer_accounts")
    if not customer_accounts_raw:
        customer_accounts_raw = _extract_summary(agent_outputs.get("customer_agent"))
    return parse_customer_accounts(customer_accounts_raw)


def _resolve_product_candidates(state: AgentState, product_result: dict, agent_outputs: dict) -> list[dict]:
    existing_candidates = state.get("product_candidates")
    if existing_candidates:
        cleaned = _clean_existing_product_candidates(existing_candidates)
        if cleaned:
            return cleaned

    result_candidates = _extract_product_candidates_from_result(product_result)
    cleaned_result_candidates = _clean_existing_product_candidates(result_candidates)
    if cleaned_result_candidates:
        return cleaned_result_candidates

    summary_candidates = extract_product_candidates(_extract_summary(agent_outputs.get("product_agent")))
    return _clean_existing_product_candidates(summary_candidates)


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


def _looks_like_invalid_customer_profile(profile: dict) -> bool:
    if not isinstance(profile, dict):
        return True

    job = str(profile.get("job") or "").strip().lower()
    invalid_jobs = {
        "",
        "annual_income",
        "income_level",
        "main_bank_yn",
        "salary_transfer_yn",
        "auto_transfer_yn",
        "card_usage_yn",
        "marketing_agree_yn",
        "transaction_months",
        "available_monthly_saving",
    }
    return job in invalid_jobs


def _clean_existing_product_candidates(product_candidates: list[dict]) -> list[dict]:
    cleaned = []
    seen = set()

    for item in product_candidates:
        if not isinstance(item, dict):
            continue

        product_name = str(item.get("product_name") or "").strip()
        if not product_name:
            continue

        cleaned_name = _strip_display_prefix(product_name)
        if _looks_like_suspicious_product_name(cleaned_name):
            continue

        normalized = _canonicalize_product_name(cleaned_name)
        if not normalized or normalized in seen:
            continue

        seen.add(normalized)
        cleaned.append({**item, "product_name": cleaned_name})

    return cleaned


def _looks_like_suspicious_product_name(product_name: str) -> bool:
    text = str(product_name or "").strip()
    if not text:
        return True
    if len(text) > 40:
        return True
    if any(mark in text for mark in ["|", ":", "->", "→", "▪", "•", "✅"]):
        return True
    if not any(keyword in text for keyword in ["적금", "예금", "부금", "통장", "저축"]):
        return True
    if text in GENERIC_PRODUCT_NAMES:
        return True
    if any(token in text for token in SUSPICIOUS_PRODUCT_TOKENS):
        return True

    weak_prefixes = [
        "전용",
        "우대",
        "군인 전용",
        "직업군인",
        "일반",
        "도약",
    ]
    if not text.startswith("KB") and any(text.startswith(prefix) for prefix in weak_prefixes):
        return True

    return False


def _canonicalize_product_name(product_name: str) -> str:
    normalized = _strip_display_prefix(product_name).lower()
    normalized = "".join(normalized.split())
    if normalized.startswith("kb"):
        normalized = normalized[2:]
    return normalized


def _strip_display_prefix(product_name: str) -> str:
    cleaned = str(product_name or "").strip()
    cleaned = cleaned.replace("1️⃣", "").replace("2️⃣", "").replace("3️⃣", "")
    cleaned = cleaned.replace("4️⃣", "").replace("5️⃣", "")
    cleaned = re.sub(r"^[✅☑✔•▪]+\s*", "", cleaned)
    cleaned = re.sub(r"^[①②③④⑤⑥⑦⑧⑨⑩]\s*", "", cleaned)
    cleaned = re.sub(r"^\d+[.)]\s*", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()
