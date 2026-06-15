"""
Product agent wrapper.
"""
from __future__ import annotations

import re

from agents.base import run_agent_loop
from agents.product.prompts import PRODUCT_SYSTEM_PROMPT
from agents.product.tools import PRODUCT_TOOLS, extract_product_candidates_from_search_results, search_terms
from graph.state import AgentState


def product_agent_node(state: AgentState) -> dict:
    task_type = state.get("task_type")
    max_iterations = 1 if task_type in {"recommendation", "eligibility_check"} else 2

    result = run_agent_loop(
        state=state,
        system_prompt=PRODUCT_SYSTEM_PROMPT,
        tools=PRODUCT_TOOLS,
        output_key="product_agent",
        result_key="product_result",
        max_iterations=max_iterations,
    )

    product_result = result.get("product_result") or {}
    payload = product_result.get("result", {}) if isinstance(product_result, dict) else {}
    tool_results = payload.get("tool_results", []) if isinstance(payload, dict) else []

    raw_search_results = [
        item.get("tool_result")
        for item in tool_results
        if isinstance(item, dict) and isinstance(item.get("tool_result"), str)
    ]

    augmented_search_result = _run_augmented_product_search(state)
    if augmented_search_result:
        raw_search_results.insert(0, augmented_search_result)

    product_candidates = extract_product_candidates_from_search_results(raw_search_results)

    summary_text = payload.get("summary", "") if isinstance(payload, dict) else ""
    summary_candidates = extract_product_candidates_from_search_results(summary_text)

    if task_type in {"recommendation", "switch_analysis", "early_termination", "eligibility_check"}:
        primary_candidates = product_candidates or summary_candidates
        product_candidates = _merge_product_candidates(primary_candidates)
    else:
        product_candidates = _merge_product_candidates(summary_candidates, product_candidates)

    if isinstance(product_result, dict) and isinstance(payload, dict):
        payload["product_candidates"] = product_candidates
        payload["searched_products"] = [item["product_name"] for item in product_candidates]
        product_result["result"] = payload
        result["product_result"] = product_result

    agent_outputs = dict(result.get("agent_outputs") or {})
    if isinstance(agent_outputs.get("product_agent"), dict):
        agent_outputs["product_agent"]["result"] = product_result.get("result", {})

    result["agent_outputs"] = agent_outputs
    result["product_candidates"] = product_candidates
    return result


def _run_augmented_product_search(state: AgentState) -> str:
    task_type = state.get("task_type")
    if task_type not in {"recommendation", "eligibility_check", "switch_analysis"}:
        return ""

    query = _build_augmented_query(state)
    if not query:
        return ""

    try:
        return str(search_terms.invoke({"query": query}) or "")
    except Exception:
        return ""


def _build_augmented_query(state: AgentState) -> str:
    user_query = str(state.get("user_query") or "").strip()
    customer_profile = state.get("customer_profile") or {}
    customer_result = state.get("customer_result") or {}

    signals = []

    job = str(customer_profile.get("job") or "")
    if not job:
        job = _extract_customer_signal(customer_result, "군인")
    if "군인" in job:
        signals.extend(["군인", "직업군인", "장병"])

    age = customer_profile.get("age")
    if isinstance(age, int) and 19 <= age <= 34:
        signals.append("청년")

    monthly_saving_amount = customer_profile.get("monthly_saving_amount")
    if isinstance(monthly_saving_amount, int) and monthly_saving_amount <= 300000:
        signals.append("적금")

    if not signals:
        return user_query

    suffix = " ".join(dict.fromkeys(signals))
    return f"{user_query} {suffix} KB국민은행 상품"


def _extract_customer_signal(customer_result: dict, fallback: str) -> str:
    if not isinstance(customer_result, dict):
        return fallback
    payload = customer_result.get("result", {})
    if not isinstance(payload, dict):
        return fallback
    summary = str(payload.get("summary") or "")
    return summary if summary else fallback


def _merge_product_candidates(*candidate_groups: list[dict]) -> list[dict]:
    merged = []
    seen = set()

    for group in candidate_groups:
        if not isinstance(group, list):
            continue
        for item in group:
            if not isinstance(item, dict):
                continue

            product_name = str(item.get("product_name") or "").strip()
            clean_name = _strip_display_prefix(product_name)
            if not _looks_like_good_product_name(clean_name):
                continue

            normalized = _canonicalize_product_name(clean_name)
            if not normalized or normalized in seen:
                continue

            seen.add(normalized)
            merged.append({**item, "product_name": clean_name})

    return merged[:5]


def _looks_like_good_product_name(product_name: str) -> bool:
    if not product_name:
        return False
    if len(product_name) > 40:
        return False

    suspicious_markers = [
        "상품군",
        "최적",
        "적합한",
        "추천",
        "추가 확인",
        "가입 불가",
        "함께 볼 상품",
        "가장 적합한",
        "고액예금",
        "본인 명의",
        "가입일",
        "급여",
        "개월",
        "우대 금리",
        "기본 금리",
        "계약기간",
        "납입",
        "추천 이유",
        "|",
        ":",
        "->",
        "입니다",
        "세후",
    ]
    if any(marker in product_name for marker in suspicious_markers):
        return False

    if product_name in {
        "미확인 상품",
        "KB 예적금 상품군",
        "최적의 적금",
        "고액예금",
        "KB 고액예금",
    }:
        return False

    return any(keyword in product_name for keyword in ["적금", "예금", "부금", "통장", "저축"])


def _canonicalize_product_name(product_name: str) -> str:
    normalized = str(product_name or "").strip().lower()
    normalized = re.sub(r"^\s*\d+[.)]\s*", "", normalized)
    normalized = re.sub(r"^[①②③④⑤⑥⑦⑧⑨⑩]+\s*", "", normalized)
    normalized = re.sub(r"\((?:[^()]*(?:개월|우대|연\s*[0-9]+(?:\.[0-9]+)?\s*%)\s*[^()]*)\)$", "", normalized)
    if normalized.startswith("kb"):
        normalized = normalized[2:]
    return "".join(normalized.split())


def _strip_display_prefix(product_name: str) -> str:
    cleaned = str(product_name or "").strip()
    cleaned = cleaned.replace("1️⃣", "").replace("2️⃣", "").replace("3️⃣", "")
    cleaned = cleaned.replace("4️⃣", "").replace("5️⃣", "")
    cleaned = re.sub(r"^[✅☑✔•▪]+\s*", "", cleaned)
    cleaned = re.sub(r"^[①②③④⑤⑥⑦⑧⑨⑩]\s*", "", cleaned)
    cleaned = re.sub(r"^\d+[.)]\s*", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()
