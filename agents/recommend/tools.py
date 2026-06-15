"""
Recommend agent tools.
"""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.tools import tool


PRODUCT_SUFFIXES = ("적금", "예금", "부금", "통장", "저축")

GENERIC_EXACT_NAMES = {
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
}

GENERIC_CONTAINS = [
    "추천 이유",
    "가입 가능",
    "기본 조건 충돌 없음",
    "우대조건 충족",
    "다음 단계",
    "최종 추천",
    "자동이체 설정",
    "가입 신청",
    "배분하면",
    "확인해 보세요",
    "선택하세요",
    "상품 유형",
    "유형 ",
]

SENTENCE_LIKE_PATTERNS = [
    "언제든 말씀",
    "말씀해 주세요",
    "도와드릴",
    "확인해 보세요",
    "선택하시면",
    "원하시면",
    "있으면",
    "좋습니다",
    "드립니다",
    "알려 주시면",
    "안내해 드릴",
    "가입 신청",
    "배분하면",
]


def parse_financial_results(raw_financial: Any) -> list[dict]:
    if isinstance(raw_financial, list):
        return [item for item in raw_financial if isinstance(item, dict)]

    text = str(raw_financial or "").strip()
    if not text:
        return []

    results: list[dict] = []
    current: dict[str, Any] | None = None

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        if _looks_like_actual_product_name(stripped):
            if current:
                results.append(current)
            current = {"product_name": _clean_product_name(stripped)}
            continue

        if current is None:
            continue

        if "예상 이자" in stripped:
            amount = _extract_amount(stripped)
            if amount is not None:
                current["estimated_interest"] = amount

        if "만기" in stripped and any(keyword in stripped for keyword in ["수령", "금액", "예상"]):
            amount = _extract_amount(stripped)
            if amount is not None:
                current["maturity_amount"] = amount

        if any(keyword in stripped for keyword in ["갈아타기", "유리", "불리", "전환"]):
            current["switch_comparison"] = stripped

    if current:
        results.append(current)

    return results


def classify_eligibility_results(eligibility_results: list[dict]) -> dict:
    recommendable = []
    needs_check = []
    rejected = []

    for item in eligibility_results:
        if not isinstance(item, dict):
            rejected.append(
                {
                    "product_name": str(item),
                    "eligible": False,
                    "ineligibility_reasons": ["결과 형식 오류"],
                }
            )
            continue

        if item.get("eligible") is True and not item.get("check_required"):
            recommendable.append(item)
        elif item.get("eligible") is True and item.get("check_required"):
            needs_check.append(item)
        else:
            rejected.append(item)

    return {
        "recommendable": recommendable,
        "needs_check": needs_check,
        "rejected": rejected,
    }


def build_recommendations(
    eligibility_results: list[dict],
    financial_results: list[dict] | None = None,
    customer_profile: dict | None = None,
) -> list[dict]:
    financial_map = {
        _normalize_name(item.get("product_name", "")): item
        for item in (financial_results or [])
        if isinstance(item, dict)
    }

    recommendable_items = [
        item
        for item in eligibility_results
        if isinstance(item, dict)
        and item.get("eligible") is True
        and not item.get("check_required")
    ]
    recommendable_items = _dedupe_recommendable_items(recommendable_items)

    recommendations = []
    for item in recommendable_items:
        product_name = _clean_product_name(str(item.get("product_name") or "미확인 상품"))
        if not _looks_like_actual_product_name(product_name):
            continue

        financial = financial_map.get(_normalize_name(product_name), {})

        score = 70
        score += len(item.get("bonus_conditions_met", [])) * 5
        score -= len(item.get("bonus_conditions_missing", [])) * 2

        if product_name.startswith("KB"):
            score += 3
        if customer_profile and customer_profile.get("is_soldier") and item.get("military_only"):
            score += 12
        if customer_profile and customer_profile.get("is_miso_target") and item.get("miso_dream_only"):
            score += 10

        estimated_interest = financial.get("estimated_interest")
        maturity_amount = financial.get("maturity_amount")
        switch_comparison = str(financial.get("switch_comparison") or "")

        if estimated_interest:
            score += min(int(estimated_interest / 100000), 10)
        if maturity_amount:
            score += min(int(maturity_amount / 1000000), 10)
        if switch_comparison:
            if any(keyword in switch_comparison for keyword in ["유리", "이득", "추천"]):
                score += 5
            if any(keyword in switch_comparison for keyword in ["불리", "손해", "비추천"]):
                score -= 5

        recommendation = {
            "product_name": product_name,
            "eligible": True,
            "score": score,
            "reason": _build_reason(item, financial),
        }
        if estimated_interest is not None:
            recommendation["estimated_interest"] = estimated_interest
        if maturity_amount is not None:
            recommendation["maturity_amount"] = maturity_amount
        if switch_comparison:
            recommendation["switch_comparison"] = switch_comparison

        recommendations.append(recommendation)

    recommendations.sort(
        key=lambda item: (
            item.get("score", 0),
            1 if str(item.get("product_name", "")).startswith("KB") else 0,
            len(str(item.get("product_name", ""))),
        ),
        reverse=True,
    )

    for index, item in enumerate(recommendations, start=1):
        item["rank"] = index

    return recommendations


def build_recommendation_summary(
    recommendations: list[dict],
    needs_check: list[dict],
    rejected: list[dict],
) -> str:
    lines = ["추천 가능한 상품을 정리했습니다."]

    if recommendations:
        lines.append("")
        lines.append("추천 상품:")
        for item in recommendations[:5]:
            name = _sanitize_text(item.get("product_name"))
            score = item.get("score", "-")
            reason = _sanitize_text(item.get("reason"))
            lines.append(f"- {name} (점수 {score})")
            if reason:
                lines.append(f"  이유: {reason}")
            if item.get("estimated_interest") is not None:
                lines.append(f"  예상 이자: {_format_currency(item.get('estimated_interest'))}")
    else:
        lines.append("")
        lines.append("추천 가능한 상품이 없습니다.")

    filtered_needs_check = [
        item
        for item in needs_check
        if isinstance(item, dict) and _looks_like_actual_product_name(str(item.get("product_name") or ""))
    ]
    if filtered_needs_check:
        lines.append("")
        lines.append("추가 확인 필요 상품:")
        for item in filtered_needs_check[:5]:
            name = _sanitize_text(item.get("product_name"))
            reason = ", ".join(item.get("check_required", [])) or "상세 조건 확인 필요"
            lines.append(f"- {name}: {_sanitize_text(reason)}")

    filtered_rejected = [
        item
        for item in rejected
        if isinstance(item, dict) and _looks_like_actual_product_name(str(item.get("product_name") or ""))
    ]
    if filtered_rejected:
        lines.append("")
        lines.append("가입 불가 또는 제외 상품:")
        for item in filtered_rejected[:5]:
            name = _sanitize_text(item.get("product_name"))
            reason = ", ".join(item.get("ineligibility_reasons", [])) or "가입 불가"
            lines.append(f"- {name}: {_sanitize_text(reason)}")

    return "\n".join(lines)


def _build_reason(eligibility_item: dict, financial: dict) -> str:
    parts = []
    met = eligibility_item.get("bonus_conditions_met", [])
    missing = eligibility_item.get("bonus_conditions_missing", [])

    if met:
        parts.append(f"우대조건 충족: {', '.join(met)}")
    if missing:
        parts.append(f"추가 우대 여지: {', '.join(missing)}")
    if financial.get("estimated_interest") is not None:
        parts.append(f"예상 이자 {_format_currency(financial['estimated_interest'])}")
    if financial.get("maturity_amount") is not None:
        parts.append(f"만기 금액 {_format_currency(financial['maturity_amount'])}")
    if financial.get("switch_comparison"):
        parts.append(str(financial["switch_comparison"]))

    if not parts:
        return "가입 가능하고 기본 조건 충돌이 없습니다."

    return "; ".join(parts)


def _sanitize_text(value: Any) -> str:
    text = str(value or "")
    text = text.replace("\r", " ").replace("\n", " ").replace("|", "/")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _format_currency(amount: Any) -> str:
    if amount is None:
        return "-"
    try:
        return f"{int(amount):,}원"
    except (TypeError, ValueError):
        return _sanitize_text(amount)


def _extract_amount(text: str) -> int | None:
    match = re.search(r"([0-9][0-9,]*)\s*원", str(text))
    if match:
        return int(match.group(1).replace(",", ""))
    return None


def _normalize_name(name: str) -> str:
    text = _clean_product_name(name)
    text = re.sub(r"\((?:[^()]*(?:개월|우대|연\s*[0-9]+(?:\.[0-9]+)?\s*%)\s*[^()]*)\)$", "", text).strip()
    text = re.sub(r"\s+", "", text).lower()
    if text.startswith("kb"):
        text = text[2:]
    return text


def _dedupe_recommendable_items(items: list[dict]) -> list[dict]:
    best_by_name: dict[str, dict] = {}

    for item in items:
        product_name = _clean_product_name(str(item.get("product_name") or ""))
        normalized = _normalize_name(product_name)
        if not normalized or not _looks_like_actual_product_name(product_name):
            continue

        current = {**item, "product_name": product_name}
        previous = best_by_name.get(normalized)
        if previous is None or _is_better_product_name(product_name, str(previous.get("product_name") or "")):
            best_by_name[normalized] = current

    return list(best_by_name.values())


def _is_better_product_name(candidate: str, existing: str) -> bool:
    candidate_score = (
        1 if candidate.startswith("KB") else 0,
        0 if _looks_like_generic_name(candidate) else 1,
        len(candidate),
    )
    existing_score = (
        1 if existing.startswith("KB") else 0,
        0 if _looks_like_generic_name(existing) else 1,
        len(existing),
    )
    return candidate_score > existing_score


def _looks_like_generic_name(name: str) -> bool:
    text = _clean_product_name(name)
    if not text:
        return True
    if text in GENERIC_EXACT_NAMES:
        return True
    return any(token in text for token in GENERIC_CONTAINS)


def _looks_like_actual_product_name(name: str) -> bool:
    text = _clean_product_name(name)
    if not text or text == "미확인 상품":
        return False
    if len(text) > 40:
        return False
    if _looks_like_generic_name(text):
        return False
    if any(pattern in text for pattern in SENTENCE_LIKE_PATTERNS):
        return False
    if any(mark in text for mark in ["|", ":", "->"]):
        return False
    if any(token in text for token in ["추천", "이유", "가입 신청", "설정", "배분", "확인", "선택"]):
        return False
    if not any(text.endswith(suffix) or suffix in text for suffix in PRODUCT_SUFFIXES):
        return False
    if _looks_like_weak_generic_product(text):
        return False
    return True


def _looks_like_weak_generic_product(name: str) -> bool:
    text = re.sub(r"\s+", " ", name).strip()
    generic_prefixes = [
        "전용",
        "우대",
        "군인 전용",
        "직업군인",
        "일반",
        "청약",
    ]
    if text.startswith("KB "):
        return False
    return any(text.startswith(prefix) for prefix in generic_prefixes)


def _clean_product_name(name: str) -> str:
    text = str(name or "").strip()
    text = re.sub(r"^[✅☑✔•▪]+\s*", "", text)
    text = re.sub(r"^\d+\.\s*", "", text)
    text = re.sub(r"^[0-9]+[.)]\s*", "", text)
    text = re.sub(r"^[①②③④⑤⑥⑦⑧⑨⑩]\s*", "", text)
    text = text.replace("1️⃣", "").replace("2️⃣", "").replace("3️⃣", "")
    text = text.replace("4️⃣", "").replace("5️⃣", "")
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


@tool
def rank_products(
    products_info: str = "",
    purpose: str = "",
    period_months: int = 0,
    monthly_amount: int = 0,
    eligibility_results: str = "",
    financial_results: str = "",
) -> str:
    """추천 가능한 상품을 순위화하고 계산 결과가 있으면 함께 반영합니다."""
    try:
        eligible_items = json.loads(eligibility_results) if eligibility_results else []
    except json.JSONDecodeError:
        eligible_items = []

    try:
        financial_items = json.loads(financial_results) if financial_results else []
    except json.JSONDecodeError:
        financial_items = parse_financial_results(financial_results)

    recommendations = build_recommendations(eligible_items, financial_items)
    return json.dumps(recommendations, ensure_ascii=False)


RECOMMEND_TOOLS = [rank_products]
