"""
Recommend Agent 도구
- 추천 가능한 상품만 골라 점수를 매기고 요약 표를 만듭니다.
"""
import json
import re
from typing import Any

from langchain_core.tools import tool


def parse_financial_results(raw_financial: Any) -> list[dict]:
    # Financial Agent의 텍스트 결과를 상품별 계산 정보 목록으로 정리합니다.
    if isinstance(raw_financial, list):
        return raw_financial

    text = str(raw_financial or "")
    if not text.strip():
        return []

    results = []
    current_name = None
    current = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(keyword in stripped for keyword in ["KB", "적금", "예금", "부금"]):
            if current_name:
                results.append(current)
            current_name = stripped
            current = {"product_name": stripped}
        if "이자" in stripped and "예상" in stripped:
            amount = _extract_amount(stripped)
            if amount is not None:
                current["estimated_interest"] = amount
        if "만기" in stripped and ("수령" in stripped or "금액" in stripped):
            amount = _extract_amount(stripped)
            if amount is not None:
                current["maturity_amount"] = amount
        if "갈아타기" in stripped or "유지" in stripped:
            current["switch_comparison"] = stripped
    if current_name:
        results.append(current)
    return results


def classify_eligibility_results(eligibility_results: list[dict]) -> dict:
    # eligibility 결과를 추천 가능/추가 확인 필요/제외 대상으로 나눕니다.
    recommendable = []
    needs_check = []
    rejected = []

    for item in eligibility_results:
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
) -> list[dict]:
    # 추천 가능한 상품만 점수화해 recommendation_results 형태로 만듭니다.
    financial_map = {
        _normalize_name(item.get("product_name", "")): item for item in (financial_results or [])
    }

    recommendations = []
    recommendable_items = [
        item
        for item in eligibility_results
        if item.get("eligible") is True and not item.get("check_required")
    ]

    for item in recommendable_items:
        product_name = item.get("product_name", "미확인 상품")
        normalized_name = _normalize_name(product_name)
        financial = financial_map.get(normalized_name, {})

        score = 70
        score += len(item.get("bonus_conditions_met", [])) * 5
        score -= len(item.get("bonus_conditions_missing", [])) * 2

        estimated_interest = financial.get("estimated_interest")
        maturity_amount = financial.get("maturity_amount")
        switch_comparison = financial.get("switch_comparison", "")

        if estimated_interest:
            score += min(int(estimated_interest / 100000), 10)
        if maturity_amount:
            score += min(int(maturity_amount / 1000000), 10)
        if switch_comparison:
            if any(keyword in switch_comparison for keyword in ["유리", "이득", "추천"]):
                score += 5
            if any(keyword in switch_comparison for keyword in ["불리", "비추천", "손실"]):
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

    recommendations.sort(key=lambda item: item.get("score", 0), reverse=True)
    for index, item in enumerate(recommendations, 1):
        item["rank"] = index
    return recommendations


def build_recommendation_summary(
    recommendations: list[dict],
    needs_check: list[dict],
    rejected: list[dict],
) -> str:
    # 추천 표와 별도 안내 섹션을 합쳐 최종 요약 문자열을 만듭니다.
    lines = [
        "| 순위 | 상품명 | 점수 | 예상 이자 | 추천 이유 |",
        "|---:|---|---:|---:|---|",
    ]

    if recommendations:
        for item in recommendations:
            lines.append(
                "| {rank} | {name} | {score} | {interest} | {reason} |".format(
                    rank=item.get("rank", "-"),
                    name=_sanitize_cell(item.get("product_name", "-")),
                    score=item.get("score", "-"),
                    interest=_format_currency(item.get("estimated_interest")),
                    reason=_truncate_reason(item.get("reason", "")),
                )
            )
    else:
        lines.append("| - | 추천 가능한 상품 없음 | - | - | 조건 충족 상품 없음 |")

    if needs_check:
        lines.append("")
        lines.append("추가 확인 필요 상품:")
        for item in needs_check:
            reason = ", ".join(item.get("check_required", [])) or "상세 조건 확인 필요"
            lines.append(
                f"- {_sanitize_cell(item.get('product_name', '미확인 상품'))}: {_sanitize_cell(reason)}"
            )

    if rejected:
        lines.append("")
        lines.append("가입 불가/제외 상품:")
        for item in rejected:
            reason = ", ".join(item.get("ineligibility_reasons", [])) or "가입 불가"
            lines.append(
                f"- {_sanitize_cell(item.get('product_name', '미확인 상품'))}: {_sanitize_cell(reason)}"
            )

    return "\n".join(lines)


def _build_reason(eligibility_item: dict, financial: dict) -> str:
    # 추천 이유를 짧은 설명 문장으로 합칩니다.
    parts = []
    met = eligibility_item.get("bonus_conditions_met", [])
    missing = eligibility_item.get("bonus_conditions_missing", [])
    if met:
        parts.append(f"우대조건 충족 {', '.join(met)}")
    if missing:
        parts.append(f"추가 우대 여지 {', '.join(missing)}")
    if financial.get("estimated_interest") is not None:
        parts.append(f"예상 이자 {financial['estimated_interest']:,}원 반영")
    if financial.get("maturity_amount") is not None:
        parts.append(f"만기 금액 {financial['maturity_amount']:,}원 반영")
    if financial.get("switch_comparison"):
        parts.append(financial["switch_comparison"])
    if not parts:
        return "가입 가능, 기본 조건 충돌 없음"
    return "; ".join(parts)


def _sanitize_cell(value: Any) -> str:
    text = str(value or "")
    text = text.replace("\r", " ").replace("\n", " ").replace("|", "/")
    text = re.sub(r"\s+", " ", text).strip()
    return text or "-"


def _truncate_reason(reason: str) -> str:
    sanitized = _sanitize_cell(reason)
    if len(sanitized) <= 35:
        return sanitized
    return sanitized[:35].rstrip()


def _format_currency(amount: Any) -> str:
    if amount is None:
        return "-"
    try:
        return f"{int(amount):,}원"
    except (TypeError, ValueError):
        return _sanitize_cell(amount)


def _normalize_name(name: str) -> str:
    return re.sub(r"\s+", "", str(name or "")).lower()


def _extract_amount(text: str) -> int | None:
    match = re.search(r"([0-9][0-9,]*)원", text)
    if match:
        return int(match.group(1).replace(",", ""))
    return None


@tool
def rank_products(
    products_info: str = "",
    purpose: str = "",
    period_months: int = 0,
    monthly_amount: int = 0,
    eligibility_results: str = "",
    financial_results: str = "",
) -> str:
    """추천 가능한 상품만 순위화하고 계산 결과가 있으면 함께 반영합니다."""
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
