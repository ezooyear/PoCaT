"""
Recommend Agent 도구
- 추천 가능한 상품만 골라 점수를 매기고 요약 표를 만듭니다.
- Financial Agent가 계산한 월 납입액/기간/원금/이자/만기금액을 최종 답변에 명확히 표시합니다.
"""
import json
import re
from typing import Any

from langchain_core.tools import tool


def parse_financial_results(raw_financial: Any) -> list[dict]:
    """Financial Agent의 결과를 상품별 계산 정보 목록으로 정리합니다."""
    if isinstance(raw_financial, list):
        return raw_financial

    text = str(raw_financial or "")
    if not text.strip():
        return []

    results: list[dict[str, Any]] = []
    current_name = None
    current: dict[str, Any] = {}

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        if any(keyword in stripped for keyword in ["KB", "적금", "예금", "부금", "정기"]):
            if current_name:
                results.append(current)
            current_name = stripped
            current = {"product_name": stripped}

        if "월" in stripped and ("납입" in stripped or "저축" in stripped):
            amount = _extract_amount(stripped)
            if amount is not None:
                current["monthly_amount"] = amount

        if "개월" in stripped and any(keyword in stripped for keyword in ["기간", "납입", "가입"]):
            months = _extract_months(stripped)
            if months is not None:
                current["term_months"] = months

        if "원금" in stripped or "총 납입" in stripped:
            amount = _extract_amount(stripped)
            if amount is not None:
                current["total_principal"] = amount

        if "이자" in stripped and "예상" in stripped:
            amount = _extract_amount(stripped)
            if amount is not None:
                current["estimated_interest"] = amount

        if "만기" in stripped and ("수령" in stripped or "금액" in stripped or "예상액" in stripped):
            amount = _extract_amount(stripped)
            if amount is not None:
                current["maturity_amount"] = amount

        if "갈아타기" in stripped or "유지" in stripped:
            current["switch_comparison"] = stripped

    if current_name:
        results.append(current)

    return results


def classify_eligibility_results(eligibility_results: list[dict]) -> dict:
    """eligibility 결과를 추천 가능/추가 확인 필요/제외 대상으로 나눕니다."""
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

        status = item.get("status")
        if item.get("eligible") is True and status == "eligible" and not item.get("check_required"):
            recommendable.append(item)
        elif item.get("eligible") is True or status == "needs_check":
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
    """추천 가능한 상품만 점수화해 recommendation_results 형태로 만듭니다."""
    financial_map = {
        _normalize_name(item.get("product_name", "")): item
        for item in (financial_results or [])
        if isinstance(item, dict)
    }

    recommendations: list[dict[str, Any]] = []
    recommendable_items = [
        item
        for item in eligibility_results
        if isinstance(item, dict)
        and item.get("eligible") is True
        and item.get("status", "eligible") == "eligible"
        and not item.get("check_required")
    ]

    for item in recommendable_items:
        product_name = item.get("product_name", "미확인 상품")
        normalized_name = _normalize_name(product_name)
        financial = financial_map.get(normalized_name, {})

        score = 70
        score += len(item.get("bonus_conditions_met", [])) * 5
        score -= len(item.get("bonus_conditions_missing", [])) * 2

        estimated_interest = _first_not_none(
            financial.get("estimated_interest"),
            financial.get("estimated_interest_after_tax"),
            financial.get("estimated_interest_before_tax"),
        )
        maturity_amount = _first_not_none(
            financial.get("maturity_amount"),
            financial.get("estimated_maturity_amount"),
        )
        monthly_amount = _first_not_none(
            financial.get("monthly_amount"),
            financial.get("monthly_payment"),
        )
        term_months = _first_not_none(
            financial.get("term_months"),
            financial.get("payment_count"),
            financial.get("months"),
        )
        total_principal = _first_not_none(
            financial.get("total_principal"),
            financial.get("principal"),
        )
        applied_rate = _first_not_none(
            financial.get("applied_rate"),
            financial.get("base_rate"),
            financial.get("annual_rate"),
        )
        switch_comparison = financial.get("switch_comparison", "")

        if estimated_interest:
            score += min(int(int(estimated_interest) / 100000), 10)
        if maturity_amount:
            score += min(int(int(maturity_amount) / 1000000), 10)
        if switch_comparison:
            if any(keyword in switch_comparison for keyword in ["유리", "이득", "추천"]):
                score += 5
            if any(keyword in switch_comparison for keyword in ["불리", "비추천", "손실"]):
                score -= 5

        payment_plan_text = financial.get("payment_plan_text") or _build_payment_plan_text(
            monthly_amount,
            term_months,
        )
        calculation_assumption = financial.get("calculation_assumption") or _build_calculation_assumption(
            financial,
            monthly_amount=monthly_amount,
            term_months=term_months,
        )

        recommendation = {
            "product_name": product_name,
            "eligible": True,
            "score": score,
            "reason": _build_reason(item, financial),
            "monthly_amount": monthly_amount,
            "term_months": term_months,
            "total_principal": total_principal,
            "applied_rate": applied_rate,
            "payment_plan_text": payment_plan_text,
            "calculation_assumption": calculation_assumption,
            "monthly_amount_source": financial.get("monthly_amount_source"),
            "monthly_amount_source_label": financial.get("monthly_amount_source_label"),
            "term_months_source": financial.get("term_months_source"),
            "term_months_source_label": financial.get("term_months_source_label"),
        }

        if estimated_interest is not None:
            recommendation["estimated_interest"] = int(estimated_interest)
        if maturity_amount is not None:
            recommendation["maturity_amount"] = int(maturity_amount)
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
    lines = [
        "| 순위 | 상품명 | 납입 기준 | 예상 세후 이자 | 만기 예상액 | 추천 포인트 |",
        "|---:|---|---|---:|---:|---|",
    ]

    if recommendations:
        for item in recommendations:
            lines.append(
                "| {rank} | {name} | {payment_basis} | {interest} | {maturity} | {reason} |".format(
                    rank=item.get("rank", "-"),
                    name=_sanitize_cell(item.get("product_name", "-")),
                    payment_basis=_sanitize_cell(_format_payment_basis(item)),
                    interest=_format_currency(item.get("estimated_interest")),
                    maturity=_format_currency(item.get("maturity_amount")),
                    reason=_truncate_reason(item.get("reason", "")),
                )
            )
    else:
        lines.append("| - | 바로 추천 가능한 상품 없음 | - | - | - | 현재 조건만으로는 확정 추천이 어렵습니다. |")

    if recommendations:
        lines.append("")
        lines.append(
            f"현재 조건에서 바로 추천드릴 수 있는 상품은 {len(recommendations)}개입니다. "
            "추가 확인이 필요한 상품과 이번 추천에서 제외한 상품은 아래에 따로 정리했습니다."
        )

    if needs_check:
        lines.append("")
        lines.append("가입 전 한 번 더 확인해 볼 상품:")
        for item in needs_check:
            reason = ", ".join(item.get("check_required", [])) or "세부 가입 조건 확인 필요"
            lines.append(
                f"- {_sanitize_cell(item.get('product_name', '미확인 상품'))}: "
                f"{_soften_needs_check_reason(reason)}"
            )

    if rejected:
        lines.append("")
        lines.append("이번 추천에서 제외한 상품:")
        for item in rejected:
            reason = ", ".join(item.get("ineligibility_reasons", [])) or "현재 조건과 맞지 않음"
            lines.append(
                f"- {_sanitize_cell(item.get('product_name', '미확인 상품'))}: "
                f"{_soften_rejected_reason(reason)}"
            )

    lines.append("")
    lines.append("참고")
    lines.append("- 월 납입액이나 가입 기간을 따로 입력하지 않은 경우, 현재 확인된 월 저축 가능액과 상품의 가입 가능 기간을 기준으로 계산했습니다.")
    lines.append("- 예상 이자와 만기 금액은 실제 가입 시점의 금리, 우대조건 충족 여부, 납입일에 따라 달라질 수 있습니다.")

    return "\n".join(lines)

def _build_reason(eligibility_item: dict, financial: dict) -> str:
    monthly_amount = financial.get("monthly_amount")
    term_months = financial.get("term_months")
    estimated_interest = financial.get("estimated_interest")
    maturity_amount = financial.get("maturity_amount")

    parts = []

    if monthly_amount is not None and term_months is not None:
        parts.append(
            f"월 {int(monthly_amount):,}원씩 {int(term_months)}개월 납입 기준으로 계산했습니다."
        )
    amount_cap_note = financial.get("amount_cap_note")
    if amount_cap_note:
        parts.append(str(amount_cap_note))

    if estimated_interest is not None and maturity_amount is not None:
        parts.append(
            f"예상 세후 이자는 약 {int(estimated_interest):,}원, 만기 예상액은 약 {int(maturity_amount):,}원입니다."
        )
    elif estimated_interest is not None:
        parts.append(f"예상 세후 이자는 약 {int(estimated_interest):,}원입니다.")
    elif maturity_amount is not None:
        parts.append(f"만기 예상액은 약 {int(maturity_amount):,}원입니다.")

    met = eligibility_item.get("bonus_conditions_met", [])
    if met:
        parts.append(f"현재 충족된 우대조건은 {', '.join(met)}입니다.")

    missing = eligibility_item.get("bonus_conditions_missing", [])
    if missing:
        parts.append(f"{', '.join(missing)} 조건을 충족하면 추가 우대 가능성이 있습니다.")

    if not parts:
        return "현재 확인된 가입 조건과 충돌이 없어 추천 후보로 판단했습니다."

    return " ".join(parts)

def _format_payment_basis(item: dict) -> str:
    monthly_amount = item.get("monthly_amount")
    term_months = item.get("term_months")

    if monthly_amount is not None and term_months is not None:
        return f"월 {int(monthly_amount):,}원 × {int(term_months)}개월"

    if monthly_amount is not None:
        return f"월 {int(monthly_amount):,}원 기준"

    if term_months is not None:
        return f"{int(term_months)}개월 기준"

    return "-"


def _soften_needs_check_reason(reason: str) -> str:
    text = str(reason or "")

    if "상품 가입조건을 충분히 확인할 수 없어" in text:
        return "가입 대상이나 세부 조건이 더 확인되어야 해서, 이번에는 확정 추천에는 넣지 않았습니다."

    if "product_target" in text or "product_job_condition" in text:
        return "가입 대상이나 직업 조건을 한 번 더 확인해야 합니다."

    return _sanitize_cell(text)


def _soften_rejected_reason(reason: str) -> str:
    text = str(reason or "")

    if "직업군인 전용 상품 가입 대상이 아닙니다" in text:
        return "직업군인 전용 상품으로 확인되어, 현재 고객님의 직업 조건과 맞지 않아 추천에서 제외했습니다."

    if "고객 직업이" in text and "직업군인" in text:
        return "현재 고객님의 직업 조건과 맞지 않아 추천에서 제외했습니다."

    return _sanitize_cell(text)


def _build_payment_plan_text(monthly_amount: Any, term_months: Any) -> str:
    if monthly_amount is None and term_months is None:
        return "-"
    if monthly_amount is not None and term_months is not None:
        return f"월 {int(monthly_amount):,}원 × {int(term_months)}개월"
    if monthly_amount is not None:
        return f"월 {int(monthly_amount):,}원"
    return f"{int(term_months)}개월"


def _build_calculation_assumption(
    financial: dict,
    *,
    monthly_amount: Any = None,
    term_months: Any = None,
) -> str:
    parts = []

    monthly_label = financial.get("monthly_amount_source_label") or financial.get("monthly_amount_source")
    term_label = financial.get("term_months_source_label") or financial.get("term_months_source")

    if monthly_amount is not None:
        if monthly_label:
            parts.append(str(monthly_label))
        else:
            parts.append("고객 DB 월 저축 가능액 기준")

    if term_months is not None:
        if term_label:
            parts.append(str(term_label))
        else:
            parts.append("상품 가입 가능 기간 기준")

    return ", ".join(_dedupe([part for part in parts if part]))


def _build_payment_basis_for_summary(item: dict) -> str:
    plan = item.get("payment_plan_text") or _build_payment_plan_text(
        item.get("monthly_amount"),
        item.get("term_months"),
    )
    assumption = item.get("calculation_assumption")

    if plan and plan != "-" and assumption:
        return f"{plan} ({assumption})"
    if plan:
        return plan
    return "-"


def _sanitize_cell(value: Any) -> str:
    text = str(value or "")
    text = text.replace("\r", " ").replace("\n", " ").replace("|", "/")
    text = re.sub(r"\s+", " ", text).strip()
    return text or "-"


def _truncate_reason(reason: str) -> str:
    sanitized = _sanitize_cell(reason)
    if len(sanitized) <= 60:
        return sanitized
    return sanitized[:60].rstrip() + "..."

def _format_currency(amount: Any) -> str:
    if amount is None:
        return "-"
    try:
        return f"{int(float(amount)):,}원"
    except (TypeError, ValueError):
        return _sanitize_cell(amount)


def _normalize_name(name: str) -> str:
    return re.sub(r"\s+", "", str(name or "")).lower()


def _extract_amount(text: str) -> int | None:
    match = re.search(r"([0-9][0-9,]*)원", text)
    if match:
        return int(match.group(1).replace(",", ""))
    return None


def _extract_months(text: str) -> int | None:
    match = re.search(r"([0-9]{1,3})\s*개월", text)
    if match:
        return int(match.group(1))
    return None


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    output = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


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
