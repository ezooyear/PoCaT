"""
Financial 에이전트 - 이자 계산, 만기, 납입, 중도해지, 비교, 갈아타기
"""

import re
from datetime import date, datetime
from typing import Any

from graph.state import AgentState
from agents.base import run_agent_loop
from agents.financial.prompts import FINANCIAL_SYSTEM_PROMPT
from agents.financial.tools import FINANCIAL_TOOLS, compare_switch_benefit


DEFAULT_TAX_RATE = 0.154


def financial_agent_node(state: AgentState) -> dict:
    result = run_agent_loop(
        state=state,
        system_prompt=FINANCIAL_SYSTEM_PROMPT,
        tools=FINANCIAL_TOOLS,
        output_key="financial_agent",
        result_key="financial_result", # 추가 : validation에서 활용
        max_iterations=5,
    )

    if state.get("task_type") == "switch_analysis":
        result = _augment_switch_analysis_result(state, result)

    result = _augment_maturity_estimate_result(state, result)
    result = _enforce_financial_role_boundary(result)
    result = _augment_recommendation_calculations(state, result)

    return result


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


def _augment_recommendation_calculations(state: AgentState, result: dict) -> dict:
    """recommendation 태스크에서 eligible 상품별 구조화된 계산값을 financial_result에 추가한다."""
    if str(state.get("task_type") or "") != "recommendation":
        return result

    financial_result = result.get("financial_result")
    if not isinstance(financial_result, dict):
        return result

    # 이미 calculations가 있으면 중복 추가하지 않음
    payload = financial_result.get("result", {})
    if isinstance(payload, dict) and isinstance(payload.get("calculations"), list) and payload["calculations"]:
        return result

    eligible_products = _extract_eligible_products(state)
    if not eligible_products:
        return result

    product_map = _build_product_map(state)
    monthly_amount = _extract_monthly_saving_amount(state)

    calculations = []
    for ep in eligible_products:
        if not isinstance(ep, dict):
            continue
        product_name = str(ep.get("product_name") or "").strip()
        if not product_name:
            continue

        product = _find_matching_product(product_name, product_map)
        calc = _compute_product_estimate(product_name, product, monthly_amount)
        if calc:
            calculations.append(calc)

    if not calculations:
        return result

    if not isinstance(payload, dict):
        payload = {}

    payload["calculations"] = calculations
    payload["financial_calculation_count"] = len(calculations)
    payload["financial_calculation_product_names"] = [c["product_name"] for c in calculations]
    financial_result["result"] = payload
    financial_result["calculations"] = calculations
    financial_result["financial_calculation_count"] = len(calculations)
    financial_result["financial_calculation_product_names"] = [c["product_name"] for c in calculations]
    result["financial_result"] = financial_result

    agent_outputs = dict(result.get("agent_outputs") or {})
    if isinstance(agent_outputs.get("financial_agent"), dict):
        agent_outputs["financial_agent"]["result"] = payload
        agent_outputs["financial_agent"]["calculations"] = calculations
        agent_outputs["financial_agent"]["financial_calculation_count"] = len(calculations)
        agent_outputs["financial_agent"]["financial_calculation_product_names"] = [
            c["product_name"] for c in calculations
        ]
    result["agent_outputs"] = agent_outputs

    return result


def _extract_eligible_products(state: AgentState) -> list[dict]:
    eligibility_result = state.get("eligibility_result") or {}
    if not isinstance(eligibility_result, dict):
        return []
    payload = eligibility_result.get("result", {})
    if not isinstance(payload, dict):
        return []
    return list(payload.get("eligible_products") or [])


def _build_product_map(state: AgentState) -> dict[str, dict]:
    product_result = state.get("product_result") or {}
    products: list[dict] = []
    if isinstance(product_result, dict):
        products = (
            product_result.get("products")
            or (product_result.get("result") or {}).get("products")
            or []
        )
    return {
        _norm(str(p.get("product_name") or "")): p
        for p in products
        if isinstance(p, dict) and p.get("product_name")
    }


def _extract_monthly_saving_amount(state: AgentState) -> int:
    customer_profile = state.get("customer_profile") or {}
    if not isinstance(customer_profile, dict):
        customer_profile = {}
    amount = customer_profile.get("monthly_saving_amount")
    if amount:
        return int(amount)

    customer_result = state.get("customer_result") or {}
    payload = customer_result.get("result", {}) if isinstance(customer_result, dict) else {}
    profile = (
        payload.get("customer_profile") or payload.get("profile") or {}
        if isinstance(payload, dict) else {}
    )
    amount = profile.get("monthly_saving_amount") if isinstance(profile, dict) else None
    return int(amount) if amount else 100000


def _find_matching_product(product_name: str, product_map: dict[str, dict]) -> dict:
    norm = _norm(product_name)
    if norm in product_map:
        return product_map[norm]
    # 괄호 제거 alias 매칭
    alias = re.sub(r"\([^)]*\)", "", norm).strip()
    if alias:
        for key, product in product_map.items():
            key_alias = re.sub(r"\([^)]*\)", "", key).strip()
            if alias == key_alias or key.startswith(alias) or alias.startswith(key):
                return product
    return {}


def _compute_product_estimate(product_name: str, product: dict, monthly_amount: int) -> dict | None:
    product_type = str(product.get("product_type") or "적금")
    applied_rate = product.get("max_rate") or product.get("base_rate")
    base_rate = product.get("base_rate")
    term_months = _to_int(product.get("term_months")) or 12

    if applied_rate is None:
        applied_rate = 2.0
    applied_rate = float(applied_rate)
    base_rate_val = float(base_rate) if base_rate is not None else None

    rate = applied_rate / 100

    if "예금" in product_type:
        principal = monthly_amount * term_months
        before_tax = principal * rate * term_months / 12
        tax = before_tax * DEFAULT_TAX_RATE
        after_tax = before_tax - tax
        maturity = principal + after_tax
        return {
            "product_name": product_name,
            "product_type": product_type,
            "monthly_amount": monthly_amount,
            "term_months": term_months,
            "payment_count": 1,
            "principal": round(principal),
            "applied_rate": applied_rate,
            "base_rate": base_rate_val,
            "bonus_rate": round(applied_rate - base_rate_val, 2) if base_rate_val is not None else None,
            "estimated_interest_before_tax": round(before_tax),
            "estimated_interest_after_tax": round(after_tax),
            "estimated_maturity_amount": round(maturity),
            "estimated_interest": round(after_tax),
            "maturity_amount": round(maturity),
            "calculation_method": "lump_sum_simple_estimate",
            "calculation_note": "단리 기준 단순 추정치이며 실제 은행 계산 방식과 다를 수 있습니다.",
        }
    else:
        total_payment = monthly_amount * term_months
        before_tax = sum(
            monthly_amount * rate * (term_months - i) / 12
            for i in range(term_months)
        )
        tax = before_tax * DEFAULT_TAX_RATE
        after_tax = before_tax - tax
        maturity = total_payment + after_tax
        return {
            "product_name": product_name,
            "product_type": product_type,
            "monthly_amount": monthly_amount,
            "term_months": term_months,
            "payment_count": term_months,
            "principal": round(total_payment),
            "applied_rate": applied_rate,
            "base_rate": base_rate_val,
            "bonus_rate": round(applied_rate - base_rate_val, 2) if base_rate_val is not None else None,
            "estimated_interest_before_tax": round(before_tax),
            "estimated_interest_after_tax": round(after_tax),
            "estimated_maturity_amount": round(maturity),
            "estimated_interest": round(after_tax),
            "maturity_amount": round(maturity),
            "calculation_method": "monthly_installment_simple_estimate",
            "calculation_note": "단리 기준 단순 추정치이며 실제 은행 계산 방식과 다를 수 있습니다.",
        }


def _norm(name: str) -> str:
    return re.sub(r"\s+", "", str(name or "")).lower()


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
        "?뺣낫",
        "遺議",
        "?뚮젮",
    ]
    return any(marker in str(summary or "") for marker in missing_markers)
