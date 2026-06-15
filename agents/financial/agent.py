"""
Financial agent - interest, maturity, early termination, switch analysis.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

from agents.base import run_agent_loop
from agents.financial.prompts import FINANCIAL_SYSTEM_PROMPT
from agents.financial.tools import FINANCIAL_TOOLS, compare_switch_benefit
from graph.state import AgentState


def financial_agent_node(state: AgentState) -> dict:
    task_type = state.get("task_type")
    max_iterations = 2 if task_type in {"financial_analysis", "switch_analysis", "early_termination"} else 1

    result = run_agent_loop(
        state=state,
        system_prompt=FINANCIAL_SYSTEM_PROMPT,
        tools=FINANCIAL_TOOLS,
        output_key="financial_agent",
        result_key="financial_result",
        max_iterations=max_iterations,
    )

    if task_type == "switch_analysis":
        result = _augment_switch_analysis_result(state, result)

    return result


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
    payload["summary"] = switch_result

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

    query = re.sub(r"\s+", "", user_query)
    candidates = rows

    if "적금" in query:
        savings = [
            row for row in candidates
            if "적금" in str(row.get("product_type", "")) or "적금" in str(row.get("product_name", ""))
        ]
        if savings:
            candidates = savings
    elif "예금" in query:
        deposits = [
            row for row in candidates
            if "예금" in str(row.get("product_type", "")) or "예금" in str(row.get("product_name", ""))
        ]
        if deposits:
            candidates = deposits

    active_rows = [row for row in candidates if str(row.get("account_status", "")).upper() == "ACTIVE"]
    matured_rows = [row for row in candidates if str(row.get("account_status", "")).upper() == "MATURED"]
    ordered = active_rows or matured_rows or candidates

    ordered.sort(
        key=lambda row: (
            1 if str(row.get("account_status", "")).upper() == "ACTIVE" else 0,
            _to_int(row.get("current_balance")) or 0,
            _to_int(row.get("monthly_amount")) or 0,
        ),
        reverse=True,
    )
    return ordered[0] if ordered else None


def _pick_target_product(product_candidates: list[dict[str, Any]], user_query: str) -> dict[str, Any] | None:
    if not product_candidates:
        return None

    query = re.sub(r"\s+", "", user_query).lower()

    explicit_matches = []
    for candidate in product_candidates:
        name = str(candidate.get("product_name", ""))
        normalized = re.sub(r"\s+", "", name).lower()
        if normalized and normalized in query:
            explicit_matches.append(candidate)

    if explicit_matches:
        explicit_matches.sort(key=lambda item: len(str(item.get("product_name", ""))), reverse=True)
        return _enrich_product_candidate(explicit_matches[0])

    if "일반정기적금" in query:
        for candidate in product_candidates:
            if "일반정기적금" in str(candidate.get("product_name", "")):
                return _enrich_product_candidate(candidate)

    if "적금" in query:
        for candidate in product_candidates:
            if "적금" in str(candidate.get("product_name", "")):
                return _enrich_product_candidate(candidate)

    if "예금" in query:
        for candidate in product_candidates:
            if "예금" in str(candidate.get("product_name", "")):
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
        if parts and parts[0] == "":
            parts = parts[1:]
        if parts and parts[-1] == "":
            parts = parts[:-1]

        if not header and "customer_id" in line.lower():
            header = parts
            continue

        if header and all(part and set(part) <= {"-", ":"} for part in parts):
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
