"""Simple financial analysis helpers for product comparisons and early-withdrawal estimates.

These are approximate and intended for demo/testing. Replace with
bank-grade calculations when available.
"""
from datetime import datetime, date
from typing import Dict


def _parse_date(d: str):
    return datetime.strptime(d, "%Y-%m-%d").date()


def estimate_term_years(product: Dict) -> float:
    if product.get("start_date") and product.get("end_date"):
        start = _parse_date(product["start_date"])
        end = _parse_date(product["end_date"])
        return max((end - start).days / 365.0, 0.0)
    total = product.get("total_count") or 0
    return (total / 12.0) if total else 0.0


def estimate_remaining_years(product: Dict) -> float:
    today = date.today()
    if product.get("end_date"):
        end = _parse_date(product["end_date"])
        return max((end - today).days / 365.0, 0.0)
    term_years = estimate_term_years(product)
    paid = product.get("paid_count") or 0
    return max(term_years - paid / 12.0, 0.0)


def estimate_deposit_maturity(product: Dict) -> float:
    """Estimate maturity amount for a product.

    - 정기예금: principal * (1 + rate_total * term_years)
    - 적금: monthly_payment * total_count (approx) + simple interest
    """
    rate = (product.get("rate", 0) + product.get("bonus_rate", 0)) / 100.0
    if product.get("type") == "정기예금":
        principal = product.get("principal", 0)
        if product.get("start_date") and product.get("end_date"):
            start = _parse_date(product["start_date"]) 
            end = _parse_date(product["end_date"]) 
            years = max((end - start).days / 365.0, 0.0)
        else:
            years = 1.0
        return principal * (1 + rate * years)

    # 적금
    monthly = product.get("monthly_payment", 0)
    total = product.get("total_count", 0) or 0
    # naive interest: apply avg balance factor 0.5 over term
    if product.get("start_date") and product.get("end_date"):
        start = _parse_date(product["start_date"]) 
        end = _parse_date(product["end_date"]) 
        years = max((end - start).days / 365.0, 0.0)
    else:
        years = (total / 12.0) if total else 1.0
    principal_sum = monthly * total
    interest = principal_sum * rate * 0.5 * years
    return principal_sum + interest


def estimate_early_withdrawal(product: Dict) -> float:
    """Estimate amount received if withdrawn today.

    Very simplified:
    - 정기예금: principal + accrued interest proportional to elapsed time
    - 적금: paid_count * monthly_payment + small accrued interest
    """
    rate = (product.get("rate", 0) + product.get("bonus_rate", 0)) / 100.0
    today = datetime.today().date()

    if product.get("type") == "정기예금":
        principal = product.get("principal", 0)
        start = _parse_date(product["start_date"]) if product.get("start_date") else today
        end = _parse_date(product["end_date"]) if product.get("end_date") else today
        term_days = max((end - start).days, 1)
        elapsed_days = max((today - start).days, 0)
        accrued = principal * rate * (elapsed_days / term_days)
        # apply simple penalty factor (e.g., lose 50% of accrued interest)
        penalty_accrued = accrued * 0.5
        return principal + penalty_accrued

    # 적금
    paid = product.get("paid_count", 0) or 0
    monthly = product.get("monthly_payment", 0)
    principal_paid = paid * monthly
    # small interest on paid installments (avg duration ~ paid/2 months)
    avg_months = paid / 2.0
    interest = principal_paid * rate * (avg_months / 12.0)
    # assume bank pays only part of interest on early withdrawal
    return principal_paid + interest * 0.5


def compare_early_withdrawal(products: list) -> list:
    """Return list of (product_name, maturity_est, early_est, loss) for comparison."""
    rows = []
    for p in products:
        mat = estimate_deposit_maturity(p)
        early = estimate_early_withdrawal(p)
        loss = mat - early
        rows.append({
            "name": p.get("name"),
            "type": p.get("type"),
            "rate": p.get("rate", 0),
            "bonus_rate": p.get("bonus_rate", 0),
            "maturity": mat,
            "early": early,
            "loss": loss,
            "remaining_term_years": estimate_term_years(p),
        })
    # sort by loss ascending: 가장 손실이 작은 상품 우선
    return sorted(rows, key=lambda r: r["loss"])


def compare_products(products: list) -> list:
    """Return a structured comparison table for available products."""
    table = []
    for p in products:
        term_years = estimate_term_years(p)
        paid = p.get("paid_count") or 0
        total = p.get("total_count") or 0
        paid_percent = (paid / total * 100) if total else None
        table.append({
            "name": p.get("name"),
            "type": p.get("type"),
            "rate": p.get("rate", 0),
            "bonus_rate": p.get("bonus_rate", 0),
            "total_term_years": round(term_years, 2),
            "remaining_term_years": round(estimate_remaining_years(p), 2),
            "paid_ratio": f"{paid_percent:.0f}%" if paid_percent is not None else "-",
            "maturity_est": round(estimate_deposit_maturity(p), 0),
            "early_est": round(estimate_early_withdrawal(p), 0),
        })
    return table


def evaluate_switch_benefit(current_products: list, new_product: dict) -> dict:
    """Compare current portfolio with a candidate new product for 갈아타기 판단."""
    current_value = sum(estimate_early_withdrawal(p) for p in current_products)
    new_maturity = estimate_deposit_maturity(new_product)
    return {
        "current_early_cash": current_value,
        "new_maturity_est": new_maturity,
        "difference": new_maturity - current_value,
        "switch_advantage": new_maturity > current_value,
    }
