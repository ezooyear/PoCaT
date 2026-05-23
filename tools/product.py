"""Product helper tools

Contains lightweight helpers used by `product_agent`.
These are intentionally simple so they can be replaced later by
database-backed or policy-backed implementations (Postgres, etc.).
"""
from datetime import datetime, date
from typing import Tuple


def summarize_products(customer: dict, products: list) -> str:
    """Return a human-readable summary text for customer's products."""
    if not products:
        return "가입 상품이 없습니다."

    today = date.today()
    lines = []
    for i, prod in enumerate(products, 1):
        lines.append(f"{i}. {prod.get('name')} ({prod.get('type')})")
        if prod.get('type') == '정기예금':
            lines.append(f"   - 예치금액: {prod.get('principal', 0):,}원")
        else:
            lines.append(f"   - 월 납입액: {prod.get('monthly_payment', 0):,}원")
            if prod.get('paid_count') is not None and prod.get('total_count') is not None:
                lines.append(f"   - 납입 현황: {prod['paid_count']}/{prod['total_count']}회")
                lines.append(f"   - 남은 납입 횟수: {prod['total_count'] - prod['paid_count']}회")

        # rates
        lines.append(f"   - 기본금리: {prod.get('rate', 0)}%")
        if prod.get('bonus_rate', 0) > 0:
            lines.append(f"   - 우대금리: +{prod.get('bonus_rate')}%")

        # dates
        start = prod.get('start_date')
        end = prod.get('end_date')
        if start:
            lines.append(f"   - 가입일: {start}")
        if end:
            lines.append(f"   - 만기일: {end}")
            try:
                end_date = datetime.strptime(end, "%Y-%m-%d").date()
                days_left = (end_date - today).days
                if days_left > 0:
                    lines.append(f"   - 만기까지 남은 기간: {days_left}일")
                elif days_left == 0:
                    lines.append(f"   - ⚠️ 오늘 만기")
                else:
                    lines.append(f"   - 이미 만기됨 ({abs(days_left)}일 전)")
            except Exception:
                pass

        lines.append("")

    return "\n".join(lines)


def evaluate_eligibility(customer: dict, product: dict) -> Tuple[bool, str]:
    """A lightweight eligibility check.

    This is intentionally conservative and simplistic — replace with
    real business rules or a DB-backed policy later.

    Returns (can_subscribe, reason)
    """
    age = customer.get('age')
    min_age = product.get('min_age')
    max_age = product.get('max_age')

    if min_age is not None and age is not None and age < min_age:
        return False, f"최소 가입 연령 {min_age}세 미만"
    if max_age is not None and age is not None and age > max_age:
        return False, f"최대 가입 연령 {max_age}세 초과"

    # 월 납입 기반 체크 (optional)
    monthly = product.get('monthly_payment')
    capacity = customer.get('monthly_capacity')
    if monthly and capacity and monthly > capacity:
        return False, "월 납입 가능액 초과"

    return True, ""
