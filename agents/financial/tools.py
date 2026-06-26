"""
Financial 에이전트 전용 도구
이자 계산, 상품 비교 연산, 갈아타기 시뮬레이션 분석

※ 고객 DB 및 상품 DB에 직접 접근하지 않습니다.
※ Customer Agent가 제공한 고객/계좌 정보와 Product Agent가 제공한 상품 정보를 인자로 받아
   계산/비교/시뮬레이션만 수행합니다.
"""

from langchain_core.tools import tool


DEFAULT_TAX_RATE = 0.154


def _format_money(value: float) -> str:
    return f"{value:,.0f}원"


def _format_rate(value: float) -> str:
    return f"{value:.2f}%"


def _calculate_deposit_interest(
    principal: float,
    annual_rate: float,
    months: int,
    interest_type: str,
    tax_rate: float,
) -> dict:
    rate = annual_rate / 100
    years = months / 12

    if interest_type == "복리":
        before_tax_interest = principal * ((1 + rate / 12) ** months - 1)
    else:
        before_tax_interest = principal * rate * years

    tax = before_tax_interest * tax_rate
    after_tax_interest = before_tax_interest - tax
    maturity_amount = principal + after_tax_interest

    return {
        "principal": principal,
        "before_tax_interest": before_tax_interest,
        "tax": tax,
        "after_tax_interest": after_tax_interest,
        "maturity_amount": maturity_amount,
    }


def _calculate_savings_interest(
    monthly_payment: float,
    annual_rate: float,
    months: int,
    tax_rate: float,
) -> dict:
    rate = annual_rate / 100
    total_payment = monthly_payment * months

    before_tax_interest = sum(
        monthly_payment * rate * (months - payment_index) / 12
        for payment_index in range(months)
    )

    tax = before_tax_interest * tax_rate
    after_tax_interest = before_tax_interest - tax
    maturity_amount = total_payment + after_tax_interest

    return {
        "total_payment": total_payment,
        "before_tax_interest": before_tax_interest,
        "tax": tax,
        "after_tax_interest": after_tax_interest,
        "maturity_amount": maturity_amount,
    }


@tool
def calculate_interest(
    principal: float,
    annual_rate: float,
    months: int,
    interest_type: str = "단리",
    monthly_payment: float = 0,
    tax_rate: float = DEFAULT_TAX_RATE,
) -> str:
    """예금 또는 적금의 예상 이자와 만기 수령액을 계산합니다.

    Args:
        principal: 예치 원금. 정기예금일 때 사용합니다.
        annual_rate: 연이율. 예: 3.5
        months: 가입 기간. 개월 단위입니다.
        interest_type: "단리" 또는 "복리". 기본값은 "단리"입니다.
        monthly_payment: 월 납입액. 적금일 때 사용합니다.
        tax_rate: 이자소득 원천징수세율. 기본값은 0.154입니다.
    """
    if annual_rate < 0:
        return "연이율은 0 이상이어야 합니다."

    if months <= 0:
        return "가입 기간은 1개월 이상이어야 합니다."

    if tax_rate < 0 or tax_rate >= 1:
        return "세율은 0 이상 1 미만의 소수로 입력해야 합니다. 예: 0.154"

    if monthly_payment > 0:
        result = _calculate_savings_interest(
            monthly_payment=monthly_payment,
            annual_rate=annual_rate,
            months=months,
            tax_rate=tax_rate,
        )

        return (
            "적금 이자 계산 결과\n\n"
            "계산 기준\n"
            f"- 월 납입액: {_format_money(monthly_payment)}\n"
            f"- 연 이율: {_format_rate(annual_rate)}\n"
            f"- 가입 기간: {months}개월\n"
            f"- 적용 세율: {_format_rate(tax_rate * 100)}\n\n"
            "계산 과정\n"
            f"1. 총 납입액 = {_format_money(monthly_payment)} x {months}개월 = {_format_money(result['total_payment'])}\n"
            f"2. 세전 이자 = 월 납입액별 예치 기간을 반영하여 {_format_money(result['before_tax_interest'])}\n"
            f"3. 세금 = {_format_money(result['before_tax_interest'])} x {_format_rate(tax_rate * 100)} = {_format_money(result['tax'])}\n"
            f"4. 세후 이자 = {_format_money(result['before_tax_interest'])} - {_format_money(result['tax'])} = {_format_money(result['after_tax_interest'])}\n"
            f"5. 만기 예상 수령액 = {_format_money(result['total_payment'])} + {_format_money(result['after_tax_interest'])} = {_format_money(result['maturity_amount'])}\n\n"
            "결과\n"
            f"- 총 납입액: {_format_money(result['total_payment'])}\n"
            f"- 세전 이자: {_format_money(result['before_tax_interest'])}\n"
            f"- 세금: {_format_money(result['tax'])}\n"
            f"- 세후 이자: {_format_money(result['after_tax_interest'])}\n"
            f"- 만기 예상 수령액: {_format_money(result['maturity_amount'])}"
        )

    if principal <= 0:
        return "정기예금 계산 시 예치 원금은 0원보다 커야 합니다."

    if interest_type not in ["단리", "복리"]:
        return 'interest_type은 "단리" 또는 "복리"만 사용할 수 있습니다.'

    result = _calculate_deposit_interest(
        principal=principal,
        annual_rate=annual_rate,
        months=months,
        interest_type=interest_type,
        tax_rate=tax_rate,
    )

    if interest_type == "복리":
        interest_formula = (
            f"세전 이자 = {_format_money(principal)}에 월 복리 이율을 {months}개월 적용한 금액 "
            f"- 원금 = {_format_money(result['before_tax_interest'])}"
        )
    else:
        interest_formula = (
            f"세전 이자 = {_format_money(principal)} x {_format_rate(annual_rate)} "
            f"x {months}개월 / 12 = {_format_money(result['before_tax_interest'])}"
        )

    return (
        f"정기예금 이자 계산 결과 ({interest_type})\n\n"
        "계산 기준\n"
        f"- 예치 원금: {_format_money(principal)}\n"
        f"- 연 이율: {_format_rate(annual_rate)}\n"
        f"- 예치 기간: {months}개월\n"
        f"- 이자 방식: {interest_type}\n"
        f"- 적용 세율: {_format_rate(tax_rate * 100)}\n\n"
        "계산 과정\n"
        f"1. {interest_formula}\n"
        f"2. 세금 = {_format_money(result['before_tax_interest'])} x {_format_rate(tax_rate * 100)} = {_format_money(result['tax'])}\n"
        f"3. 세후 이자 = {_format_money(result['before_tax_interest'])} - {_format_money(result['tax'])} = {_format_money(result['after_tax_interest'])}\n"
        f"4. 만기 예상 수령액 = {_format_money(principal)} + {_format_money(result['after_tax_interest'])} = {_format_money(result['maturity_amount'])}\n\n"
        "결과\n"
        f"- 세전 이자: {_format_money(result['before_tax_interest'])}\n"
        f"- 세금: {_format_money(result['tax'])}\n"
        f"- 세후 이자: {_format_money(result['after_tax_interest'])}\n"
        f"- 만기 예상 수령액: {_format_money(result['maturity_amount'])}"
    )


@tool
def compare_products(products_info: str) -> str:
    """Product Agent가 제공한 여러 상품 정보를 비교 분석합니다.

    Args:
        products_info: Product Agent가 수집한 상품 정보 문자열.
    """
    if not products_info or not products_info.strip():
        return (
            "상품 비교를 수행할 수 없습니다.\n"
            "비교 대상 상품 정보가 필요합니다."
        )

    return (
        "금융 상품 비교 분석\n\n"
        "전달받은 상품 정보\n"
        f"{products_info}\n\n"
        "비교 관점\n"
        "- 기본 금리와 최고 우대 금리 차이를 비교합니다.\n"
        "- 가입 기간과 만기 조건을 비교합니다.\n"
        "- 월 납입 한도 또는 예치 한도를 비교합니다.\n"
        "- 우대금리 조건 충족 가능성을 비교합니다.\n"
        "- 중도해지 조건과 유의사항을 비교합니다.\n\n"
        "위 비교는 Product Agent가 제공한 상품 정보 범위 안에서만 수행됩니다."
    )


@tool
def compare_switch_benefit(
    current_balance: float,
    current_rate: float,
    remaining_months: int,
    new_rate: float,
    new_months: int,
    early_termination_rate: float = 0,
    monthly_payment: float = 0,
    tax_rate: float = DEFAULT_TAX_RATE,
    new_product_info: str = "",
    customer_accounts: str = "",
) -> str:
    """기존 상품 유지와 신규 상품 가입의 갈아타기 유불리를 계산합니다.

    Args:
        current_balance: 기존 계좌 현재 잔액.
        current_rate: 기존 상품 연이율.
        remaining_months: 기존 상품 만기까지 남은 개월 수.
        new_rate: 신규 상품 연이율.
        new_months: 신규 상품 가입 기간.
        early_termination_rate: 기존 상품 중도해지 연이율.
        monthly_payment: 적금형 비교 시 월 납입액. 예금형 비교면 0입니다.
        tax_rate: 이자소득 원천징수세율. 기본값은 0.154입니다.
        new_product_info: Product Agent가 제공한 신규 상품 정보.
        customer_accounts: Customer Agent가 제공한 기존 계좌 정보.
    """
    if current_balance <= 0:
        return "갈아타기 비교를 위해 기존 계좌 잔액이 필요합니다."

    if current_rate < 0 or new_rate < 0 or early_termination_rate < 0:
        return "금리는 0 이상이어야 합니다."

    if remaining_months <= 0 or new_months <= 0:
        return "비교 기간은 1개월 이상이어야 합니다."

    if tax_rate < 0 or tax_rate >= 1:
        return "세율은 0 이상 1 미만의 소수로 입력해야 합니다. 예: 0.154"

    keep_result = _calculate_deposit_interest(
        principal=current_balance,
        annual_rate=current_rate,
        months=remaining_months,
        interest_type="단리",
        tax_rate=tax_rate,
    )

    if early_termination_rate > 0:
        early_result = _calculate_deposit_interest(
            principal=current_balance,
            annual_rate=early_termination_rate,
            months=remaining_months,
            interest_type="단리",
            tax_rate=tax_rate,
        )
        early_amount = early_result["maturity_amount"]
        early_loss = keep_result["maturity_amount"] - early_amount
    else:
        early_amount = current_balance
        early_loss = keep_result["maturity_amount"] - current_balance

    if monthly_payment > 0:
        new_result = _calculate_savings_interest(
            monthly_payment=monthly_payment,
            annual_rate=new_rate,
            months=new_months,
            tax_rate=tax_rate,
        )
        new_maturity_amount = new_result["maturity_amount"]
        new_principal = new_result["total_payment"]
    else:
        new_result = _calculate_deposit_interest(
            principal=early_amount,
            annual_rate=new_rate,
            months=new_months,
            interest_type="단리",
            tax_rate=tax_rate,
        )
        new_maturity_amount = new_result["maturity_amount"]
        new_principal = early_amount

    difference = new_maturity_amount - keep_result["maturity_amount"]

    if difference > 0:
        summary = f"금액 기준으로 신규 상품 가입 시 {_format_money(difference)} 더 높게 계산됩니다."
    elif difference < 0:
        summary = f"금액 기준으로 기존 상품 유지 시 {_format_money(abs(difference))} 더 높게 계산됩니다."
    else:
        summary = "금액 기준으로 기존 유지와 신규 가입의 예상 수령액이 동일하게 계산됩니다."

    lines = [
        "갈아타기 유불리 시뮬레이션",
        "",
        "기존 상품 유지 시",
        f"- 현재 잔액: {_format_money(current_balance)}",
        f"- 기존 연이율: {_format_rate(current_rate)}",
        f"- 남은 기간: {remaining_months}개월",
        f"- 적용 세율: {_format_rate(tax_rate * 100)}",
        f"- 만기 예상 수령액: {_format_money(keep_result['maturity_amount'])}",
        "",
        "중도해지 후 신규 상품 가입 시",
        f"- 중도해지 적용 연이율: {_format_rate(early_termination_rate)}",
        f"- 중도해지 예상 수령액: {_format_money(early_amount)}",
        f"- 중도해지로 인한 예상 손실: {_format_money(early_loss)}",
        f"- 신규 상품 연이율: {_format_rate(new_rate)}",
        f"- 신규 상품 가입 기간: {new_months}개월",
        f"- 신규 계산 원금 또는 총 납입액: {_format_money(new_principal)}",
        f"- 신규 상품 만기 예상 수령액: {_format_money(new_maturity_amount)}",
        "",
        "비교 결과",
        f"- 기존 유지 예상 수령액: {_format_money(keep_result['maturity_amount'])}",
        f"- 갈아타기 예상 수령액: {_format_money(new_maturity_amount)}",
        f"- 차이: {_format_money(difference)}",
        f"- 결론: {summary}",
    ]

    if customer_accounts:
        lines.extend(["", "참고한 기존 계좌 정보", customer_accounts])

    if new_product_info:
        lines.extend(["", "참고한 신규 상품 정보", new_product_info])

    lines.extend([
        "",
        "실제 적용 금리, 세금, 우대 조건, 중도해지 기준에 따라 결과가 달라질 수 있습니다.",
    ])

    return "\n".join(lines)


FINANCIAL_TOOLS = [
    calculate_interest,
    compare_products,
    compare_switch_benefit,
]
