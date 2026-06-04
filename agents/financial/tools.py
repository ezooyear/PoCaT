"""
Financial 에이전트 전용 도구
이자 계산, 상품 비교 연산, 갈아타기 시뮬레이션 분석
※ 고객 DB 및 상품 DB에 직접 접근하지 않습니다. (get_product_info, nl2sql_query 제거)
※ 오직 Customer Agent가 제공한 고객/계좌 정보와 Product Agent가 RAG로 수집해 준 상품 정보를 인자로 받아 연산/시뮬레이션만 수행합니다.
"""
from langchain_core.tools import tool


@tool
def calculate_interest(
    principal: float, annual_rate: float, months: int,
    interest_type: str = "단리", monthly_payment: float = 0,
) -> str:
    """예적금 이자를 계산합니다. 정기예금(단리/복리)과 적금(단리) 모두 계산 가능합니다.

    Args:
        principal: 예치 원금 (정기예금일 때 사용, 적금이면 0)
        annual_rate: 연이율 (%, 예: 3.5)
        months: 가입 기간 (개월)
        interest_type: "단리" 또는 "복리" (기본값: 단리)
        monthly_payment: 월 납입액 (적금일 때 사용, 예금이면 0)
    """
    rate = annual_rate / 100

    if monthly_payment > 0:
        total_payment = monthly_payment * months
        total_interest = sum(
            monthly_payment * rate * (months - i) / 12 for i in range(months)
        )
        tax = total_interest * 0.154
        after_tax = total_interest - tax
        return (
            f"━━━ 적금 이자 계산 결과 ━━━\n"
            f"월 납입액: {monthly_payment:,.0f}원\n연이율: {annual_rate}%\n"
            f"가입기간: {months}개월\n총 납입액: {total_payment:,.0f}원\n"
            f"세전 이자: {total_interest:,.0f}원\n이자소득세(15.4%): {tax:,.0f}원\n"
            f"세후 이자: {after_tax:,.0f}원\n만기 수령액: {total_payment + after_tax:,.0f}원"
        )
    else:
        years = months / 12
        if interest_type == "복리":
            total_interest = principal * (1 + rate / 12) ** months - principal
        else:
            total_interest = principal * rate * years
        tax = total_interest * 0.154
        after_tax = total_interest - tax
        return (
            f"━━━ 정기예금 이자 계산 결과 ({interest_type}) ━━━\n"
            f"예치금액: {principal:,.0f}원\n연이율: {annual_rate}%\n"
            f"가입기간: {months}개월\n세전 이자: {total_interest:,.0f}원\n"
            f"이자소득세(15.4%): {tax:,.0f}원\n세후 이자: {after_tax:,.0f}원\n"
            f"만기 수령액: {principal + after_tax:,.0f}원"
        )


@tool
def compare_products(products_info: str) -> str:
    """여러 상품들의 금리, 가입 조건, 우대 조건 등을 서로 대조하여 일대일 비교 분석합니다.

    Args:
        products_info: Product Agent가 RAG를 통해 수집해 준 대상 상품들의 상세 요건 정보 (금리, 한도 등)
    """
    lines = [
        "━━━ 금융 상품 정보 비교 분석 연산 ━━━\n",
        "【전달받은 비교 대상 상품 정보】", products_info, "",
        "━━━ 비교 분석 요점 ━━━",
        "• 기본 금리와 최고 우대 금리의 차이 대조",
        "• 가입 한도(금액) 및 의무 가입 기간 비교",
        "• 각 상품의 우대 혜택 적용을 위한 장단점 나열",
    ]
    return "\n".join(lines)


@tool
def compare_switch_benefit(new_product_info: str, customer_accounts: str = "") -> str:
    """신규 상품의 조건과 고객의 기존 가입 계좌 현황을 비교하여 중도해지 vs 갈아타기의 유불리를 시뮬레이션합니다.

    Args:
        new_product_info: Product Agent가 RAG를 통해 수집해 준 신규 가입 후보 상품의 상세 정보 (금리, 가입 조건 등)
        customer_accounts: Customer Agent가 조회해 준 고객의 기존 예적금 계좌 목록 및 상태 정보
    """
    lines = [
        "━━━ 갈아타기 유불리 시뮬레이션 분석 ━━━\n",
        "【전달받은 신규 후보 상품 정보】", new_product_info, "",
    ]
    if customer_accounts:
        lines.extend(["【전달받은 기존 가입 계좌 현황】", customer_accounts, ""])

    lines.extend([
        "━━━ 갈아타기 유불리 판단 기준 ━━━",
        "• 남은 만기일이 3개월 이내라면 기존 계좌를 유지하는 것이 일반적으로 유리",
        "• 신규 상품의 최고 우대금리가 기존 금리보다 최소 1.0%p 이상 높아야 갈아타기 메리트 발생",
        "• 기존 상품 중도해지 시 중도해지이율(기본금리의 40~60% 감면 등)로 인한 이자 손실액 비교 판단",
    ])
    return "\n".join(lines)


# 이 에이전트에 바인딩될 도구 목록
FINANCIAL_TOOLS = [calculate_interest, compare_products, compare_switch_benefit]
