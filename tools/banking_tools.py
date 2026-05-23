"""
Banking Tools - 에이전트들이 사용하는 Tool 모음
- LangChain @tool 데코레이터로 정의
- 각 에이전트가 필요한 Tool을 선택적으로 바인딩하여 사용
"""
from datetime import datetime, date
from langchain_core.tools import tool

from db.vectorstore import search_products
from db.customer_db import get_customer, get_customer_products as _get_customer_products


# ───────────────────────────────────────────
# 1. RAG 검색 Tool
# ───────────────────────────────────────────
@tool
def search_product_info(query: str) -> str:
    """상품 정보를 검색합니다. 예금/적금 상품의 금리, 조건, 특징 등을 Vector DB에서 검색합니다.
    
    Args:
        query: 검색할 질문 (예: "정기예금 금리", "청년 적금 조건")
    """
    results = search_products(query, k=5)
    if not results:
        return "검색된 상품 정보가 없습니다. Vector DB가 구축되지 않았거나 관련 정보가 없습니다."

    output = []
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source_file", "알 수 없음")
        page = doc.metadata.get("page", "?")
        output.append(f"[{i}] 출처: {source} / p.{page}\n{doc.page_content}")

    return "\n\n---\n\n".join(output)


# ───────────────────────────────────────────
# 2. 고객 기본 정보 조회 Tool
# ───────────────────────────────────────────
@tool
def get_customer_info(customer_id: str) -> str:
    """고객의 기본 정보(이름, 나이, 직업, 연락처)를 조회합니다.
    
    Args:
        customer_id: 고객 ID (예: "C001")
    """
    customer = get_customer(customer_id)
    if not customer:
        return f"고객 ID '{customer_id}'에 해당하는 고객을 찾을 수 없습니다."

    return (
        f"이름: {customer['name']}\n"
        f"나이: {customer['age']}세\n"
        f"직업: {customer['job']}\n"
        f"연락처: {customer.get('phone', '미등록')}\n"
        f"가입 상품 수: {len(customer.get('products', []))}건"
    )


# ───────────────────────────────────────────
# 3. 고객 가입 상품 조회 Tool
# ───────────────────────────────────────────
@tool
def get_customer_products_tool(customer_id: str) -> str:
    """고객이 가입한 예금/적금 상품 목록을 상세하게 조회합니다.
    
    Args:
        customer_id: 고객 ID (예: "C001")
    """
    products = _get_customer_products(customer_id)
    if not products:
        return f"고객 ID '{customer_id}'의 가입 상품이 없거나 고객을 찾을 수 없습니다."

    today = date.today()
    lines = []
    for i, prod in enumerate(products, 1):
        lines.append(f"━━━ {i}. {prod['name']} ({prod['type']}) ━━━")

        if prod["type"] == "정기예금":
            lines.append(f"  예치금액: {prod['principal']:,.0f}원")
        else:
            lines.append(f"  월 납입액: {prod['monthly_payment']:,.0f}원")
            if prod.get("paid_count") is not None and prod.get("total_count") is not None:
                lines.append(f"  납입 현황: {prod['paid_count']}/{prod['total_count']}회")
                remaining = prod["total_count"] - prod["paid_count"]
                lines.append(f"  남은 납입: {remaining}회")

        total_rate = prod["rate"] + prod.get("bonus_rate", 0)
        lines.append(f"  기본금리: {prod['rate']}%")
        if prod.get("bonus_rate", 0) > 0:
            lines.append(f"  우대금리: +{prod['bonus_rate']}%")
        lines.append(f"  적용금리: {total_rate}%")
        lines.append(f"  가입일: {prod['start_date']}")
        lines.append(f"  만기일: {prod['end_date']}")
        lines.append("")

    return "\n".join(lines)


# ───────────────────────────────────────────
# 4. 이자 계산 Tool
# ───────────────────────────────────────────
@tool
def calculate_interest(
    principal: float,
    annual_rate: float,
    months: int,
    interest_type: str = "단리",
    monthly_payment: float = 0,
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
        # 적금 이자 계산 (단리)
        total_payment = monthly_payment * months
        total_interest = 0
        for i in range(months):
            remaining_months = months - i
            total_interest += monthly_payment * rate * remaining_months / 12
        tax = total_interest * 0.154  # 이자소득세 15.4%
        after_tax_interest = total_interest - tax

        return (
            f"━━━ 적금 이자 계산 결과 ━━━\n"
            f"월 납입액: {monthly_payment:,.0f}원\n"
            f"연이율: {annual_rate}%\n"
            f"가입기간: {months}개월\n"
            f"총 납입액: {total_payment:,.0f}원\n"
            f"세전 이자: {total_interest:,.0f}원\n"
            f"이자소득세(15.4%): {tax:,.0f}원\n"
            f"세후 이자: {after_tax_interest:,.0f}원\n"
            f"만기 수령액: {total_payment + after_tax_interest:,.0f}원"
        )
    else:
        # 정기예금 이자 계산
        years = months / 12
        if interest_type == "복리":
            total = principal * (1 + rate / 12) ** months
            total_interest = total - principal
        else:  # 단리
            total_interest = principal * rate * years

        tax = total_interest * 0.154
        after_tax_interest = total_interest - tax

        return (
            f"━━━ 정기예금 이자 계산 결과 ({interest_type}) ━━━\n"
            f"예치금액: {principal:,.0f}원\n"
            f"연이율: {annual_rate}%\n"
            f"가입기간: {months}개월\n"
            f"세전 이자: {total_interest:,.0f}원\n"
            f"이자소득세(15.4%): {tax:,.0f}원\n"
            f"세후 이자: {after_tax_interest:,.0f}원\n"
            f"만기 수령액: {principal + after_tax_interest:,.0f}원"
        )


# ───────────────────────────────────────────
# 5. 금리 비교 Tool
# ───────────────────────────────────────────
@tool
def compare_rates(customer_id: str) -> str:
    """고객이 가입한 상품들의 금리를 비교표로 보여줍니다.
    
    Args:
        customer_id: 고객 ID (예: "C001")
    """
    products = _get_customer_products(customer_id)
    if not products:
        return f"고객 ID '{customer_id}'의 가입 상품이 없습니다."

    lines = []
    lines.append("| 상품명 | 유형 | 기본금리 | 우대금리 | 적용금리 |")
    lines.append("|--------|------|---------|---------|---------|")

    for prod in products:
        total_rate = prod["rate"] + prod.get("bonus_rate", 0)
        bonus = f"+{prod['bonus_rate']}%" if prod.get("bonus_rate", 0) > 0 else "-"
        lines.append(
            f"| {prod['name']} | {prod['type']} | {prod['rate']}% | {bonus} | {total_rate}% |"
        )

    return "\n".join(lines)


# ───────────────────────────────────────────
# 6. 만기 확인 Tool
# ───────────────────────────────────────────
@tool
def check_maturity(customer_id: str) -> str:
    """고객 상품의 만기 현황을 확인합니다. 만기까지 남은 기간을 계산해줍니다.
    
    Args:
        customer_id: 고객 ID (예: "C001")
    """
    products = _get_customer_products(customer_id)
    if not products:
        return f"고객 ID '{customer_id}'의 가입 상품이 없습니다."

    today = date.today()
    lines = []

    for prod in products:
        try:
            end_date = datetime.strptime(prod["end_date"], "%Y-%m-%d").date()
            days_left = (end_date - today).days

            if days_left > 0:
                status = f"만기까지 {days_left}일 남음"
            elif days_left == 0:
                status = "⚠️ 오늘 만기!"
            else:
                status = f"✅ 이미 만기됨 ({abs(days_left)}일 전)"
        except ValueError:
            status = "날짜 확인 불가"

        lines.append(f"• {prod['name']}: {prod['end_date']} ({status})")

    return "\n".join(lines)


# ───────────────────────────────────────────
# 에이전트별 Tool 그룹
# ───────────────────────────────────────────
PRODUCT_TOOLS = [search_product_info, get_customer_info, get_customer_products_tool, check_maturity]
ANALYSIS_TOOLS = [search_product_info, get_customer_info, get_customer_products_tool, calculate_interest, compare_rates]
RECOMMEND_TOOLS = [search_product_info, get_customer_info, get_customer_products_tool]
