"""
Recommend 에이전트 전용 도구
추천 후보 필터링 및 순위 비교 연산
※ 고객 DB 및 상품 DB에 직접 접근하지 않습니다. (get_product_info, nl2sql_query 제거)
※ 오직 Customer Agent가 제공한 고객 프로필과 Product Agent가 RAG로 수집해 준 상품 정보를 인자로 받아 추천 연산만 수행합니다.
"""
from langchain_core.tools import tool


@tool
def rank_products(
    products_info: str,
    purpose: str = "",
    period_months: int = 0,
    monthly_amount: int = 0,
) -> str:
    """추천 후보 상품 목록을 사용자의 목적, 가입 기간, 가용 금액 조건과 비교하여 필터링 및 점수화(순위 매김)를 처리합니다.

    Args:
        products_info: Product Agent가 RAG를 통해 수집해 준 은행 예적금 상품 목록 및 가입조건 전체 정보
        purpose: 사용자의 저축 목적 (예: "결혼자금", "비상금", "노후자금" 등)
        period_months: 희망 가입 기간 (개월)
        monthly_amount: 매월 납입 가능 금액 (적금일 때 가용 금액, 원)
    """
    lines = [
        "━━━ 추천 상품 후보 분석 연산 ━━━\n",
        "【전달받은 전체 상품 목록 정보】", products_info, "",
    ]
    if purpose:
        lines.append(f"【희망 저축 목적】 {purpose}")
    if period_months > 0:
        lines.append(f"【희망 가입 기간】 {period_months}개월")
    if monthly_amount > 0:
        lines.append(f"【가용 납입 금액】 {monthly_amount:,}원")
        
    lines.extend([
        "",
        "━━━ 순위 연산 규칙 ━━━",
        "• 최고 금리가 높은 순으로 1차 정렬",
        "• 희망 가입 기간이 최소~최대 가입 기간 범위 내에 안전하게 포함되는지 확인",
        "• 월 가용 납입 금액이 상품의 최소 가입 금액 조건 이상을 충족하는지 대조",
    ])
    return "\n".join(lines)


# 이 에이전트에 바인딩될 도구 목록
RECOMMEND_TOOLS = [rank_products]
