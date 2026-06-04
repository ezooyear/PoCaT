"""
Eligibility 에이전트 전용 도구
가입 가능 여부 판단, 우대금리 충족 확인, 가입 가능 상품 필터링
※ 고객 데이터베이스(DB) 및 금융상품 데이터베이스(DB)에 직접 접근하지 않습니다.
※ 오직 Customer Agent가 제공한 고객 데이터와 Product Agent가 RAG를 통해 수집해 준 상품 데이터를 인자로 받아 비교/필터링 연산만 수행합니다.
"""
from langchain_core.tools import tool


@tool
def evaluate_eligibility(customer_profile: str, product_info: str) -> str:
    """고객 프로필 정보와 상품 가입 조건 정보를 서로 대조하고 비교하여 최종 가입 가능 여부를 판단합니다.
    나이, 가입 금액(월 가용 저축액) 등의 가입 제한 조건을 비교 연산합니다.

    Args:
        customer_profile: Customer Agent가 조회해준 고객 프로필 정보 (나이, 직업, 소득, 가용저축액 등)
        product_info: Product Agent가 RAG를 통해 조회해준 상품 가입 조건 정보 (최소나이, 최대나이, 최소가입금액 등)
    """
    lines = [
        f"━━━ 가입 가능 여부 비교 연산 ━━━\n",
        "【전달받은 고객 정보】", customer_profile, "",
        "【전달받은 상품 조건 정보】", product_info, "",
        "━━━ 판단 연산 기준 ━━━",
        "• 고객 나이가 상품의 최소나이 ~ 최대나이 범위 내인지 비교",
        "• 고객의 월 가용 저축액이 상품의 최소 가입금액 이상인지 비교",
        "• 상품의 판매중 여부가 정상(TRUE 또는 활성)인지 대조",
    ]
    return "\n".join(lines)


@tool
def evaluate_bonus_rate(customer_profile: str, customer_accounts: str = "") -> str:
    """고객의 거래 조건 및 계좌 현황을 기반으로 우대금리 충족 여부를 판단합니다.
    급여이체, 자동이체, 카드 실적, 주거래 여부 등을 분석합니다.

    Args:
        customer_profile: Customer Agent가 조회해준 고객 프로필 정보 (급여이체, 자동이체, 카드 사용, 주거래 여부 등)
        customer_accounts: Customer Agent가 조회해준 기존 계좌 및 거래 현황 정보.
    """
    lines = [
        "━━━ 우대금리 조건 충족 여부 판단 ━━━\n",
        "【전달받은 고객 조건 현황】", customer_profile, "",
    ]
    if customer_accounts:
        lines.extend(["【전달받은 가입 상품 현황】", customer_accounts, ""])

    lines.extend([
        "━━━ 적용 가능한 우대금리 연산 기준 ━━━",
        "• 급여이체 실적 존재 여부 매핑 (보통 +0.1~0.3%p)",
        "• 자동이체 등록 여부 매핑 (보통 +0.1~0.2%p)",
        "• 신용/체크카드 사용 실적 여부 매핑 (보통 +0.1~0.2%p)",
        "• 주거래 은행 여부 매핑 (보통 +0.1~0.3%p)",
        "• 마케팅 동의 여부 매핑 (보통 +0.1%p)",
    ])
    return "\n".join(lines)


@tool
def filter_eligible_products(customer_profile: str, products_info: str) -> str:
    """고객 정보(나이, 가용소득 등)와 전체 예적금 상품의 가입 조건 목록을 서로 비교 연산하여,
    가입 가능한 상품 목록을 최종 필터링합니다.

    Args:
        customer_profile: Customer Agent가 조회해준 고객 프로필 정보 (나이, 직업, 소득, 가용저축액 등)
        products_info: Product Agent가 RAG를 통해 수집하여 전달해준 전체 금융상품 가입 요건 정보
    """
    lines = [
        "━━━ 가입 가능 상품 필터링 비교 연산 ━━━\n",
        "【전달받은 고객 정보】", customer_profile, "",
        "【전달받은 전체 상품 조건 정보】", products_info, "",
        "━━━ 필터링 규칙 ━━━",
        "• 고객 나이가 각 상품의 가입 가능 연령 범위 내에 있는 상품만 추출",
        "• 고객의 월 가용 저축액이 상품의 최소 가입 금액 이상인 상품만 추출",
    ]
    return "\n".join(lines)


# 이 에이전트에 바인딩될 도구 목록
ELIGIBILITY_TOOLS = [evaluate_eligibility, evaluate_bonus_rate, filter_eligible_products]
