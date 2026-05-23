"""
Banking Tools - 에이전트들이 사용하는 Tool 모음
- NL2SQL: 자연어 → SQL 변환 → DB 조회
- RAG: Vector DB 검색
- 이자 계산
"""
from datetime import datetime, date
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage

from db.vectorstore import search_products
from db.postgres_db import DB_SCHEMA, execute_query
from config.settings import get_llm


# ───────────────────────────────────────────
# 1. NL2SQL - 데이터베이스 조회 Tool
# ───────────────────────────────────────────
NL2SQL_SYSTEM_PROMPT = f"""당신은 PostgreSQL 전문가입니다.
사용자의 자연어 질문을 SQL SELECT 쿼리로 변환해주세요.

{DB_SCHEMA}

## 규칙
- 반드시 SELECT 쿼리만 작성하세요. INSERT, UPDATE, DELETE 등은 절대 사용하지 마세요.
- SQL 쿼리만 출력하세요. 설명이나 다른 텍스트는 포함하지 마세요.
- 코드 블록(```)도 사용하지 마세요. 순수 SQL만 출력하세요.
- 테이블 JOIN이 필요한 경우 적절한 JOIN을 사용하세요.
- 결과가 너무 많을 수 있으므로 LIMIT을 적절히 사용하세요.
- 금액은 원 단위, 소득은 만원 단위임을 기억하세요.
- 고객 이름은 "고객_001" 형식입니다.

## 도메인 용어 매핑 규칙 (매우 중요)
- 사용자가 "군인", "직업군인", "장기복무", "장교", "부사관" 등을 언급하면, 상품명(product_name) 검색 시 `LIKE '%국군%' OR LIKE '%장병%' OR LIKE '%장기복무%'` 등 관련 키워드를 폭넓게 사용하여 검색하세요.
"""


@tool
def query_database(question: str) -> str:
    """자연어 질문을 SQL로 변환하여 PostgreSQL 데이터베이스를 조회합니다.
    고객 정보, 계좌, 상품, 납입 이력 등을 조회할 수 있습니다.
    
    Args:
        question: 자연어 질문 (예: "고객_001의 가입 상품을 보여줘", "적금 상품 목록 조회")
    """
    llm = get_llm()

    # 1. LLM에게 SQL 생성 요청
    messages = [
        SystemMessage(content=NL2SQL_SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]

    response = llm.invoke(messages)
    sql = response.content.strip()

    # 코드 블록 제거 (혹시 포함된 경우)
    if sql.startswith("```"):
        lines = sql.split("\n")
        sql = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        sql = sql.strip()

    # 2. SQL 실행 및 결과 반환
    result = execute_query(sql)
    return f"실행된 SQL:\n{sql}\n\n{result}"


# ───────────────────────────────────────────
# 2. DB 스키마 조회 Tool
# ───────────────────────────────────────────
@tool
def get_db_schema() -> str:
    """데이터베이스의 테이블 구조(스키마)를 조회합니다.
    어떤 테이블과 컬럼이 있는지 확인할 때 사용합니다.
    """
    return DB_SCHEMA


# ───────────────────────────────────────────
# 3. RAG 검색 Tool (기존 유지)
# ───────────────────────────────────────────
@tool
def search_product_info(query: str) -> str:
    """상품의 약관, 상세 조건, 유의사항 등을 PDF 문서에서 검색합니다.
    DB에 없는 상세 정보(약관, 우대금리 조건 등)를 찾을 때 사용합니다.
    
    Args:
        query: 검색할 질문 (예: "정기예금 중도해지 규정", "청년 적금 우대금리 조건")
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
# 4. 이자 계산 Tool (기존 유지)
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
# 5. 중도해지 손실 계산 Tool
# ───────────────────────────────────────────
@tool
def check_early_termination(customer_name: str) -> str:
    """고객의 ACTIVE 상태 계좌를 조회하고, 각 계좌를 중도해지할 경우의 손실을 계산합니다.
    어떤 상품을 해지하는 것이 손해가 적은지 비교할 때 사용합니다.
    
    Args:
        customer_name: 고객 이름 (예: "고객_001")
    """
    # 1. 고객의 ACTIVE 계좌 조회
    sql = f"""
    SELECT 
        a.account_id, p.product_name, p.product_type,
        a.deposit_amount, a.monthly_amount, a.current_balance,
        a.applied_rate, p.base_rate,
        a.join_date, a.maturity_date, a.contract_months,
        (CURRENT_DATE - a.join_date) as days_elapsed,
        (a.maturity_date - CURRENT_DATE) as days_remaining
    FROM customers c
    JOIN customer_accounts a ON c.customer_id = a.customer_id
    JOIN products p ON a.product_id = p.product_id
    WHERE c.customer_name = '{customer_name}'
    AND a.account_status = 'ACTIVE'
    ORDER BY a.current_balance DESC;
    """
    db_result = execute_query(sql)

    if "조회 결과가 없습니다" in db_result:
        return f"{customer_name}님의 ACTIVE 상태 계좌가 없습니다."

    # 2. 중도해지 손실 분석
    lines = []
    lines.append(f"━━━ {customer_name}님 중도해지 분석 ━━━\n")
    lines.append("※ 중도해지 시 일반적으로 기본금리의 40~60% 수준의 금리가 적용됩니다.\n")
    lines.append(db_result)
    lines.append("")
    lines.append("━━━ 중도해지 시 참고사항 ━━━")
    lines.append("• 중도해지 금리는 약정 금리보다 크게 낮습니다 (보통 기본금리의 40~60%)")
    lines.append("• 잔여 기간이 짧을수록 유지하는 것이 유리합니다")
    lines.append("• current_balance(현재 잔액)가 낮은 계좌를 해지하는 것이 손실이 적습니다")
    lines.append("• 정확한 중도해지 금리는 약관을 확인해야 합니다")

    return "\n".join(lines)


# ───────────────────────────────────────────
# 6. 갈아타기 분석 Tool
# ───────────────────────────────────────────
@tool
def analyze_switch_product(customer_name: str, new_product_name: str) -> str:
    """현재 가입한 상품을 해지하고 새 상품으로 갈아타는 것이 유리한지 분석합니다.
    기존 상품 유지 vs 새 상품 갈아타기를 비교합니다.
    
    Args:
        customer_name: 고객 이름 (예: "고객_001")
        new_product_name: 갈아타려는 새 상품명 (예: "KB Star 정기예금")
    """
    # 1. 고객의 현재 ACTIVE 계좌 조회
    current_sql = f"""
    SELECT 
        a.account_id, p.product_name, p.product_type,
        a.deposit_amount, a.monthly_amount, a.current_balance,
        a.applied_rate, a.contract_months,
        a.join_date, a.maturity_date,
        EXTRACT(MONTH FROM AGE(a.maturity_date, CURRENT_DATE)) as months_remaining
    FROM customers c
    JOIN customer_accounts a ON c.customer_id = a.customer_id
    JOIN products p ON a.product_id = p.product_id
    WHERE c.customer_name = '{customer_name}'
    AND a.account_status = 'ACTIVE';
    """
    current_result = execute_query(current_sql)

    # 2. 새 상품 정보 조회
    new_sql = f"""
    SELECT product_name, product_type, base_rate, max_rate, 
           min_amount, max_amount, min_period_months, max_period_months
    FROM products
    WHERE product_name LIKE '%{new_product_name}%'
    AND is_active = TRUE;
    """
    new_result = execute_query(new_sql)

    # 3. 분석 결과 구성
    lines = []
    lines.append(f"━━━ 갈아타기 분석: {customer_name} → {new_product_name} ━━━\n")
    lines.append("【현재 가입 상품 (ACTIVE)】")
    lines.append(current_result)
    lines.append("")
    lines.append(f"【새 상품 정보: {new_product_name}】")
    lines.append(new_result)
    lines.append("")
    lines.append("━━━ 갈아타기 판단 기준 ━━━")
    lines.append("• 남은 기간이 3개월 이내 → 유지하는 것이 유리합니다")
    lines.append("• 새 상품 금리가 현재보다 1%p 이상 높음 → 갈아타기 고려 가능")
    lines.append("• 중도해지 시 적용 금리(약 40~60%)를 고려해야 합니다")
    lines.append("• 이미 납입한 기간의 이자 손실을 감안해야 합니다")

    return "\n".join(lines)


# ───────────────────────────────────────────
# 7. 우대금리 조건 확인 Tool
# ───────────────────────────────────────────
@tool
def check_bonus_rate_eligibility(customer_name: str) -> str:
    """고객의 우대금리 충족 조건을 확인합니다.
    급여이체, 자동이체, 카드 사용, 주거래 은행 등의 조건과 
    가입 상품의 기본금리/최대금리를 비교하여 추가 충족 가능한 조건을 안내합니다.
    
    Args:
        customer_name: 고객 이름 (예: "고객_001")
    """
    # 1. 고객 조건 조회
    customer_sql = f"""
    SELECT 
        c.customer_name, c.customer_job, c.annual_income, c.income_level,
        c.main_bank_yn, c.salary_transfer_yn, c.auto_transfer_yn,
        c.card_usage_yn, c.marketing_agree_yn, c.transaction_months
    FROM customers c
    WHERE c.customer_name = '{customer_name}';
    """
    customer_result = execute_query(customer_sql)

    # 2. 고객의 ACTIVE 계좌 + 상품 금리 조회
    account_sql = f"""
    SELECT 
        p.product_name, p.product_type,
        p.base_rate, p.max_rate,
        a.applied_rate,
        (p.max_rate - a.applied_rate) as additional_rate_possible
    FROM customers c
    JOIN customer_accounts a ON c.customer_id = a.customer_id
    JOIN products p ON a.product_id = p.product_id
    WHERE c.customer_name = '{customer_name}'
    AND a.account_status = 'ACTIVE';
    """
    account_result = execute_query(account_sql)

    # 3. 분석 결과 구성
    lines = []
    lines.append(f"━━━ {customer_name}님 우대금리 조건 분석 ━━━\n")
    lines.append("【고객 조건 현황】")
    lines.append(customer_result)
    lines.append("")
    lines.append("【가입 상품별 금리 현황】")
    lines.append(account_result)
    lines.append("")
    lines.append("━━━ 일반적인 우대금리 조건 ━━━")
    lines.append("• 급여이체(salary_transfer_yn) → 보통 +0.1~0.3%p")
    lines.append("• 자동이체(auto_transfer_yn) → 보통 +0.1~0.2%p")
    lines.append("• 카드사용(card_usage_yn) → 보통 +0.1~0.2%p")
    lines.append("• 주거래은행(main_bank_yn) → 보통 +0.1~0.3%p")
    lines.append("• 마케팅동의(marketing_agree_yn) → 보통 +0.1%p")
    lines.append("")
    lines.append("※ 정확한 우대금리 조건은 상품별 약관을 확인해주세요.")
    lines.append("  → search_product_info Tool로 약관에서 상세 조건을 검색할 수 있습니다.")

    return "\n".join(lines)


# ───────────────────────────────────────────
# 에이전트별 Tool 그룹
# ───────────────────────────────────────────
PRODUCT_TOOLS = [
    query_database, get_db_schema, search_product_info,
    check_early_termination, check_bonus_rate_eligibility,
]
ANALYSIS_TOOLS = [
    query_database, get_db_schema, search_product_info, calculate_interest,
    check_early_termination, analyze_switch_product, check_bonus_rate_eligibility,
]
RECOMMEND_TOOLS = [
    query_database, get_db_schema, search_product_info,
    check_bonus_rate_eligibility,
]

