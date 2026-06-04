"""
Customer 에이전트 전용 도구
고객 프로필, 계좌, 납입 이력 조회
※ 로컬 헬퍼 _nl2sql_query를 통해 PostgreSQL DB에 접근합니다.
"""
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage

from db.postgres_db import DB_SCHEMA, execute_query
from config.settings import get_llm

# ─── NL2SQL 시스템 프롬프트 (Customer 에이전트 내부 캡슐화) ───
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
"""


def _nl2sql_query(question: str) -> str:
    """
    고객 데이터베이스용 NL2SQL 내부 함수
    """
    llm = get_llm()

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

    result = execute_query(sql)
    return result


@tool
def get_customer_profile(customer_name: str) -> str:
    """고객의 기본 정보를 조회합니다.
    나이, 직업, 연소득, 거래 기간, 급여이체 여부 등을 확인할 수 있습니다.

    Args:
        customer_name: 고객 이름 (예: "고객_001")
    """
    question = (
        f"{customer_name}의 기본 정보 조회: "
        f"이름, 생년월일, 직업, 연소득(만원), 소득수준, "
        f"주거래은행 여부, 급여이체 여부, 자동이체 여부, "
        f"카드사용 여부, 마케팅동의 여부, 거래개월수, 월가용저축액"
    )
    return _nl2sql_query(question)


@tool
def get_customer_accounts(customer_name: str) -> str:
    """고객이 현재 가입한 예·적금 계좌 정보를 조회합니다.
    가입일, 만기일, 현재 잔액, 적용 금리, 계좌 상태 등을 확인할 수 있습니다.

    Args:
        customer_name: 고객 이름 (예: "고객_001")
    """
    question = (
        f"{customer_name}의 가입 계좌 정보 조회: "
        f"계좌번호, 상품명, 상품유형, 가입일, 만기일, 계약기간(개월), "
        f"예치금액, 월납입액, 현재잔액, 적용금리, 계좌상태"
    )
    return _nl2sql_query(question)


@tool
def get_payment_history(customer_name: str) -> str:
    """고객의 납입 이력을 조회합니다.
    납입 횟수, 납입 금액, 최근 납입일 등을 확인할 수 있습니다.

    Args:
        customer_name: 고객 이름 (예: "고객_001")
    """
    question = (
        f"{customer_name}의 납입 이력 조회: "
        f"상품명, 납입일, 납입금액, 납입회차 (최근 납입일 순으로 정렬)"
    )
    return _nl2sql_query(question)


# 이 에이전트에 바인딩될 도구 목록
CUSTOMER_TOOLS = [get_customer_profile, get_customer_accounts, get_payment_history]
