"""
Customer 에이전트 전용 도구
고객 프로필, 계좌, 납입 이력 조회
※ 로컬 헬퍼 _nl2sql_query를 통해 PostgreSQL DB에 접근합니다.
"""

import re

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage

import os

from db.postgres_db import DB_SCHEMA, execute_query
from mcp_servers.postgres_mcp_client import call_postgres_mcp_tool
from config.settings import get_llm


NL2SQL_SYSTEM_PROMPT = f"""당신은 PostgreSQL 전문가입니다.
사용자의 자연어 질문을 SQL SELECT 쿼리로 변환해주세요.

{DB_SCHEMA}

## 규칙
- 반드시 SELECT 쿼리만 작성하세요. INSERT, UPDATE, DELETE 등은 절대 사용하지 마세요.
- SQL 쿼리만 출력하세요. 설명이나 다른 텍스트는 포함하지 마세요.
- 코드 블록도 사용하지 마세요. 순수 SQL만 출력하세요.
- 테이블 JOIN이 필요한 경우 적절한 JOIN을 사용하세요.
- 결과가 너무 많을 수 있으므로 LIMIT을 적절히 사용하세요.
- 금액은 원 단위, 소득은 만원 단위임을 기억하세요.
- 고객 이름은 "고객_001" 형식입니다.
"""


def _clean_sql(sql: str) -> str:
    """
    LLM 응답에서 SQL만 추출합니다.
    """
    sql = sql.strip()

    if sql.startswith("```"):
        lines = sql.splitlines()
        sql = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()

    return sql


def _validate_select_sql(sql: str) -> None:
    """
    Customer Agent는 읽기 전용 SELECT 쿼리만 허용합니다.
    """
    normalized = sql.strip().lower()

    if not normalized.startswith("select"):
        raise ValueError("Customer Agent는 SELECT 쿼리만 실행할 수 있습니다.")

    forbidden_keywords = [
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "truncate",
        "create",
        "grant",
        "revoke",
    ]

    pattern = r"\\b(" + "|".join(forbidden_keywords) + r")\\b"

    if re.search(pattern, normalized):
        raise ValueError("허용되지 않은 SQL 키워드가 포함되어 있습니다.")

    if ";" in normalized.rstrip(";"):
        raise ValueError("여러 SQL 문장을 한 번에 실행할 수 없습니다.")


# mcp 추가
"""
Customer Agent
→ NL2SQL
→ MCP Tool execute_select_query
→ db.postgres_db.execute_query
→ PostgreSQL 의 흐름 
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
    sql = _clean_sql(response.content)

    _validate_select_sql(sql)

    use_mcp = os.getenv("USE_MCP_DB", "true").lower() == "true"

    if use_mcp:
        try:
            return call_postgres_mcp_tool(
                "execute_select_query",
                {
                    "sql": sql,
                    "max_rows": 50,
                },
            )
        except Exception as e:
            # MCP 서버가 꺼져 있거나 연결 실패 시 기존 DB 직접 조회로 fallback
            return (
                f"⚠️ MCP DB 조회 실패로 기존 DB 조회로 fallback합니다: {str(e)}\n\n"
                + execute_query(sql)
            )

    return execute_query(sql)


@tool
def get_customer_profile(customer_name: str) -> str:
    """고객의 기본 정보를 조회합니다.

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
    """고객이 현재 가입한 예금 및 적금 계좌 정보를 조회합니다.

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

    Args:
        customer_name: 고객 이름 (예: "고객_001")
    """
    question = (
        f"{customer_name}의 납입 이력 조회: "
        f"상품명, 납입일, 납입금액, 납입회차 "
        f"최근 납입일 순으로 정렬"
    )
    return _nl2sql_query(question)


CUSTOMER_TOOLS = [
    get_customer_profile,
    get_customer_accounts,
    get_payment_history,
]