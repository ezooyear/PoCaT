"""
PostgreSQL MCP Server

- PoCaT의 고객 DB 조회 기능을 MCP Tool로 노출합니다.
- 기존 db/postgres_db.py의 validate_sql, execute_query, test_connection을 재사용합니다.
- SELECT 쿼리만 허용하는 기존 보안 정책을 유지합니다.
"""

from mcp.server.fastmcp import FastMCP

from db.postgres_db import DB_SCHEMA, execute_query, test_connection, validate_sql


mcp = FastMCP(
    "PoCaT PostgreSQL MCP Server",
    json_response=True,
    stateless_http=True,
)


@mcp.tool()
def get_db_schema() -> str:
    """
    PoCaT 고객 데이터베이스 스키마 정보를 반환합니다.
    NL2SQL 생성 시 참고할 수 있습니다.
    """
    return DB_SCHEMA


@mcp.tool()
def check_db_connection() -> str:
    """
    PostgreSQL 연결 상태를 확인합니다.
    """
    ok = test_connection()
    if ok:
        return "PostgreSQL 연결 성공"
    return "PostgreSQL 연결 실패"


@mcp.tool()
def execute_select_query(sql: str, max_rows: int = 50) -> str:
    """
    SELECT SQL을 실행하고 결과를 텍스트 표 형태로 반환합니다.

    Args:
        sql: 실행할 SELECT SQL
        max_rows: 최대 반환 행 수
    """
    is_safe, message = validate_sql(sql)
    if not is_safe:
        return f"⚠️ MCP 쿼리 차단: {message}"

    return execute_query(sql, max_rows=max_rows)


if __name__ == "__main__":
    # 공식 SDK 예시와 동일하게 streamable-http transport로 실행합니다.
    # 기본 접속 경로는 일반적으로 http://localhost:8000/mcp 입니다.
    mcp.run(transport="streamable-http")