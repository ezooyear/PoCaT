## MCP 기반 PostgreSQL 조회

기존에는 Customer Agent가 `db/postgres_db.py`의 `execute_query()`를 직접 호출하여 PostgreSQL을 조회했습니다.

개선 후에는 PostgreSQL 조회 기능을 MCP Tool Server로 분리하고, Customer Agent가 MCP Client를 통해 `execute_select_query` tool을 호출하도록 변경했습니다.

실행 흐름:

```text
Customer Agent
→ NL2SQL
→ PostgreSQL MCP Client
→ PostgreSQL MCP Server
→ db/postgres_db.py
→ PostgreSQL

