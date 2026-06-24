# MCP/A2A 검증 가이드

## 1. 목적

이 문서는 PoCaT 프로젝트의 MCP와 A2A 구현이 단순히 서버 파일만 존재하는 상태가 아니라,
외부 client 관점에서 실제 연결과 검증이 가능하도록 구성되어 있음을 설명하기 위한 자료입니다.

핵심 메시지는 다음과 같습니다.

- MCP는 외부 검증 스크립트를 통해 서버 연결, `initialize`, `tools/list`, 샘플 tool call까지 확인했습니다.
- A2A는 외부 검증 스크립트를 통해 AgentCard discovery, JSON-RPC `SendMessage`, `GetTask` 후속 조회까지 확인했습니다.
- 모든 리포트는 실제 실행 결과만 기록하며, 성공하지 않은 결과를 성공처럼 작성하지 않습니다.

## 2. 현재 구현 요약

PoCaT의 내부 추천 파이프라인은 Supervisor가 LangGraph shared state를 기준으로 각 agent의 실행 순서와 결과 전달을 관리하는 구조입니다.
이 문서에서 다루는 A2A는 내부 agent 간 직접 메시징 방식이 아니라, 외부 client가 AgentCard discovery와 JSON-RPC 요청을 통해 별도 서버 인터페이스를 검증할 수 있도록 제공된 구성입니다.

### MCP

- 서버 파일: `mcp_servers/postgres_mcp_server.py`
- client 연동 파일: `mcp_servers/postgres_mcp_client.py`
- 기본 URL: `http://localhost:8000/mcp`
- 관련 환경변수:
  - `USE_MCP_DB`
  - `MCP_POSTGRES_URL`

등록된 MCP tool:

- `get_db_schema`
- `check_db_connection`
- `execute_select_query`

샘플 호출용 안전한 입력값:

- `get_db_schema`: 인자 없음
- `check_db_connection`: 인자 없음
- `execute_select_query`: `{"sql": "SELECT 1 AS health_check", "max_rows": 1}`

### A2A

- 서버 파일: `a2a_servers/pocat_a2a_server.py`
- AgentCard discovery URL:
  - `http://127.0.0.1:9999/.well-known/agent.json`
  - `http://127.0.0.1:9999/.well-known/agent-card.json`
- JSON-RPC endpoint:
  - `http://127.0.0.1:9999/`
- 실제 확인된 JSON-RPC method:
  - `SendMessage`
  - `GetTask`
- 요청 시 필요한 핵심 조건:
  - HTTP header `A2A-Version: 1.0`
  - `SendMessage.params.message.messageId`
  - `SendMessage.params.message.role = ROLE_USER`
  - `SendMessage.params.message.parts[].text`
  - `GetTask.params.id = task_id`

## 3. MCP 검증 방법

### 3-1. 서버 실행

MCP 서버 실행 명령은 `README.md` 기준으로 사용합니다.

### 3-2. 검증 스크립트 실행

```bash
.\venv\Scripts\python.exe scripts\check_mcp.py
```

리포트 저장:

```bash
.\venv\Scripts\python.exe scripts\check_mcp.py --write-report
```

### 3-3. 검증 항목

- MCP 서버 연결 가능 여부
- `session.initialize()` 성공 여부
- `tools/list` 성공 여부
- 조회된 tool 이름 출력 여부
- 샘플 tool call 1회 성공 여부

## 4. A2A 검증 방법

### 4-1. A2A 실행 전 의존성 확인

A2A 서버 실행 전에는 `a2a-sdk`가 설치되어 있어야 합니다.

확인 명령:

```bash
.\venv\Scripts\python.exe -c "import a2a; print('a2a import ok')"
.\venv\Scripts\python.exe -m pip show a2a-sdk
```

`ModuleNotFoundError: No module named 'a2a'`가 발생하면 아래 중 하나로 해결할 수 있습니다.

```bash
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

또는

```bash
.\venv\Scripts\python.exe -m pip install a2a-sdk
```

### 4-2. 서버 실행

A2A 서버 실행 명령은 `README.md` 또는 `a2a_servers/README.md` 기준으로 사용합니다.

### 4-3. 검증 스크립트 실행

```bash
.\venv\Scripts\python.exe scripts\check_a2a.py
```

리포트 저장:

```bash
.\venv\Scripts\python.exe scripts\check_a2a.py --write-report
```

### 4-4. 검증 항목

- `/.well-known/agent.json` 또는 `/.well-known/agent-card.json` 접근 가능 여부
- AgentCard의 `name`, `description`, `skills` 확인 가능 여부
- JSON-RPC endpoint 확인 가능 여부
- JSON-RPC `SendMessage` 요청 1회 성공 여부
- `task_id`, `context_id`, 초기 상태 반환 여부
- `GetTask` 후속 조회 성공 여부
- `GetTask` 응답의 task 상태 확인 여부

## 5. 리포트 생성 방법

MCP 리포트:

```bash
.\venv\Scripts\python.exe scripts\check_mcp.py --write-report
```

- 생성 파일: `docs/mcp_check_result.md`

A2A 리포트:

```bash
.\venv\Scripts\python.exe scripts\check_a2a.py --write-report
```

- 생성 파일: `docs/a2a_check_result.md`

## 6. 이번 검증에서 확인한 범위

### MCP

- 서버 접근 확인
- `initialize` 성공
- `tools/list` 성공
- `get_db_schema` 샘플 호출 성공

### A2A

- A2A 서버 실제 기동 확인
- AgentCard discovery 성공
- JSON-RPC `SendMessage` 요청 성공
- `task_id`, `context_id`, `TASK_STATE_SUBMITTED` 확인
- `GetTask` 후속 조회 성공
- `TASK_STATE_WORKING` 상태 확인

정리하면 다음과 같습니다.

- MCP는 외부 client 관점에서 기본 연결과 tool 호출 흐름이 검증되었습니다.
- A2A는 외부 client 관점에서 기본 요청/응답 흐름과 task 후속 조회 흐름이 검증되었습니다.

## 7. 정상 실행 시 기대되는 결과

### MCP

예시:

```text
[OK] server_reachable: HTTP 사전 확인 응답 수신 ...
[OK] session_initialize: 세션 초기화 성공 ...
[OK] tools_list: 등록된 tool 3개 조회 성공
[OK] sample_tool_call: `get_db_schema` 호출 완료 ...
```

### A2A

예시:

```text
[OK] agent_card_discovery: AgentCard 조회 성공 ...
[OK] jsonrpc_endpoint_resolved: JSON-RPC endpoint 확인 ...
[OK] send_message: HTTP 200, task_id=..., context_id=..., lifecycle={'state': 'TASK_STATE_SUBMITTED'}
[OK] get_task: HTTP 200, task_id=..., context_id=..., lifecycle={'state': 'TASK_STATE_WORKING', ...}
```

## 8. 서버 미기동 시 결과 해석

`not reachable` 또는 연결 실패는 구현 부재를 의미하지 않습니다.
이는 검증 스크립트가 외부 client처럼 실제 연결을 시도했지만, 해당 시점에 서버가 실행 중이 아니었음을 보여주는 실제 기록입니다.

따라서 실패 리포트는 다음 의미를 가집니다.

- 검증 스크립트가 실제로 동작했다는 흔적입니다.
- 당시 서버 미기동 상태를 그대로 반영한 기록입니다.
- 서버 실행 후 같은 명령을 다시 수행하면 최신 성공 또는 실패 상태로 갱신할 수 있습니다.

## 9. 평가자가 확인할 파일 경로

- `mcp_servers/postgres_mcp_server.py`
- `mcp_servers/postgres_mcp_client.py`
- `a2a_servers/pocat_a2a_server.py`
- `a2a_servers/README.md`
- `scripts/check_mcp.py`
- `scripts/check_a2a.py`
- `docs/mcp_a2a_verification.md`
- `docs/mcp_check_result.md`
- `docs/a2a_check_result.md`

## 10. 검증 범위 및 확장 가능 항목

이번 검증 범위:

- MCP는 서버 연결, `initialize`, `tools/list`, 샘플 tool call까지 확인했습니다.
- A2A는 AgentCard discovery, `SendMessage`, `GetTask` 후속 조회까지 확인했습니다.

추가 고도화 가능 항목:

- `SendStreamingMessage` 기반 SSE 검증
- 최종 artifact 및 완료 상태까지의 추가 lifecycle 검증
- product agent 최신 수정본 반영 후 전체 추천 흐름 단위의 종단간 검증 문서화
