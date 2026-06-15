## A2A 서버 실행

PoCaT는 내부적으로 LangGraph 기반 멀티 에이전트 워크플로우를 사용합니다.
외부 A2A 통신 표준을 지원하기 위해, 전체 PoCaT 상담 그래프를 하나의 A2A Agent로 감싼 서버를 제공합니다.

A2A 서버를 실행하면 Agent Card를 통해 PoCaT Agent의 이름, 설명, 기능(skill), 입출력 방식 등을 외부 클라이언트가 조회할 수 있습니다.

### 주요 구조

```text
A2A Client
  ↓
PoCaT A2A Server
  ↓
LangGraph Supervisor
  ↓
Customer / Product / Eligibility / Financial / Recommend / Validation Agent
```

A2A 서버는 Streamlit UI와 별도로 실행됩니다.
Streamlit은 사용자가 직접 사용하는 챗봇 화면이고, A2A 서버는 외부 Agent 또는 평가자가 PoCaT을 A2A Agent로 확인하거나 호출할 수 있도록 제공하는 별도 진입점입니다.

### 설치

`requirements.txt`에 다음 패키지가 포함되어 있어야 합니다.

```txt
a2a-sdk[http-server]
```

패키지 설치는 아래 명령어로 진행합니다.

```bash
python -m pip install -r requirements.txt
```

A2A SDK가 정상 설치되었는지 확인하려면 다음 명령어를 실행합니다.

```bash
python -c "import a2a; print('A2A SDK imported successfully')"
```

### A2A 서버 실행

프로젝트 루트 디렉토리에서 다음 명령어를 실행합니다.

```bash
python -m a2a_servers.pocat_a2a_server
```

정상 실행 시 다음과 같은 주소에서 서버가 실행됩니다.

```text
http://127.0.0.1:9999
```

### Agent Card 확인

A2A 서버가 실행 중인 상태에서 다른 터미널을 열고 아래 명령어를 실행합니다.

```bash
curl http://127.0.0.1:9999/.well-known/agent.json
```

또는 다음 경로로도 Agent Card를 확인할 수 있습니다.

```bash
curl http://127.0.0.1:9999/.well-known/agent-card.json
```

정상 실행되면 다음과 같은 정보가 JSON 형태로 반환됩니다.

```text
name
description
url
version
capabilities
defaultInputModes
defaultOutputModes
skills
supportsAuthenticatedExtendedCard
```

### 제공 Skill

현재 PoCaT A2A Agent는 다음 기능을 Agent Card에 공개합니다.

| Skill ID                        | 설명                                                                    |
| ------------------------------- | --------------------------------------------------------------------- |
| `pocat_deposit_recommendation`  | 고객 정보, 보유 계좌, 상품 조건, 가입 가능 여부, 금융 계산 결과를 바탕으로 KB 예적금 상품 추천 상담을 수행합니다. |
| `pocat_customer_product_lookup` | 고객 ID를 기준으로 현재 가입한 예금·적금 상품과 계좌 정보를 조회합니다.                            |
| `pocat_product_compare`         | 예금·적금 상품의 금리, 가입 조건, 우대 조건, 기간, 납입 방식을 비교합니다.                         |

### Streamlit / MCP / A2A 실행 관계

PoCaT 실행 시 서버 역할은 다음과 같이 구분됩니다.

| 구분        | 역할                                           | 기본 주소                   |
| --------- | -------------------------------------------- | ----------------------- |
| MCP 서버    | Customer Agent가 PostgreSQL DB를 조회하기 위한 도구 서버 | `http://127.0.0.1:8000` |
| Streamlit | 사용자가 직접 이용하는 챗봇 UI                           | `http://localhost:8501` |
| A2A 서버    | PoCaT 전체 그래프를 외부 A2A Agent로 공개하는 서버          | `http://127.0.0.1:9999` |

Streamlit 챗봇만 실행할 경우에는 A2A 서버가 필수는 아닙니다.
다만 A2A 구조를 시연하거나 Agent Card를 확인하려면 A2A 서버를 별도 터미널에서 실행해야 합니다.

### 전체 시연 시 실행 예시

터미널 1에서 MCP 서버를 실행합니다.

```bash
python -m mcp_servers.postgres_mcp_server
```

터미널 2에서 Streamlit을 실행합니다.

```bash
streamlit run app.py
```

터미널 3에서 A2A 서버를 실행합니다.

```bash
python -m a2a_servers.pocat_a2a_server
```

A2A Agent Card 확인은 다음 명령어로 진행합니다.

```bash
curl http://127.0.0.1:9999/.well-known/agent.json
```

### 현재 구현 범위

현재 A2A 구현은 PoCaT 내부 LangGraph 멀티 에이전트 워크플로우를 외부 A2A 표준 형식으로 공개하기 위한 wrapper입니다.
즉, 기존 Streamlit 챗봇 구조를 대체하는 것이 아니라, 동일한 PoCaT 상담 그래프를 외부 Agent가 인식하고 호출할 수 있도록 Agent Card와 A2A endpoint를 제공하는 역할을 합니다.
