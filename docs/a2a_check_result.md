# A2A 검증 결과

이 문서는 **A2A 서버를 실제로 실행한 상태에서** 외부 client 관점으로 `scripts/check_a2a.py --write-report`를 수행한 최신 결과입니다.
아래 내용은 실제 실행 결과 기준입니다.

## 실행 정보

- 실행 명령: `.\venv\Scripts\python.exe scripts\check_a2a.py --write-report`
- 실행 시각(UTC): `2026-06-24 16:09:49 UTC`
- Base URL: `http://127.0.0.1:9999`
- 전체 상태: `PASS`

## 이번 검증에서 확인한 범위

- A2A 서버는 실제 기동 확인됨
- AgentCard discovery 성공
- JSON-RPC `SendMessage` 요청 성공
- `task_id`, `context_id`, `TASK_STATE_SUBMITTED` 확인
- `GetTask` 후속 조회 성공
- `TASK_STATE_WORKING` 상태 확인

따라서 외부 client 관점에서 A2A 서버와의 기본 요청/응답 흐름은 검증되었습니다.

## 단계별 결과

- PASS | `agent_card_discovery` | AgentCard 조회 성공 (`http://127.0.0.1:9999/.well-known/agent.json`)
- PASS | `jsonrpc_endpoint_resolved` | JSON-RPC endpoint 확인 (`http://127.0.0.1:9999`)
- PASS | `send_message` | HTTP 200, `task_id=9f9bda55-cfcf-4386-9e14-4522b66e32f4`, `context_id=95a09003-7a06-4325-a77d-efbb1a627b9d`, `lifecycle={'state': 'TASK_STATE_SUBMITTED'}`
- PASS | `get_task` | HTTP 200, `task_id=9f9bda55-cfcf-4386-9e14-4522b66e32f4`, `context_id=95a09003-7a06-4325-a77d-efbb1a627b9d`, `lifecycle={'state': 'TASK_STATE_WORKING', ...}`

## 실제 확인된 요청 방식

- JSON-RPC method:
  - `SendMessage`
  - `GetTask`
- 필수 헤더: `A2A-Version: 1.0`
- 주요 요청 필드:
  - `SendMessage.params.message.messageId`
  - `SendMessage.params.message.role = ROLE_USER`
  - `SendMessage.params.message.parts[].text`
  - `GetTask.params.id = task_id`

## 요청 헤더

```json
{
  "Content-Type": "application/json",
  "A2A-Version": "1.0"
}
```

## 추출된 식별자

- task_id: `9f9bda55-cfcf-4386-9e14-4522b66e32f4`
- context_id: `95a09003-7a06-4325-a77d-efbb1a627b9d`

## SendMessage 요청

```json
{
  "jsonrpc": "2.0",
  "id": "pocat-a2a-sendmessage-check",
  "method": "SendMessage",
  "params": {
    "message": {
      "messageId": "4b9170cc-be0b-4aa3-a8a7-49b8437140e5",
      "role": "ROLE_USER",
      "parts": [
        {
          "text": "A2A connectivity check. Please return a short acknowledgement.",
          "mediaType": "text/plain"
        }
      ]
    },
    "configuration": {
      "acceptedOutputModes": [
        "text/plain"
      ],
      "returnImmediately": true
    }
  }
}
```

## SendMessage 응답

```json
{
  "result": {
    "task": {
      "id": "9f9bda55-cfcf-4386-9e14-4522b66e32f4",
      "contextId": "95a09003-7a06-4325-a77d-efbb1a627b9d",
      "status": {
        "state": "TASK_STATE_SUBMITTED"
      },
      "history": [
        {
          "messageId": "4b9170cc-be0b-4aa3-a8a7-49b8437140e5",
          "contextId": "95a09003-7a06-4325-a77d-efbb1a627b9d",
          "taskId": "9f9bda55-cfcf-4386-9e14-4522b66e32f4",
          "role": "ROLE_USER",
          "parts": [
            {
              "text": "A2A connectivity check. Please return a short acknowledgement.",
              "mediaType": "text/plain"
            }
          ]
        }
      ]
    }
  },
  "id": "pocat-a2a-sendmessage-check",
  "jsonrpc": "2.0"
}
```

## GetTask 요청

```json
{
  "jsonrpc": "2.0",
  "id": "pocat-a2a-gettask-check",
  "method": "GetTask",
  "params": {
    "id": "9f9bda55-cfcf-4386-9e14-4522b66e32f4"
  }
}
```

## GetTask 응답

```json
{
  "result": {
    "id": "9f9bda55-cfcf-4386-9e14-4522b66e32f4",
    "contextId": "95a09003-7a06-4325-a77d-efbb1a627b9d",
    "status": {
      "state": "TASK_STATE_WORKING",
      "message": {
        "messageId": "5dfc3745-c272-45a2-bd7b-3b3037a412e6",
        "role": "ROLE_AGENT",
        "parts": [
          {
            "text": "PoCaT 상담 그래프를 실행 중입니다."
          }
        ]
      },
      "timestamp": "2026-06-24T16:09:49.136043Z"
    },
    "history": [
      {
        "messageId": "4b9170cc-be0b-4aa3-a8a7-49b8437140e5",
        "contextId": "95a09003-7a06-4325-a77d-efbb1a627b9d",
        "taskId": "9f9bda55-cfcf-4386-9e14-4522b66e32f4",
        "role": "ROLE_USER",
        "parts": [
          {
            "text": "A2A connectivity check. Please return a short acknowledgement.",
            "mediaType": "text/plain"
          }
        ]
      }
    ]
  },
  "id": "pocat-a2a-gettask-check",
  "jsonrpc": "2.0"
}
```

## 검증 범위 및 확장 가능 항목

이번 검증은 외부 client 기준으로 AgentCard discovery와 `SendMessage` 기반 task 생성, `GetTask` 후속 조회까지 확인하는 범위입니다.
`SendStreamingMessage` 기반 스트리밍과 최종 artifact 검증은 향후 task lifecycle을 더 세밀하게 확인하기 위한 추가 고도화 가능 항목으로 분리했습니다.
