## Supervisor / Validation 담당 수정 사항

### 수정 파일 목록

```text
1. graph/state.py
2. agents/base.py
3. agents/validation/agent.py
4. agents/validation/prompts.py
5. agents/validation/tools.py
6. agents/supervisor/prompts.py
7. agents/supervisor/agent.py
8. graph/builder.py
9. Supervisor Plan 생성 테스트

---

## 1. `graph/state.py`

### 수정 내용

멀티 에이전트 간 데이터 공유와 Supervisor/Validation 흐름 관리를 위해 `AgentState` 구조를 확장했습니다.

### 주요 변경 사항

* `TypedDict(total=False)` 구조로 변경하여 초기 state에 모든 필드가 없어도 동작할 수 있도록 유연성 확보
* Supervisor 실행 관리를 위한 필드 추가

  * `user_query`
  * `task_type`
  * `plan`
  * `current_step`
  * `current_agent`
  * `completed_agents`
  * `next`
* 고객 식별 정보 분리

  * `member_id`: 현재 코드에서 사용하는 회원 식별자
  * `customer_id`: DB의 `customers` 테이블 기준 고객 ID
* Agent별 구조화 결과 저장 필드 추가

  * `customer_result`
  * `product_result`
  * `financial_result`
  * `eligibility_result`
  * `recommend_result`
  * `validation_result`
* Validation 및 최종 응답 생성을 위한 필드 추가

  * `draft_answer`
  * `final_answer`
* 디버깅 및 오류 추적을 위한 필드 추가

  * `agent_logs`
  * `errors`
* 기존 에이전트 간 전체 출력 기록을 위해 `agent_outputs` 유지

### 수정 이유

기존에는 `agent_outputs` 중심으로 에이전트 결과가 저장되어 Validation Agent가 각 결과를 구조적으로 검증하기 어려웠습니다.
이에 따라 각 Agent의 결과를 별도 필드에 저장하도록 state를 확장하여, 추후 조건 충돌, 계산 결과, RAG 근거, 추천 적합성 등을 검증할 수 있는 기반을 마련했습니다.

---

## 2. `agents/base.py`

### 수정 내용

모든 전문 Agent가 공통으로 사용하는 실행 로직을 보강했습니다.

### 주요 변경 사항

* `make_agent_result()` 함수 추가

  * Agent 결과를 공통 포맷으로 저장하기 위한 함수
  * 공통 포맷:

```python
{
    "status": "success" 또는 "failed",
    "result": {},
    "evidence": [],
    "error": None 또는 str
}
```

* `run_agent_loop()`에 `result_key` 인자 추가

  * 각 Agent 결과를 `customer_result`, `product_result`, `financial_result`, `eligibility_result`, `recommend_result`, `validation_result` 등에 저장할 수 있도록 확장
* Tool 실행 결과를 `tool_results`로 수집

  * 어떤 Tool이 어떤 입력값으로 실행되었고, 어떤 결과를 반환했는지 기록
* `agent_outputs` 저장 방식 보강

  * 단순 문자열 저장에서 `summary`, `tool_results`를 함께 저장하는 구조로 확장
* Supervisor 실행 추적을 위한 값 업데이트

  * `current_step`
  * `current_agent`
  * `completed_agents`
* Tool 실행 중 오류 발생 시 `errors`에 기록하도록 보강

### 수정 이유

기존 공통 실행 로직은 Agent의 최종 답변만 저장하는 구조에 가까웠습니다.
Validation Agent가 각 Agent의 결과와 근거를 확인할 수 있도록, 공통 실행 단계에서 구조화된 결과와 Tool 실행 기록을 함께 남기도록 수정했습니다.

---

## 3. `agents/validation/agent.py`

### 수정 내용

Validation Agent가 이전 Agent 결과를 검증 대상으로 받아 LLM 기반 검증을 수행하고, 검증 결과를 state에 저장하도록 수정했습니다.

### 주요 변경 사항

* `agent_outputs`에 저장된 이전 Agent 결과들을 검증 대상으로 구성
* `VALIDATION_SYSTEM_PROMPT`에 이전 Agent 결과를 추가하여 LLM이 검증할 수 있도록 처리
* `run_agent_loop()` 호출 시 `output_key="validation_agent"` 사용
* `result_key="validation_result"`를 추가하여 검증 결과가 `state["validation_result"]`에 저장되도록 보강

### 수정 이유

기존에는 Validation 결과가 전체 출력 기록인 `agent_outputs`에만 남을 가능성이 있었습니다.
이번 수정으로 Validation 결과를 별도 필드인 `validation_result`에 구조화하여 저장할 수 있게 되었고, 이후 Supervisor가 검증 결과를 바탕으로 최종 응답 생성 또는 보완 여부를 판단할 수 있도록 했습니다.

---

## 4. `agents/validation/prompts.py`

### 수정 내용

Validation Agent의 역할과 검증 기준을 명확히 하는 시스템 프롬프트를 정리했습니다.

### 주요 검증 기준

* Agent 결과 누락 여부 확인
* 고객 조건, 상품 조건, 가입 가능 여부, 추천 결과 간 모순 확인
* 계산 결과의 오류 여부 확인
* RAG 근거 없이 단정한 상품 정보 탐지
* 부적절하거나 과도하게 확정적인 금융 추천 표현 확인
* 검증할 수 없는 항목은 무리하게 단정하지 않도록 안내

### 수정 이유

Validation Agent는 단순 문장 검토가 아니라, 멀티 에이전트 결과의 적합성, 계산 정확성, 근거 여부, 할루시네이션 가능성을 검증해야 합니다.
이에 따라 LLM이 검증 시 참고할 역할과 기준을 프롬프트에 명확히 반영했습니다.

---

## 5. `agents/validation/tools.py`

### 수정 내용

Validation Agent 전용 LLM 기반 검증 Tool을 유지 및 정리했습니다.

### 주요 내용

* `@tool` 기반 `verify_result()` 유지
* 다른 Agent가 생성한 결과 텍스트를 입력받아 LLM이 검증하도록 구성
* 검증 항목:

  * 계산 오류
  * 논리 오류
  * 정보 오류
  * 추천 오류
  * 할루시네이션 여부
* `VALIDATION_TOOLS = [verify_result]` 형태로 Validation Agent에 바인딩할 Tool 목록 유지

### 수정 이유

본 프로젝트는 LLM 기반 멀티 에이전트 상담 시스템이므로, Validation 역시 단순 규칙 기반 검사만이 아니라 LLM을 활용한 결과 검증 흐름을 유지하는 것이 적절합니다.
다만 현재는 팀원들의 Customer/Product/Financial/Eligibility/Recommend Agent 코드가 모두 취합되지 않은 상태이므로, 세부적인 조건 충돌, 금리/금액/납입 횟수 일치, RAG 근거 검증은 추후 각 Agent 결과 구조가 확정된 뒤 보강할 예정입니다.

## 6. `agents/supervisor/prompts.py`

### 수정 내용

Supervisor가 사용자 질문을 역할표 기준으로 분류하고, 질문 유형별 실행 계획을 생성할 수 있도록 시스템 프롬프트를 수정했습니다.

### 주요 변경 사항

* Supervisor 역할을 단순 라우터가 아니라 계획 수립 및 최종 취합 담당으로 명확화
* 사용 가능한 `task_type`을 역할표 기준으로 정리

  * `casual`
  * `customer_lookup`
  * `product_info`
  * `financial_analysis`
  * `eligibility_check`
  * `recommendation`
  * `early_termination`
  * `switch_analysis`
* 각 Agent의 역할을 명확히 구분

  * `customer_agent`: 고객 기본 정보, 금융 조건, 가입 계좌, 납입 이력 조회
  * `product_agent`: 상품 기본 정보, 금리, 가입 조건, 약관/RAG 근거 조회
  * `financial_agent`: 이자, 만기, 납입 현황, 중도해지 손실, 갈아타기 비교 계산
  * `eligibility_agent`: 고객 조건과 상품 조건 비교, 가입 가능 여부 및 우대금리 충족 여부 판단
  * `recommend_agent`: 가입 가능 후보를 목적별로 순위화하고 추천 이유 및 주의사항 생성
  * `validation_agent`: 조건 충돌, 금리/금액/납입 횟수 일치, RAG 근거, 부적절 추천 여부 검증
* 기존 프롬프트의 구체적인 예시 방식을 유지하되, 현재 Agent 역할표에 맞게 실행 계획 예시를 수정
* `plan` 최대 길이를 기존 3단계에서 6단계로 확장
* 추천, 가입 가능 여부, 금융 분석, 중도해지, 갈아타기 분석에는 `validation_agent`가 마지막에 포함되도록 명시
* 개인화 추천에서는 `eligibility_agent` 결과를 `recommend_agent`가 우선 반영하도록 명시
* “내 가입 상품”, “내 계좌”, “내 상품”과 같은 고객 기준 질문은 `product_agent`가 아니라 `customer_agent`부터 실행되도록 명시

### 질문 유형별 실행 계획 예시

```python
# 일반 대화
["FINISH"]

# 고객 가입 상품 / 계좌 조회
["customer_agent"]

# 상품 정보 / 금리 / 약관 조회
["product_agent"]

# 고객 기준 금융 분석
["customer_agent", "financial_agent", "validation_agent"]

# 가입 가능 여부 판단
["customer_agent", "product_agent", "eligibility_agent", "validation_agent"]

# 개인화 상품 추천
["customer_agent", "financial_agent", "product_agent", "eligibility_agent", "recommend_agent", "validation_agent"]

# 중도해지 손실 분석
["customer_agent", "product_agent", "financial_agent", "recommend_agent", "validation_agent"]

# 갈아타기 분석
["customer_agent", "financial_agent", "product_agent", "eligibility_agent", "recommend_agent", "validation_agent"]
```

### 수정 이유

기존 Supervisor 프롬프트는 `analysis_agent`, `calculation_agent` 등 과거 구조가 일부 남아 있었고, 현재 역할표 기준의 `financial_agent`, `eligibility_agent`, `validation_agent` 흐름이 충분히 반영되어 있지 않았습니다.

이번 수정으로 Supervisor가 질문 의도를 더 명확히 분류하고, State 기반 A2A 협업 흐름에 맞는 Agent 실행 순서를 생성할 수 있도록 했습니다.

---

## 7. `agents/supervisor/agent.py`

### 수정 내용

Supervisor Agent가 질문 의도 분석, 실행 계획 생성, Agent 순서 결정, State 확인, Validation 결과 반영, 최종 답변 생성을 담당하도록 수정했습니다.

### 주요 변경 사항

* Supervisor를 방식 A 구조에 맞게 구성

```text
START
→ supervisor(plan_mode)
→ plan 순서대로 agent 실행
→ 마지막 agent 실행 후 supervisor(synthesize_mode)
→ END
```

* `supervisor_node()`에서 `plan` 존재 여부에 따라 동작 분리

  * `plan`이 없으면 `_plan_mode()` 실행
  * `plan`이 있으면 `_synthesize_mode()` 실행
* `_plan_mode()` 추가 및 보강

  * 사용자 질문에서 `user_query` 추출
  * 규칙 기반 routing을 우선 적용
  * 필요한 경우 LLM 기반 plan 생성으로 보완
  * `task_type`, `plan`, `next`, `current_step`, `current_agent`, `completed_agents`, `agent_outputs` 저장
* `_synthesize_mode()` 추가 및 보강

  * Agent 실행 후 State에 저장된 결과를 종합
  * `customer_result`, `product_result`, `financial_result`, `eligibility_result`, `recommend_result`, `validation_result`, `agent_outputs`를 함께 확인
  * `SUPERVISOR_SYNTHESIZE_PROMPT`의 `{agent_results}` 자리에 State 결과를 문자열로 변환하여 삽입
  * 최종 답변을 `draft_answer`, `final_answer`에 저장
* `TASK_TYPES`를 현재 역할표 기준으로 정리

```python
TASK_TYPES = [
    "casual",
    "customer_lookup",
    "product_info",
    "financial_analysis",
    "eligibility_check",
    "recommendation",
    "early_termination",
    "switch_analysis",
]
```

* `calculation` 대신 `financial_analysis`를 사용하도록 수정

  * `financial_agent`가 단순 계산뿐 아니라 이자, 만기, 납입 현황, 중도해지 손실, 갈아타기 비교까지 담당하기 때문에 `financial_analysis`가 더 적절하다고 판단
* 과거 Agent명 보정 로직 추가

  * `analysis_agent` → `customer_agent`, `financial_agent`
  * `calculation_agent` → `financial_agent`
  * `supervisor_final`은 plan에서 제거
* `validation_agent` 위치 보정

  * 검증이 필요한 task_type에서는 `validation_agent`가 항상 마지막에 오도록 보정
* 일반 대화 처리 로직 추가

  * `casual` 또는 `["FINISH"]` plan일 경우 `SUPERVISOR_DIRECT_RESPONSE_PROMPT`를 사용해 직접 응답 생성

### 수정 이유

Supervisor가 단순히 다음 Agent 하나만 선택하는 구조로는 복합 질문을 처리하기 어렵습니다.
따라서 사용자 질문을 분석해 전체 실행 계획을 만들고, Agent들이 State를 통해 순차 협업한 뒤, 마지막에 Supervisor가 검증 결과까지 반영해 최종 답변을 생성하도록 수정했습니다.

---

## 8. `graph/builder.py`

### 수정 내용

Graph Builder를 Supervisor 통제형 A2A 협업 구조에 맞게 수정했습니다.

### 주요 변경 사항

* 전체 실행 흐름을 방식 A 구조로 정리

```text
START
→ Supervisor(계획 수립)
→ Agent1
→ Agent2
→ ...
→ Supervisor(최종 취합)
→ END
```

* 기존 `current_step` 기반 라우팅의 위험성 보완

  * 기존 방식은 `current_step`이 증가하지 않으면 동일 Agent가 반복 실행될 수 있는 문제가 있었습니다.
  * 이를 방지하기 위해 현재 실행된 Agent 이름을 기준으로 plan에서 다음 Agent를 찾는 `_route_after(current_node)` 방식으로 수정했습니다.
* Agent 간 직접 라우팅 구조 유지

  * Supervisor가 최초 plan을 생성한 뒤, Agent들은 plan 순서대로 직접 연결됩니다.
  * 마지막 Agent 실행 후에는 Supervisor로 돌아가 최종 취합을 수행합니다.
* `supervisor_final` 별도 노드를 사용하지 않고, Supervisor가 최종 취합까지 담당하는 구조로 정리했습니다.

### 수정된 흐름 예시

추천 질문의 경우 다음과 같이 실행됩니다.

```text
START
→ supervisor
→ customer_agent
→ financial_agent
→ product_agent
→ eligibility_agent
→ recommend_agent
→ validation_agent
→ supervisor
→ END
```

### 수정 이유

현재 프로젝트의 목표는 State 기반 A2A 협업 구조입니다.
따라서 Supervisor가 매 단계마다 개입하기보다는, 최초에 실행 계획을 만들고 Agent들이 State를 통해 결과를 주고받으며 순차 실행되는 구조가 더 적합합니다.

다만 마지막에는 Supervisor가 다시 호출되어 `validation_result`를 포함한 전체 State를 확인하고 최종 답변을 생성하도록 구성했습니다.

---

## 9. Supervisor Plan 생성 테스트

### 테스트 내용

팀원들의 Agent 코드가 모두 취합되기 전, Supervisor가 질문 의도에 맞게 `task_type`, `plan`, `next`를 생성하는지 임시 테스트 파일을 통해 확인했습니다.

테스트 파일:

```text
test_supervisor.py
```

해당 파일은 Supervisor 동작 확인을 위한 임시 테스트 파일이며, 커밋 대상에서는 제외하는 것을 권장합니다.

### 테스트 결과

```text
질문: 안녕하세요
task_type: casual
plan: ['FINISH']
next: FINISH

질문: 내 가입 상품 보여줘
task_type: customer_lookup
plan: ['customer_agent']
next: customer_agent

질문: 내 적금 만기까지 얼마나 남았어?
task_type: financial_analysis
plan: ['customer_agent', 'financial_agent', 'validation_agent']
next: customer_agent

질문: 나한테 맞는 적금 추천해줘
task_type: recommendation
plan: ['customer_agent', 'financial_agent', 'product_agent', 'eligibility_agent', 'recommend_agent', 'validation_agent']
next: customer_agent

질문: 지금 해지하면 손해가 얼마나 나?
task_type: early_termination
plan: ['customer_agent', 'product_agent', 'financial_agent', 'recommend_agent', 'validation_agent']
next: customer_agent

질문: KB Star 정기예금 우대금리 조건 알려줘
task_type: product_info
plan: ['product_agent']
next: product_agent

질문: 나 이 상품 가입 가능해?
task_type: eligibility_check
plan: ['customer_agent', 'product_agent', 'eligibility_agent', 'validation_agent']
next: customer_agent

질문: 지금 상품 유지하는 게 좋아, 갈아타는 게 좋아?
task_type: switch_analysis
plan: ['customer_agent', 'financial_agent', 'product_agent', 'eligibility_agent', 'recommend_agent', 'validation_agent']
next: customer_agent
```

### 확인 결과

Supervisor가 질문 유형을 역할표 기준으로 정상 분류하고, 각 질문에 맞는 Agent 실행 계획을 생성하는 것을 확인했습니다.

---

## 현재까지의 전체 작업 의도

이번 작업의 목적은 Supervisor와 Validation을 각각 따로 구현하는 것이 아니라, 두 흐름이 연결될 수 있는 기반을 마련하는 것입니다.

핵심 구조는 다음과 같습니다.

```text
Supervisor가 질문 의도 분석 및 실행 계획 생성
→ Agent들이 State 기반 A2A 방식으로 순차 협업
→ 각 Agent 결과를 구조화하여 State에 저장
→ Validation Agent가 전체 결과 검증
→ Supervisor가 validation_result를 반영해 최종 답변 생성
```

즉 이번 작업은 다음 방향을 기반으로 합니다.

```text
State 기반 구조화 정보
+
A2A 순차 협업
+
LLM 기반 Validation
+
Supervisor 최종 취합
```

---


---

## 추후 보강 예정 사항

팀원들의 Customer/Product/Financial/Eligibility/Recommend Agent 코드가 취합된 뒤 아래 작업을 추가로 진행할 예정입니다.

* 각 Agent 호출부가 `run_agent_loop()`의 `result_key`를 올바르게 사용하는지 확인

  * Customer Agent → `customer_result`
  * Product Agent → `product_result`
  * Financial Agent → `financial_result`
  * Eligibility Agent → `eligibility_result`
  * Recommend Agent → `recommend_result`
  * Validation Agent → `validation_result`
* 각 Agent가 `agent_outputs`를 덮어쓰지 않고 merge하는지 확인
* `product_result.evidence` 기반 RAG 근거 검증 고도화
* `financial_result` 기반 금리, 금액, 납입 횟수 검증 고도화
* `eligibility_result`와 `recommend_result` 비교를 통한 가입 불가능 상품 추천 여부 검증
* `validation_result`를 반영한 Supervisor 최종 답변 품질 확인
* 실제 DB/RAG 연동 후 end-to-end 테스트 수행
