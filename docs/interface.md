# PoCaT Agent 간 Interface / Schema 설계문서

> **현재 구현 기준** (2026-06-20)
> 이 문서는 agent 간 schema 불일치 방지를 위해 실제 코드 기준으로 작성되었습니다.
> downstream agent는 summary markdown을 재파싱하지 않고 아래 명시된 structured key를 우선 읽어야 합니다.

---

## 1. Interface 개요

PoCaT의 agent들은 **LangGraph AgentState (TypedDict)**를 통해 결과를 전달합니다.

- 각 agent는 `{agent}_result` 키에 구조화된 결과를 저장합니다.
- 동시에 `agent_outputs["{agent}_agent"]`에도 동일 결과를 저장합니다 (중복 경로 허용).
- downstream agent는 summary markdown(LLM 응답 텍스트)을 재파싱하지 않고 `result` 내 structured key를 우선 읽습니다.
- 모든 agent 결과는 공통 envelope 구조를 따릅니다.

---

## 2. 공통 응답 Envelope

**`agents/base.py`의 `make_agent_result()`** 가 생성하는 공통 구조:

```json
{
  "status": "success | needs_check | fallback | failed | error",
  "summary": "LLM이 생성한 요약 텍스트 또는 fallback 안내",
  "result": {
    "summary": "...",
    "tool_results": [
      {
        "tool_name": "도구명",
        "tool_args": {},
        "tool_result": "도구 실행 결과 텍스트"
      }
    ]
  },
  "evidence": [],
  "error": null
}
```

> **주의**: `result` 내부의 모든 키는 최상단에도 복사됩니다.
> 예: `result.customer_profile`이 있으면 최상단 `customer_profile`도 동시에 존재합니다.
> Read path는 `result.X` → 최상단 `X` 순서로 시도합니다.

---

## 3. AgentState 공유 필드 정의

**`graph/state.py`** 기준:

```python
class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]   # 대화 히스토리
    next: Optional[str]                        # 다음 실행 agent
    user_query: Optional[str]
    task_type: Optional[str]                   # casual | recommendation | ...
    plan: Optional[list[str]]                  # 실행 예정 agent 순서
    current_step: Optional[int]
    current_agent: Optional[str]
    completed_agents: Optional[list[str]]
    member_id: Optional[str]
    customer_id: Optional[int]
    context: Optional[dict]
    agent_outputs: Optional[dict]              # 누적된 모든 agent 결과
    customer_result: Optional[dict]
    product_result: Optional[dict]
    financial_result: Optional[dict]
    eligibility_result: Optional[dict]
    recommend_result: Optional[dict]
    validation_result: Optional[dict]
    draft_answer: Optional[str]
    final_answer: Optional[str]
    agent_logs: Optional[list]
    errors: Optional[list]
```

---

## 4. Agent별 Input / Output Schema

### 4-1. Customer Agent

**Input** (state에서 읽음):
- `state["member_id"]` 또는 `state["customer_id"]`
- `state["messages"]` (대화 히스토리)

**Output** (`state["customer_result"]`에 저장):

```json
{
  "status": "success",
  "summary": "고객_223 정보 조회 완료. 나이 21세, 직업 군인, ...",
  "result": {
    "summary": "...",
    "tool_results": [
      { "tool_name": "get_customer_profile", "tool_args": { "customer_name": "고객_223" }, "tool_result": "..." },
      { "tool_name": "get_customer_accounts", "tool_args": { "customer_name": "고객_223" }, "tool_result": "..." },
      { "tool_name": "get_payment_history",   "tool_args": { "customer_name": "고객_223" }, "tool_result": "..." }
    ],
    "customer_profile": {
      "customer_name": "고객_223",
      "birth_date": "2004-03-15",
      "age": 21,
      "job": "군인",
      "income": 2900,
      "income_level": "낮음",
      "monthly_saving_amount": 100000,
      "transaction_months": 7,
      "salary_transfer": true,
      "auto_transfer": false,
      "card_usage": false,
      "main_bank": false,
      "marketing_agree": false,
      "is_soldier": true,
      "raw_text": "...",
      "parsed_customer_fields": ["age", "job", "income"],
      "parsed_customer_values": {}
    }
  },
  "evidence": [],
  "error": null
}
```

**Read path** (`customer_profile`):
```
1순위: state["customer_result"]["result"]["customer_profile"]
2순위: state["customer_result"]["customer_profile"]
3순위: state["agent_outputs"]["customer_agent"]["result"]["customer_profile"]
4순위: state["agent_outputs"]["customer_agent"]["customer_profile"]
```

---

### 4-2. Product Agent

**Input** (state에서 읽음):
- `state["user_query"]`
- `state["customer_result"]` (선택: 고객 조건 참고용)

**Output** (`state["product_result"]`에 저장):

```json
{
  "status": "success",
  "summary": "KB 스타 적금, 군인 정기적금 등 2개 상품 후보 탐색 완료",
  "result": {
    "summary": "...",
    "tool_results": [
      { "tool_name": "search_terms", "tool_args": { "query": "적금 군인 우대" }, "tool_result": "약관 발췌 텍스트..." }
    ],
    "products": [
      {
        "product_id": "prod_001",
        "product_name": "KB 스타 적금",
        "bank": "KB국민은행",
        "product_type": "적금",
        "base_rate": 2.0,
        "max_rate": 2.5,
        "base_rate_text": "연 2.0%",
        "max_rate_text": "연 2.5%",
        "min_monthly_amount": 100000,
        "max_monthly_amount": 300000,
        "term_options_months": [12, 24, 36],
        "eligibility_text": "실명의 개인",
        "preferential_conditions_text": "급여이체 시 0.5% 우대",
        "preferential_conditions": [
          { "name": "급여이체", "condition": "급여이체 등록", "rate": 0.5 }
        ],
        "evidence": [
          { "source_file": "kb_star_savings.pdf", "page": 2, "source_pages": [2, 3], "evidence": [] }
        ],
        "source": "rag_search",
        "raw_text": "..."
      }
    ],
    "product_candidates": [],
    "structured_product_count": 1,
    "structured_product_names": ["KB 스타 적금"],
    "searched_products": []
  },
  "evidence": [],
  "error": null
}
```

**Read path** (`products`):
```
1순위: state["product_result"]["result"]["products"]
2순위: state["product_result"]["products"]
3순위: state["agent_outputs"]["product_agent"]["result"]["products"]
4순위: state["agent_outputs"]["product_agent"]["products"]
```

**Fallback 상태값**:
- `status="fallback"`: RAG 검색 자체가 실패하거나 미호출
- `status="error"`: 검색은 됐지만 상품 파싱 완전 실패
- `products=[]`: 추천 불가 상황 전달 (downstream이 no_eligible 처리)

---

### 4-3. Eligibility Agent

**Input** (state에서 읽음):
- `state["customer_result"]["result"]["customer_profile"]`
- `state["product_result"]["result"]["products"]`

**Output** (`state["eligibility_result"]`에 저장):

```json
{
  "status": "success",
  "summary": "KB 스타 적금 가입 가능, 군인 정기적금 추가 확인 필요",
  "result": {
    "status": "eligible",
    "summary": "...",
    "results": [
      {
        "product_name": "KB 스타 적금",
        "eligible": true,
        "status": "eligible",
        "reasons": [],
        "missing_fields": [],
        "invalid_fields": [],
        "bonus_conditions_met": ["salary_transfer"],
        "bonus_conditions_missing": [],
        "check_required": [],
        "ineligibility_reasons": [],
        "source_agent": "eligibility_agent"
      }
    ],
    "eligible_products": ["KB 스타 적금"],
    "needs_check_products": [],
    "rejected_products": [],
    "invalid_products": [],
    "result_count": 1,
    "recommendable_count": 1,
    "needs_check_count": 0,
    "rejected_count": 0,
    "invalid_product_count": 0,
    "customer_profile": {},
    "customer_accounts": [],
    "product_candidates": [],
    "fallback_reason": null,
    "missing_fields": [],
    "invalid_fields": []
  },
  "evidence": [],
  "error": null
}
```

**product `status` 분류 기준**:
- `eligible`: 가입 가능, 우대조건 충족 확인됨
- `needs_check`: 조건 불확실 (프로필 필드 부족, 약관 불명확)
- `rejected`: 명백한 부적격 (나이 미달, 직업 불일치 등)
- `invalid_product`: 상품명이 비정상 (본문 문장, 빈 문자열 등)

---

### 4-4. Financial Agent

**Input** (state에서 읽음):
- `state["product_result"]["result"]["products"]`
- `state["customer_result"]["result"]["customer_profile"]`
- `state["eligibility_result"]["result"]["eligible_products"]`

**Output** (`state["financial_result"]`에 저장):

```json
{
  "status": "success",
  "summary": "KB 스타 적금 12개월 적립 시 예상 만기수령액 약 1,216,250원",
  "result": {
    "summary": "...",
    "tool_results": [
      {
        "tool_name": "calculate_interest",
        "tool_args": {
          "product_name": "KB 스타 적금",
          "product_type": "적금",
          "monthly_payment": 100000,
          "months": 12,
          "annual_rate": 2.5
        },
        "tool_result": "예상 이자: 16,250원, 만기수령액: 1,216,250원"
      }
    ],
    "calculations": [
      {
        "product_name": "KB 스타 적금",
        "product_type": "적금",
        "monthly_amount": 100000,
        "term_months": 12,
        "payment_count": 12,
        "applied_rate": 2.5,
        "base_rate": 2.0,
        "bonus_rate": 0.5,
        "principal": 1200000,
        "estimated_interest_before_tax": 16250,
        "estimated_interest_after_tax": 13748,
        "estimated_maturity_amount": 1216250,
        "calculation_method": "monthly_installment_simple_estimate",
        "calculation_note": "세전 이자 기준 단리 추정"
      }
    ],
    "missing_fields": [],
    "fallback_reason": null
  },
  "evidence": [],
  "error": null
}
```

**Read path** (`calculations`):
```
1순위: state["financial_result"]["result"]["calculations"]
2순위: state["financial_result"]["calculations"]
3순위: state["agent_outputs"]["financial_agent"]["result"]["calculations"]
4순위: state["agent_outputs"]["financial_agent"]["calculations"]
```

**`needs_check` 반환 조건**: `term_months`, `applied_rate`, `monthly_payment` 중 하나 이상 없는 경우

---

### 4-5. Recommend Agent

**Input** (state에서 읽음):
- `state["eligibility_result"]["result"]["eligible_products"]`
- `state["financial_result"]["result"]["calculations"]`
- `state["product_result"]["result"]["products"]`

**Output** (`state["recommend_result"]`에 저장):

```json
{
  "status": "success",
  "summary": "KB 스타 적금 1위 추천. 급여이체 우대 조건 충족으로 최대 금리 적용 가능.",
  "result": {
    "status": "recommended",
    "summary": "...",
    "recommendations": [
      {
        "rank": 1,
        "product_name": "KB 스타 적금",
        "product_type": "적금",
        "recommendation_status": "recommended",
        "reason": "급여이체 우대조건 충족으로 최대금리 2.5% 적용, 12개월 만기 1,216,250원 수령 예상",
        "eligibility_status": "eligible",
        "applied_rate": 2.5,
        "estimated_maturity_amount": 1216250,
        "estimated_interest_before_tax": 16250,
        "warnings": [],
        "source_product_name": "KB 스타 적금"
      }
    ],
    "recommended_products": [],
    "recommendation_count": 1,
    "excluded_products": [
      {
        "product_name": "군인 정기적금",
        "status": "needs_check",
        "reason": "복무기간 6개월 미만 여부 확인 필요",
        "source_agent": "eligibility_agent"
      }
    ],
    "fallback_reason": null,
    "required_next_steps": []
  },
  "evidence": [],
  "error": null
}
```

**`status` 값 정의**:
- `recommended`: 추천 목록 생성됨
- `needs_more_info`: eligibility 또는 product 결과 없음
- `recommendation_deferred`: financial_result.calculations 없음
- `no_eligible_product`: eligible 상품 0개

---

### 4-6. Validation Agent

**Input** (state에서 읽음):
- 전체 state (모든 `{agent}_result` 키 포함)
- `state["plan"]`, `state["completed_agents"]`

**Output** (`state["validation_result"]`에 저장):

```json
{
  "status": "passed",
  "summary": "검증 통과. 모든 필수 항목 확인됨.",
  "result": {
    "verify_result": {
      "status": "passed",
      "is_valid": true,
      "summary": "...",
      "issues": [],
      "checked_items": {
        "common_format_checked": true,
        "required_results_checked": true,
        "plan_completion_checked": true,
        "recorded_errors_checked": true,
        "recommendation_consistency_checked": true,
        "condition_conflict_checked": true,
        "rate_amount_payment_checked": true,
        "rag_evidence_checked": true,
        "inappropriate_recommendation_checked": true
      },
      "final_notes": [],
      "revision_required": false
    },
    "is_valid": true,
    "issues": [],
    "failure_reasons": [],
    "warnings": [],
    "checks": {},
    "revision_required": false,
    "failure_type": "passed",
    "missing_fields": [],
    "blocking_issues": [],
    "awaiting_user_input": false
  },
  "error": null
}
```

**`status` 값 정의**:
- `passed`: 검증 통과
- `passed_with_warnings`: 통과하나 경고 존재
- `failed`: 검증 실패 (Supervisor가 확정 추천 차단)

**LLM 검증이 실행되는 task_type**: `financial_analysis`, `eligibility_check`, `recommendation`, `early_termination`, `switch_analysis`

**LLM 검증이 실행되지 않는 task_type**: `customer_lookup`, `product_info`, `casual` (Rule 검증만)

---

## 5. Read Path 우선순위

nested result 구조로 인해 downstream agent는 아래 우선순위로 접근합니다.

### Customer Profile

```
1순위: state["customer_result"]["result"]["customer_profile"]
2순위: state["customer_result"]["customer_profile"]
3순위: state["agent_outputs"]["customer_agent"]["result"]["customer_profile"]
4순위: state["agent_outputs"]["customer_agent"]["customer_profile"]
```

### Product Candidates

```
1순위: state["product_result"]["result"]["products"]
2순위: state["product_result"]["products"]
3순위: state["product_result"]["result"]["product_candidates"]
4순위: state["agent_outputs"]["product_agent"]["result"]["products"]
```

### Eligibility Results

```
1순위: state["eligibility_result"]["result"]["results"]
2순위: state["eligibility_result"]["results"]
3순위: state["eligibility_results"]
4순위: state["agent_outputs"]["eligibility_agent"]["result"]["results"]
```

### Financial Calculations

```
1순위: state["financial_result"]["result"]["calculations"]
2순위: state["financial_result"]["calculations"]
3순위: state["agent_outputs"]["financial_agent"]["result"]["calculations"]
4순위: state["agent_outputs"]["financial_agent"]["calculations"]
```

### Recommendations

```
1순위: state["recommend_result"]["result"]["recommendations"]
2순위: state["recommend_result"]["recommendations"]
3순위: state["recommendation_results"]
4순위: state["agent_outputs"]["recommend_agent"]["result"]["recommendations"]
```

---

## 6. Error / Fallback Interface

fallback 결과도 반드시 공통 envelope 구조를 유지해야 합니다.
downstream이 `status` 필드로 fallback 여부를 판단하므로, `status` 필드는 항상 존재해야 합니다.

### Product Agent fallback 예시

```json
{
  "status": "fallback",
  "summary": "RAG 검색 도구가 호출되지 않았습니다.",
  "result": {
    "products": [],
    "product_candidates": [],
    "structured_product_count": 0,
    "structured_product_names": [],
    "fallback_reason": "tool_not_called",
    "tool_results": []
  },
  "evidence": [],
  "error": "RAG search tool was not invoked by LLM"
}
```

### Financial Agent needs_check 예시

```json
{
  "status": "needs_check",
  "summary": "계산에 필요한 정보(납입 기간, 금리)가 부족합니다.",
  "result": {
    "calculations": [],
    "missing_fields": ["term_months", "applied_rate"],
    "fallback_reason": "missing_required_calculation_fields",
    "tool_results": []
  },
  "evidence": [],
  "error": null
}
```

### Recommend Agent deferral 예시

```json
{
  "status": "success",
  "summary": "금융 계산 결과가 없어 추천을 보류합니다.",
  "result": {
    "status": "recommendation_deferred",
    "recommendations": [],
    "recommendation_count": 0,
    "fallback_reason": "financial_results_missing",
    "required_next_steps": ["금융 계산 결과 확보 필요"],
    "excluded_products": []
  },
  "evidence": [],
  "error": null
}
```

### Validation Agent failed 예시

```json
{
  "status": "failed",
  "summary": "추천 상품의 금리 근거가 RAG에서 확인되지 않습니다.",
  "result": {
    "is_valid": false,
    "revision_required": true,
    "failure_type": "agent_output_error",
    "blocking_issues": [
      {
        "level": "error",
        "type": "rag_evidence_missing",
        "message": "KB 스타 적금의 base_rate 2.5% 근거가 약관에서 확인되지 않음",
        "related_agent": "product_agent",
        "suggestion": "RAG 검색 재시도 또는 금리 수동 확인 필요"
      }
    ],
    "failure_reasons": ["rag_evidence_missing"],
    "warnings": []
  },
  "error": "검증 이슈 발견"
}
```

---

## 7. Langfuse Metadata Interface

각 agent가 `update_observation()`으로 기록하는 최소 metadata:

```json
{
  "agent_name": "product_agent",
  "status": "fallback",
  "result_key": "product_result",
  "fallback_reason": "tool_not_called",
  "input_source": "user_query + customer_result",
  "tool_count": 0,
  "tool_names": [],
  "tool_error_count": 0,
  "search_query": null,
  "search_result_count": 0,
  "rag_result_count": 0,
  "product_count": 0,
  "structured_product_count": 0,
  "structured_product_names": [],
  "duration_ms": 1240,
  "input_preview": "KB 적금 추천해주세요 (21세, 군인)",
  "output_preview": "RAG 검색 도구 미호출"
}
```

> state 전체나 대용량 텍스트는 포함하지 않습니다.

---

## 확인한 주요 코드 위치

- [graph/state.py](../graph/state.py) — AgentState 전체 필드 정의
- [agents/base.py](../agents/base.py) — `make_agent_result()`, `run_agent_loop()`
- [agents/customer/agent.py](../agents/customer/agent.py) — customer_result 구조
- [agents/product/agent.py](../agents/product/agent.py) — product_result 구조, fallback 분기
- [agents/eligibility/agent.py](../agents/eligibility/agent.py) — eligibility_result 구조, 그룹화 로직
- [agents/financial/agent.py](../agents/financial/agent.py) — financial_result 구조, calculations 파싱
- [agents/recommend/agent.py](../agents/recommend/agent.py) — recommend_result 구조, deferred 로직
- [agents/validation/agent.py](../agents/validation/agent.py) — validation_result 구조, Rule+LLM 검증
- [observability/langfuse.py](../observability/langfuse.py) — update_observation metadata
