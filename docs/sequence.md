# PoCaT Agent 실행 순서 설계문서

> **현재 구현 기준** (2026-06-20)
> 현재 구현은 agent 간 직접 통신(A2A)이 아니라, **Supervisor 통제형 LangGraph 순차 실행**입니다.
> 각 agent는 shared state에 결과를 저장하고, 다음 agent가 이를 읽는 방식으로 협력합니다.

---

## 1. 기본 추천 흐름

### Sequence Diagram

```mermaid
sequenceDiagram
    participant U as User
    participant UI as Streamlit
    participant S as Supervisor
    participant C as CustomerAgent
    participant P as ProductAgent
    participant E as EligibilityAgent
    participant F as FinancialAgent
    participant R as RecommendAgent
    participant V as ValidationAgent
    participant L as Langfuse
    participant DB as PostgreSQL
    participant RAG as ChromaDB

    U->>UI: 추천 요청 입력 (자연어)
    UI->>S: user_query, member_id 전달 (AgentState 초기화)
    S->>L: langfuse_trace_context 시작 (trace_name, session_id)

    Note over S: plan_mode: task_type=recommendation 판단
    S->>S: plan = [customer, product, eligibility, financial, recommend, validation]

    S->>C: customer_agent_node 실행
    C->>DB: get_customer_profile(customer_name)
    DB-->>C: 고객 기본정보 반환
    C->>DB: get_customer_accounts(customer_name)
    DB-->>C: 계좌 목록 반환
    C->>DB: get_payment_history(customer_name)
    DB-->>C: 납입이력 반환
    C-->>S: customer_result.result.customer_profile 저장
    C->>L: span 기록 (customer_agent, status, duration_ms)

    S->>P: product_agent_node 실행
    P->>RAG: search_terms(query) 호출
    RAG-->>P: 약관 유사 발췌 텍스트 반환
    P->>P: extract_product_candidates() + _normalize_product_candidate()
    P-->>S: product_result.result.products 저장
    P->>L: span 기록 (product_agent, product_count, fallback_reason)

    S->>E: eligibility_agent_node 실행
    E->>E: evaluate_eligibility() × 상품 수
    E->>E: evaluate_bonus_rate() × 상품 수
    E->>E: _group_eligibility_results()
    E-->>S: eligibility_result.result (eligible/needs_check/rejected) 저장
    E->>L: span 기록 (eligibility_agent, recommendable_count)

    S->>F: financial_agent_node 실행
    F->>F: run_agent_loop() → calculate_interest() 도구 호출
    F->>F: _ensure_financial_calculations()
    F->>F: _enforce_financial_role_boundary()
    F-->>S: financial_result.result.calculations 저장
    F->>L: span 기록 (financial_agent, calculation_count)

    S->>R: recommend_agent_node 실행
    R->>R: _build_guarded_recommendations()
    R-->>S: recommend_result.result.recommendations 저장
    R->>L: span 기록 (recommend_agent, recommendation_count)

    S->>V: validation_agent_node 실행
    V->>V: run_validation_checks() [Rule 기반]
    V->>V: _run_llm_verify_result() [LLM 기반]
    V-->>S: validation_result.result.is_valid 저장
    V->>L: span 기록 (validation_agent, is_valid, blocking_issues)

    Note over S: synthesize_mode: 모든 결과 합성
    S->>S: _is_validation_failed() 확인
    S->>S: LLM 최종 답변 생성 (SUPERVISOR_SYNTHESIZE_PROMPT)
    S-->>UI: final_answer
    S->>L: trace flush

    UI-->>U: 추천 결과 표시
```

---

## 2. 예외 흐름

### A. RAG 검색 실패 / 상품 추출 실패

Product Agent가 RAG 도구를 호출했지만 상품 파싱에 실패하는 경우:

```mermaid
sequenceDiagram
    participant S as Supervisor
    participant P as ProductAgent
    participant E as EligibilityAgent
    participant R as RecommendAgent
    participant V as ValidationAgent
    participant RAG as ChromaDB

    S->>P: product_agent_node 실행
    P->>RAG: search_terms(query)
    RAG-->>P: 약관 텍스트 반환 (검색 성공)
    P->>P: extract_product_candidates() → 상품명 추출 실패
    P-->>S: product_result (status=fallback, products=[], fallback_reason=empty_product_candidates)

    S->>E: eligibility_agent_node 실행
    E->>E: product_candidates=[] 감지
    E-->>S: eligibility_result (status=needs_check, no_valid_product_candidates)

    Note over S,R: financial_agent 실행 (계산 대상 없음)

    S->>R: recommend_agent_node 실행
    R->>R: eligible_products=[] 감지
    R-->>S: recommend_result (status=no_eligible_product, recommendations=[])

    S->>V: validation_agent_node 실행
    V-->>S: validation_result (status=failed or passed_with_warnings, blocking_issues=[...])

    Note over S: synthesize_mode
    S->>S: 상품 탐색 실패 안내 답변 생성
    S-->>UI: "현재 조건에 맞는 상품을 탐색하지 못했습니다. 다시 시도해주세요."
```

---

### B. financial_result.calculations 누락

EligibilityAgent는 정상 실행되었지만 FinancialAgent가 필요 정보 부족으로 계산 실패하는 경우:

```mermaid
sequenceDiagram
    participant S as Supervisor
    participant E as EligibilityAgent
    participant F as FinancialAgent
    participant R as RecommendAgent
    participant V as ValidationAgent

    S->>E: eligibility_agent_node 실행
    E-->>S: eligibility_result (eligible_products=[KB 스타 적금])

    S->>F: financial_agent_node 실행
    F->>F: calculate_interest() 도구 호출
    F->>F: 납입 기간 / 금리 정보 부족 감지
    F-->>S: financial_result (status=needs_check, calculations=[], missing_fields=[term_months, applied_rate])

    S->>R: recommend_agent_node 실행
    R->>R: financial_result.calculations=[] 확인
    R-->>S: recommend_result (status=recommendation_deferred, fallback_reason=financial_results_missing)

    S->>V: validation_agent_node 실행
    V->>V: financial_result.calculations 없음 확인
    V-->>S: validation_result (status=passed_with_warnings or failed)

    Note over S: synthesize_mode
    S->>S: 정보 부족 안내 + 추천 보류 답변 생성
    S-->>UI: "만기금액 계산을 위해 납입 기간과 금리를 확인해주세요."
```

---

### C. Validation 실패

Recommend Agent는 추천 목록을 생성했지만 Validation Agent가 이를 실패 판정하는 경우:

```mermaid
sequenceDiagram
    participant S as Supervisor
    participant R as RecommendAgent
    participant V as ValidationAgent

    S->>R: recommend_agent_node 실행
    R-->>S: recommend_result (recommendations=[KB 스타 적금])

    S->>V: validation_agent_node 실행
    V->>V: run_validation_checks() [Rule]
    V->>V: _run_llm_verify_result() [LLM]
    V->>V: rag_evidence_missing 이슈 감지
    V-->>S: validation_result (status=failed, is_valid=false, revision_required=true)

    Note over S: synthesize_mode
    S->>S: _is_validation_failed() → True
    S->>S: _build_validation_blocked_answer()
    S-->>UI: "추천 상품의 금리 근거 확인이 필요합니다. 정확한 상품 조건을 재확인 후 안내드립니다."
```

---

## 3. LangGraph 라우팅 메커니즘

```
supervisor_node (plan_mode)
    ↓ _supervisor_router()
    ├─ task_type=casual → supervisor_node (synthesize) → END
    └─ 첫 agent 이름 → customer_agent_node
                            ↓ _route_after("customer_agent")
                            ├─ plan에 다음 agent 있으면 → 다음 agent
                            └─ 없으면 → supervisor_node (synthesize) → END
```

`_route_after(current_node)` 함수는 `state["plan"]`에서 현재 이후의 다음 agent를 찾아 반환합니다.

---

## 4. Langfuse Trace 구조

```
[Trace] pocat_recommendation
    ├─ [Span] supervisor (plan_mode)
    │      metadata: {task_type, plan, next, message_count}
    ├─ [Span] customer_agent
    │      metadata: {agent_name, status, tool_count, tool_names, duration_ms}
    ├─ [Span] product_agent
    │      metadata: {agent_name, status, search_query, rag_result_count,
    │                 product_count, structured_product_count,
    │                 structured_product_names, fallback_reason, duration_ms}
    ├─ [Span] eligibility_agent
    │      metadata: {agent_name, status, result_count, recommendable_count,
    │                 needs_check_count, rejected_count, duration_ms}
    ├─ [Span] financial_agent
    │      metadata: {agent_name, status, calculation_count,
    │                 missing_fields, fallback_reason, duration_ms}
    ├─ [Span] recommend_agent
    │      metadata: {agent_name, status, recommendation_count,
    │                 excluded_count, fallback_reason, duration_ms}
    ├─ [Span] validation_agent
    │      metadata: {agent_name, status, is_valid, revision_required,
    │                 blocking_issue_count, warning_count, duration_ms}
    └─ [Span] supervisor (synthesize_mode)
           metadata: {validation_blocked, final_answer_length, duration_ms}
```

각 agent가 Langfuse에 기록하는 metadata 기준:

| 필드 | 설명 |
|------|------|
| `agent_name` | agent 식별자 |
| `status` | success / fallback / needs_check / failed |
| `result_key` | state에 저장되는 result 키 이름 |
| `fallback_reason` | fallback 사유 (없으면 null) |
| `input_source` | 어떤 state 키를 읽었는지 |
| `tool_count` | 도구 호출 횟수 |
| `tool_error_count` | 도구 오류 횟수 |
| `duration_ms` | agent 실행 시간(ms) |
| `input_preview` | 입력 요약 (200자 이내) |
| `output_preview` | 출력 요약 (200자 이내) |

> state 전체나 대용량 텍스트는 metadata에 포함하지 않습니다.

---

## 5. Task Type별 실행 Agent 순서

| Task Type | 실행 순서 |
|-----------|-----------|
| casual | supervisor만 (직접 응답) |
| customer_lookup | customer |
| product_info | product |
| financial_analysis | customer → financial → validation |
| eligibility_check | customer → product → eligibility → validation |
| **recommendation** | **customer → product → eligibility → financial → recommend → validation** |
| early_termination | customer → product → financial → recommend → validation |
| switch_analysis | customer → financial → product → eligibility → recommend → validation |

---

## 6. 현재 구현상 주의사항

1. **순차 실행, A2A 아님**: 현재 구현은 agent 간 직접 통신이 없습니다. Supervisor가 계획(`plan`)을 수립하면 LangGraph가 순서대로 agent를 실행하고, 각 agent는 shared state를 읽고 씁니다.

2. **Shared State 기반 데이터 전달**: Product Agent의 결과(`product_result`)를 Eligibility Agent가 state에서 직접 읽습니다. 별도 메시지 패싱이나 큐가 없습니다.

3. **Supervisor 2회 실행**: Supervisor는 plan_mode(계획 수립)와 synthesize_mode(최종 답변 합성) 두 번 실행됩니다. 그래프에서 같은 노드를 두 번 진입합니다.

4. **LLM 검증 조건부 실행**: Validation Agent의 LLM 검증은 복잡한 task_type(`recommendation`, `financial_analysis` 등)에서만 실행되며, 단순 조회(`customer_lookup`, `casual`)는 Rule 검증만 실행합니다.

5. **Langfuse 비활성화 시 영향 없음**: `.env`에 Langfuse 키가 없으면 모든 trace 호출이 자동으로 no-op 처리되며 시스템 동작에 영향이 없습니다.

---

## 확인한 주요 코드 위치

- [app.py](../app.py) — 그래프 실행 진입점, session_id 관리
- [graph/builder.py](../graph/builder.py) — `_supervisor_router()`, `_route_after()`, 노드 등록
- [graph/state.py](../graph/state.py) — AgentState TypedDict
- [agents/supervisor/agent.py](../agents/supervisor/agent.py) — `_plan_mode()`, `_synthesize_mode()`, `_is_validation_failed()`
- [agents/base.py](../agents/base.py) — `run_agent_loop()` (공통 LLM 루프)
- [agents/customer/agent.py](../agents/customer/agent.py)
- [agents/product/agent.py](../agents/product/agent.py) — `_determine_fallback_reason()`
- [agents/eligibility/agent.py](../agents/eligibility/agent.py) — `_build_guarded_eligibility_results()`
- [agents/financial/agent.py](../agents/financial/agent.py) — `_ensure_financial_calculations()`, `_enforce_financial_role_boundary()`
- [agents/recommend/agent.py](../agents/recommend/agent.py) — `_build_guarded_recommendations()`
- [agents/validation/agent.py](../agents/validation/agent.py) — `run_validation_checks()`, `_run_llm_verify_result()`
- [observability/langfuse.py](../observability/langfuse.py) — `langfuse_observation()`, `update_observation()`
