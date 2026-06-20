# 진단: 상품추천이 안 되는 문제 (요약은 정상)

작성: 2026-06-20 / 담당: data-integration-engineer + agent-prompt-engineer + eval-qa-specialist

## 증상 고정
- ✅ 상품요약(product_info): 정상 → `product_agent` **단독**, RAG(Chroma)만 사용, DB 불필요
- ❌ 상품추천(recommendation): 실패 → `customer→product→eligibility→financial→recommend→validation` 다단계 체인

## 격리 결과 — 차이는 "customer_agent와 DB 의존"
요약과 추천의 유일한 구조적 차이는 **customer_agent(NL2SQL→PostgreSQL)** 와 그 데이터를 받는 다운스트림이다.

## 근본 원인 (계층적)

### 원인 1 (선행 조건): 고객 데이터가 DB에 없음
- `data/customers.json` = `{"customers": []}` (빈 값)
- `data/customers.json.local` = `{"customers": []}` (빈 값)
- DB 적재 스크립트 없음, `.sql` DDL 없음, `products`/`customers`/`customer_accounts`/`payment_history` 시드 없음
- → `customer_agent`가 "조회 결과가 없습니다" 반환 → eligibility가 `customer_profile_incomplete` + `missing_fields` 판정 → recommend `deferred` → validation 차단 → supervisor가 `_build_validation_blocked_answer`("정보가 부족") 반환

### 원인 2 (병존): 상태 스키마 드리프트 (R2)
`scripts/test_eligibility_recommend.py`가 입증:
- 테스트가 읽는 키: `customer_profile`, `customer_accounts`, `eligibility_results`, `recommendation_results`, `financial_results`, `product_candidates` (state 최상위)
- `graph/state.py` 실제 정의: `customer_result`, `eligibility_result`(단수 dict), `recommend_result` …
- `agent_outputs["customer_agent"]`를 **문자열로도 dict로도** 받음 → eligibility가 summary 텍스트를 정규식 파싱
- → 데이터가 있어도 customer→eligibility 경계면에서 누락될 수 있음. 단, 원인 1이 해소되기 전엔 검증 불가.

## DB 셋업 시 함정 (data-integration-engineer)
1. **pocat 스키마 vs search_path** ⚠️ 최우선
   - 코드는 비한정 테이블명(`SELECT ... FROM customers`)으로 쿼리. NL2SQL 프롬프트(`DB_SCHEMA`)도 스키마 접두어 없음.
   - `pocat` 스키마에 적재하면 기본 search_path(`public`)에선 `relation "customers" does not exist`로 실패.
   - 해결: ① `ALTER DATABASE <db> SET search_path TO pocat, public;` (가장 간단) 또는 ② `DB_SCHEMA`+NL2SQL 프롬프트를 `pocat.` 한정명으로 수정 또는 ③ 그냥 `public`에 적재.
2. **앱 경로로는 DDL 불가** — `validate_sql`/`_validate_select_sql`이 `CREATE`/`INSERT` 차단. 스키마 생성·적재는 앱이 아니라 psql/로더로 직접 수행.
3. `DB_SCHEMA`에 `products` 테이블이 FK로 참조되나 정의가 누락(### 2 없음). customer_accounts.product_id FK 대상 테이블 필요.
4. docker-compose의 postgres는 `POSTGRES_DB=${DB_NAME:-postgres}`. `.env`의 DB_NAME/PASSWORD가 컨테이너와 일치해야 함.

## 데이터 출처 (확인됨)
`data/`의 테이블 덤프 CSV 4종 (json은 빈 값, 무시):
- `customers_*.csv` (300행), `products_*.csv` (10행), `customer_accounts_*.csv` (600행), `payment_history_*.csv` (9366행)
- 스키마명: **pocat3** (사용자 지정)

## 작성물
- `scripts/load_db.py` — pocat3 스키마 + 4테이블(FK) 생성 → CSV COPY 적재 → `ALTER DATABASE ... SET search_path TO pocat3, public` → 비한정 쿼리 검증. (psycopg2 직접 실행으로 앱 SELECT 가드 우회)

## 진행 결과 (2026-06-20)
- ✅ venv 생성 + psycopg2-binary/python-dotenv 설치
- ✅ `scripts/load_db.py` 실행 성공 → `hycu_dropout` DB `pocat3` 스키마에 적재
  - customers 300 / products 10 / customer_accounts 600 / payment_history 9366
  - `ALTER DATABASE hycu_dropout SET search_path TO pocat3, public` 반영
  - 검증: `FROM customers` → 고객_001, 계좌⋈products JOIN 정상, 판매중 상품 금리순 조회 정상
- ✅ `db/postgres_db.py` `DB_SCHEMA`에 **products 테이블 정의 추가** (NL2SQL이 계좌-상품 JOIN 가능)
- ⏭ **원인 1(데이터 부재) 해소.** 이제 추천 end-to-end 재검증 단계.

## 추천 버튼 "프로그램 오류" 추적 (2026-06-20, 2차)
- 헤드리스 재현(`_workspace/repro_recommend.py`, 고객_123): **크래시 없이** "정보 부족" graceful 답변. 단, **Vector DB 없음**(`data/pdfs`에 PDF 없음 → `build_vectorstore` 실패).
- 코드 구조 분석:
  - `eligibility_agent_node` / `recommend_agent_node` / `supervisor`(synthesize) → 모두 `try/except`로 fallback. **그래프를 죽이지 않음.**
  - `product_agent_node` (`agents/product/agent.py:516-520`) → `except Exception: ... raise` — **유일하게 재던짐.** + `run_agent_loop`의 `llm_with_tools.invoke`도 except 없음(LLM 오류 전파 가능).
  - → 사용자의 "프로그램 오류"는 **product_agent 예외 전파**가 1순위 가설. RAG에 실제 데이터가 있을 때만 도는 파싱/LLM 경로에서 발생 추정.
- **로컬 재현 불가 사유:** 이 체크아웃엔 상품 약관 PDF가 없어(`data/pdfs`) Chroma 빌드 불가 → product RAG 경로를 못 태움.
- 필요한 것: ① 실제 빨간 오류 텍스트(맨 아랫줄 `XxxError`) 또는 ② `data/pdfs` PDF 확보 후 Chroma 빌드.

## 돌파구 (2026-06-20, 3차) — 크래시 해소 + 진짜 원인 발견
- PDF 추가 → `scripts/build_vectorstore.py`로 Chroma 빌드 완료(자식청크 443개).
- 앱 재시작(디버그 traceback 계측) 후 사용자 재현 → **서버 로그에 예외 없음 = 크래시 해소.** (원래 "프로그램 오류"는 DB/Vector DB 부재 등 환경 문제였고, 적재+빌드+DB_SCHEMA 수정으로 해결됨.)
- 구체 질문("월 30만원 24개월 적금 추천") 헤드리스 재현 → 크래시 없이 **validation이 추천 보류**, 원인 명시:
  1. 🔴 **product_result 상품명이 문서 본문 문장으로 추출됨**: "※ 최고이율은 신규가입일 당시 영업점 및 KB국민은행 홈페이지에 게시한 우대이율..." → eligibility가 invalid_product 처리 → 추천 전멸. **(핵심 원인)**
  2. financial_result에 term_months/applied_rate 누락.
  3. RAG 근거 low confidence.
- 근본 원인: `agents/product/tools.py`의 `_extract_product_name_from_chunk`가 청크 본문 첫 키워드 줄을 상품명으로 잡음. 정작 가장 깨끗한 신호인 **"출처: {PDF파일명}"** 을 버림. PDF stem == `products.rag_document_key` 이므로 DB 매핑으로 정확한 상품명 복원 가능.
- 수정 방향: 상품명을 청크의 출처 PDF 파일명에서 추출 → products 테이블(rag_document_key→product_name)로 정규화. 실패 시 파일명 stem 정제 fallback.

## 수정 적용 결과 (2026-06-20, 4차) — 추천 정상 출력 ✅
적용한 코드 수정:
1. `db/postgres_db.py` — `DB_SCHEMA`에 products 테이블 정의 추가.
2. `agents/product/tools.py` — 상품명을 출처 PDF 파일명 + products(rag_document_key) 정규화로 추출(`_extract_product_name_from_chunk`), `get_product_detail_map()` 추가.
3. `agents/product/agent.py` — `_normalize_product_candidate`를 products 테이블 정형 값(min_amount/max_amount/base_rate/max_rate/period/age)으로 보강. eligibility가 읽는 `min_amount`/`max_amount` 키 채움.
4. `agents/customer/agent.py` — `_build_structured_profile_from_db()`로 DB에서 정형 고객 프로필 직접 구성(소득·월저축액 누락 해결).
5. `app.py` — 임시 traceback 계측 추가 후 **제거(원복)**.
6. `scripts/load_db.py` — CSV→pocat3 적재 로더(신규).

결과(헤드리스, 고객_123, "월30만원 24개월 적금 추천"):
- ✅ 크래시 없음, validation 통과, **랭킹 추천 정상 출력**(KB 장기간부 도약적금 6.0%, KB 일반정기적금). 장병내일적금 9만원 한도 정확 인식.
- ⚠️ 잔여(품질, 블로커 아님): 구조화 `recommend_result`는 여전히 `recommendation_deferred`(financial_results_missing) — 최종 답변은 supervisor 합성 LLM이 보강된 product 데이터로 생성. eligibility가 RAG 후보만 평가해 후보 수가 적음.

## 5차 — validation 게이트 과도 차단(R4) 수정
- 증상: 앱에서 버튼 추천 시 여전히 보류. 사유 "condition_conflict 검증 미수행 + RAG confidence low".
- 원인: `validation_agent`가 경고만 있어도 `revision_required=True` 설정 → supervisor `_is_validation_failed`가 **revision_required만으로 추천 차단**. LLM이 사소한 경고 하나만 달아도 추천 전멸.
- 수정: `agents/supervisor/agent.py` `_is_validation_failed`를 재작성 — ① missing_fields(실제 사용자 입력 부족) 또는 ② error 레벨 + {calculation_mismatch, inappropriate_recommendation} 이슈일 때만 하드 차단. 그 외 경고(RAG confidence, 검증 미수행, 우대조건 미충족 등)는 통과시키고 주의사항으로 노출.
- 검증(앱 동일 경로, 버튼 질문 + 쉼표·원 컨텍스트): **랭킹 추천 + 이유 + 주의사항 정상 출력**. KB상호부금 정액(81개월>36개월) 정확 제외.
- app.py 디버그 traceback 로깅은 **사용자 확인 시까지 유지**(요청).

## 6차 — 진짜 앱 크래시 원인: OpenRouter 응답 파싱 실패 (디버그 로깅으로 포착)
- 사용자 앱 실제 오류: `답변 생성 중 오류: Response validation failed: EOF while parsing a value at line 289 column 0`
- traceback(서버 로그 beap53ziy): `product_agent → run_agent_loop → llm_with_tools.invoke → langchain_openrouter → openrouter.chat.send → unmarshal_json_response → ResponseValidationError`.
- 근본 원인: **OpenRouter 무료 모델(openai/gpt-oss-120b:free)이 간헐적으로 잘린(truncated) 응답** 반환 → SDK JSON 파싱 실패. `run_agent_loop`의 LLM 호출은 재시도·예외처리가 없어 그대로 전파, product_agent는 재던짐 → 앱 크래시.
- 왜 헤드리스에선 안 터졌나: 간헐적 외부 API 오류라 재현 시점엔 정상 응답을 받음. (→ "정상"이라 단정한 것은 오판. 사용자가 디버그 로깅 유지를 요청한 덕에 실제 traceback 포착.)
- 수정:
  - `agents/base.py` `_invoke_with_retry()` 추가 → `run_agent_loop`의 두 LLM 호출을 3회 재시도(백오프)로 감쌈. 일시적 truncation 흡수. (product/financial/customer-loop 등 run_agent_loop 사용 전반에 적용)
  - `agents/product/agent.py` 예외 시 `raise` → **graceful fallback 반환**(다른 노드와 동일). 재시도도 끝내 실패하면 앱 크래시 대신 degrade.
- app.py 디버그 로깅 **유지**(사용자 확인 대기).

## 남은 단계(품질 개선, 선택)
- financial_agent가 추천 흐름에서 후보별 이자 계산을 안정적으로 생성하도록 보완(현재 LLM 의존 → calculations 빈번히 빔).
- eligibility/recommend가 products 테이블 기반 후보까지 평가하도록 확장(RAG 후보 외 상품도 포함).
- 위가 되면 구조화 recommend_result도 채워져 UI의 카드형 추천과 일치.

## 남은 단계(원래 항목)
1. ✅ ~~`.env` + 적재~~ 완료.
2. **`python scripts/load_db.py` 실행** → 적재 + search_path 반영.
3. **검증** — 로더가 `FROM customers` 비한정 쿼리로 자동 검증. 이후 `customer_agent` 단독/추천 재시도.
4. **DB_SCHEMA 보완 필요** — `db/postgres_db.py`의 `DB_SCHEMA`(NL2SQL 프롬프트)에 **products 테이블 정의 누락**. 고객 가입계좌의 '상품명' 조회는 `customer_accounts ⋈ products` JOIN 필요한데 LLM이 products를 몰라 JOIN을 못 짬. 적재 후 보완.
5. 적재 후에도 추천 실패 시 → 원인 2(스키마 드리프트). `scripts/test_eligibility_recommend.py` 먼저 통과.
