# 🏦 예적금 상담 멀티 에이전트 (Deposit Agent)

LangGraph 기반의 **Supervisor 통제형 A2A(Agent-to-Agent) 협업 루프** 아키텍처를 적용한 지능형 예적금 상담 AI 어시스턴트입니다. 
자연어 데이터베이스 조회(NL2SQL)와 문서 기반 검색(RAG)을 통해 사람 수준의 금융 컨설팅을 제공합니다.

---

## 🛠 기술 스택

- **오케스트레이션**: LangGraph, LangChain
- **LLM**: OpenRouter
- **정형 데이터 조회 (NL2SQL)**: PostgreSQL (고객 정보 전용)
- **비정형 데이터 검색 (RAG)**: ChromaDB, HuggingFace Embeddings (상품 조건 및 약관 전용)
- **프론트엔드 UI**: Streamlit

---

## 🏗 시스템 아키텍처

### 1. 멀티 에이전트 구조 (Supervisor 통제형 A2A 패턴)
Supervisor가 실행 계획을 수립하면, 에이전트들이 직접 바톤을 넘기며 협업(A2A)한 뒤 최종 결과를 Supervisor가 취합합니다.

- **`Supervisor`**: 사용자의 질문을 분석하여 실행 계획(Plan)을 수립하고, 6개의 전문 에이전트를 조합 실행한 뒤 최종 응답을 취합합니다.
- **`Customer Agent`**: 고객의 프로필, 계좌 현황, 납입 이력 데이터를 PostgreSQL DB에서 **SQL 기반으로 전담 조회**합니다 (내부에 `_nl2sql_query` 함수를 독점 캡슐화).
- **`Product Agent`**: 모든 상품 가입 요건, 설명서, 약관 등을 **RAG 검색(`search_terms`)을 통해서만 조회**합니다 (PostgreSQL DB 직접 조회 방지).
- **`Eligibility Agent`**: Customer Agent가 수집해 준 고객 정보와 Product Agent가 RAG로 검색해 준 상품 조건을 대조하여 **가입 가능 여부, 우대 금리 요건, 가입 상품 리스트 필터링**을 연산합니다.
- **`Financial Agent`**: 이자 계산, 만기/납입현황, 중도해지 손실 분석, 상품 비교 및 갈아타기 유불리를 순수 비교 연산합니다.
- **`Recommend Agent`**: 고객의 조건과 한도에 맞는 맞춤 상품 선별 및 추천 순위를 매깁니다.
- **`Validation Agent`**: 결과 적합성 및 계산 정확성 검증과 할루시네이션 탐지를 전담 수행합니다.

### 2. 하이브리드 데이터 접근 격리
- **NL2SQL (고객 데이터)**:
  - 개인정보 및 보유 계좌 잔액 등 정형 데이터는 LLM이 자연어를 SQL로 변환하여 PostgreSQL에서 정확하게 조회합니다.
  - 안전 장치: `SELECT` 쿼리만 허용하며 위험한 키워드(`DROP`, `DELETE` 등)는 차단합니다.
- **RAG (금융상품 데이터)**:
  - 수시로 변하는 우대금리 요건, 가입 나이/금액 제한, 약관 상세 내용은 ChromaDB(Vector Store) RAG 검색을 통해서만 조회하여 SQL DB의 상품 정보 스키마 의존성을 완전히 제거했습니다.

---

## ✨ 에이전트별 전용 도구 (Tools)

각 에이전트는 SQL/RAG 조회에 직접 의존하지 않고, 전달받은 인자(Arguments)를 기반으로 자율적으로 호출 및 연산합니다.

| 에이전트 | 전용 도구 | 역할 및 데이터 협업 방식 |
|---|---|---|
| **Customer** | `get_customer_profile`, `get_customer_accounts`, `get_payment_history` | 고객의 프로필, 계좌, 납입 이력을 DB에서 조회 (내부 `_nl2sql_query` 호출) |
| **Product** | `search_terms` (RAG 전용) | 상품 약관 PDF를 기반으로 가입 조건, 기간, 한도, 유의사항 RAG 검색 |
| **Eligibility** | `evaluate_eligibility`, `evaluate_bonus_rate`, `filter_eligible_products` | Customer 데이터와 Product RAG 데이터를 대조하여 자격 여부 및 상품 목록 필터링 연산 |
| **Financial** | `calculate_interest`, `compare_products`, `compare_switch_benefit` | 단순 이자 연산, RAG 기반 상품 간 스펙 일대일 비교 및 갈아타기 손익 시뮬레이션 |
| **Recommend** | `rank_products` | RAG 상품 스펙 정보 전체와 고객의 투자 목적/가용한도를 연산하여 추천 순위 점수화 |
| **Validation** | `verify_result` | 수치 연산 결과 검증 및 외부 정보 생성(할루시네이션) 방지 검증 |

---

## 📁 프로젝트 구조

```text
deposit_agent/
├── agents/                          # 에이전트별 디렉토리 (완전 모듈화 구조)
│   ├── base.py                      # 공통 도구 호출 루프 (보일러플레이트 제거)
│   ├── supervisor/
│   │   ├── agent.py                 # 계획 수립 / 최종 취합
│   │   └── prompts.py               # Supervisor 시스템 프롬프트
│   ├── customer/                    # 고객 데이터 전담
│   │   ├── agent.py
│   │   ├── prompts.py
│   │   └── tools.py                 # get_customer_profile, accounts, history (내부에 nl2sql 캡슐화)
│   ├── product/                     # 상품 데이터 전담 (RAG 전용)
│   │   ├── agent.py
│   │   ├── prompts.py
│   │   └── tools.py                 # search_terms (RAG 검색 전용)
│   ├── eligibility/                 # 자격 판단
│   │   ├── agent.py
│   │   ├── prompts.py
│   │   └── tools.py                 # evaluate_eligibility, bonus_rate, filter_eligible_products
│   ├── financial/                   # 계산/분석
│   │   ├── agent.py
│   │   ├── prompts.py
│   │   └── tools.py                 # calculate_interest, compare_products, compare_switch_benefit
│   ├── recommend/                   # 추천
│   │   ├── agent.py
│   │   ├── prompts.py
│   │   └── tools.py                 # rank_products
│   └── validation/                  # 검증
│       ├── agent.py
│       ├── prompts.py
│       └── tools.py                 # verify_result
├── config/                          # 전역 설정 및 LLM 인스턴스 초기화
├── data/                            # 상품 약관 PDF 및 ChromaDB 저장소
├── db/                              # 데이터 모듈
│   ├── postgres_db.py               # PostgreSQL 연결 및 실행
│   └── vectorstore.py               # RAG 문서 로드 및 Vector DB 구축
├── graph/                           # LangGraph 상태(State) 정의 및 그래프 빌더
│   ├── state.py                     # AgentState 정의
│   └── builder.py                   # A2A 협업 루프 그래프 구성
├── scripts/                         # 유틸리티 스크립트 (build_vectorstore.py)
├── app.py                           # Streamlit 웹 기반 사용자 인터페이스
├── .env                             # 환경 변수 및 설정 (DB, API Key 등)
└── requirements.txt                 # 파이썬 의존성 패키지
```

---

## 🚀 설치 및 실행

### 1. 가상환경 및 패키지 설치
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
```

### 2. 환경 변수 설정
프로젝트 루트 경로에 `.env` 파일을 생성하고 아래 내용을 입력합니다.
```env
# OpenRouter API Key
OPENROUTER_API_KEY=your_openrouter_api_key

# 사용할 LLM 모델
LLM_MODEL=openai/gpt-oss-120b:free

# PostgreSQL DB 접속 정보
DB_HOST=localhost
DB_PORT=5432
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=your_db_password
```

### 3. Vector DB 구축 (최초 1회)
상품 약관 PDF 데이터를 기반으로 ChromaDB를 생성합니다.
```bash
python scripts/build_vectorstore.py
```

### 4. Streamlit 웹 앱 실행
```bash
streamlit run app.py
```

---

## ✅ 구현 단계 (Roadmap)
- [x] Phase 1: 기본 챗봇 기반 마련 및 환경 설정
- [x] Phase 2: LangGraph 멀티 에이전트 아키텍처(Supervisor 패턴) 도입
- [x] Phase 3: 도구(Tool) 기반 응답 체계 구축
- [x] Phase 4: PostgreSQL NL2SQL 및 하이브리드 RAG 연동
- [x] Phase 5: 고급 금융 분석 기능(중도해지, 갈아타기, 이자계산) 구현 및 Streamlit UI 통합
- [x] Phase 6: 에이전트 역할 세분화 (6개 에이전트) 및 도구 특화 적용
- [x] Phase 7: Customer Agent 신설, 에이전트별 디렉토리 구조 리팩터링, Supervisor 통제형 A2A 협업 루프 도입
- [x] Phase 8: 상품 정보 RAG 전용화(DB SQL 조회 차단), 루트 tools 폴더 삭제 및 customer_agent 내부 캡슐화 완료
