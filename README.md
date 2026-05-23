# 🏦 예적금 상담 멀티 에이전트 (Deposit Agent)

LangGraph 기반의 Supervisor 아키텍처를 적용한 지능형 예적금 상담 AI 어시스턴트입니다. 
자연어 데이터베이스 조회(NL2SQL)와 문서 기반 검색(RAG)을 통해 사람 수준의 금융 컨설팅을 제공합니다.

---

## 🛠 기술 스택

- **오케스트레이션**: LangGraph, LangChain
- **LLM**: OpenRouter (GPT 등 다양한 모델 지원)
- **정형 데이터 조회 (NL2SQL)**: PostgreSQL
- **비정형 데이터 검색 (RAG)**: ChromaDB, HuggingFace Embeddings
- **프론트엔드 UI**: Streamlit

---

## 🏗 시스템 아키텍처

### 1. 멀티 에이전트 구조 (Supervisor 패턴)
사용자의 질문 의도를 분석하여 가장 적합한 전문 에이전트에게 라우팅합니다.

- **`Supervisor`**: 사용자의 질문을 분석하고 3개의 전문 에이전트 중 하나로 라우팅하거나 직접 응답(일상 대화)
- **`Product Agent` (상품 조회)**: 고객의 가입 상품 현황, 잔액, 만기일 등 단순 조회 담당
- **`Analysis Agent` (분석/비교)**: 금리 비교, 이자 계산, 중도해지 손실 분석, 갈아타기 분석 등 복잡한 계산 담당
- **`Recommend Agent` (추천/안내)**: 조건 및 목적별 맞춤 상품 추천, 우대금리 분석, 약관 및 예금자보호 안내

### 2. 하이브리드 데이터 접근
- **NL2SQL (`query_database`)**: 
  - LLM이 자연어를 SQL로 변환하여 PostgreSQL에서 고객 정보, 계좌 현황, 상품 기본 스펙 등을 정확하게 조회합니다.
  - 안전 장치: `SELECT` 쿼리만 허용하며 위험한 키워드(`DROP`, `DELETE` 등)는 자동으로 차단합니다.
- **RAG (`search_product_info`)**: 
  - DB에 담기 힘든 길고 복잡한 상품 약관, 상세 우대금리 조건, 중도해지 규정 등을 ChromaDB(Vector Store)에서 검색하여 제공합니다.

---

## ✨ 핵심 기능 (Tools)

에이전트들은 상황에 맞게 7가지 전문 도구(Tool)를 자율적으로 선택하여 호출합니다.

1. `query_database`: 자연어 질문을 SQL로 변환하여 DB 조회
2. `get_db_schema`: LLM이 SQL을 작성할 수 있도록 테이블 구조 확인
3. `search_product_info`: PDF 상품 약관에서 상세 조건 검색
4. `calculate_interest`: 단리/복리, 예금/적금 예상 이자 및 세후 수령액 계산
5. `check_early_termination`: 현재 가입된 상품 중도해지 시 예상 손실 분석
6. `analyze_switch_product`: 기존 상품 유지 vs 새 상품 갈아타기 유불리 분석
7. `check_bonus_rate_eligibility`: 고객의 급여이체, 카드사용 등 조건을 파악하여 추가 적용 가능한 우대금리 분석

---

## 📁 프로젝트 구조

```text
deposit_agent/
├── agents/              # 전문 에이전트 및 Supervisor 노드
├── config/              # 전역 설정 및 LLM 인스턴스 초기화
├── data/                # 상품 약관 PDF 및 ChromaDB 저장소
├── db/                  # 데이터 모듈
│   ├── postgres_db.py   # PostgreSQL 연결 및 NL2SQL 쿼리 검증/실행
│   └── vectorstore.py   # RAG 문서 로드 및 Vector DB 구축
├── graph/               # LangGraph 상태(State) 정의 및 그래프 빌더
├── prompts/             # 에이전트별 행동 규칙 (System Prompt)
├── scripts/             # 유틸리티 스크립트 (build_vectorstore.py)
├── tools/               # 7개의 Banking Tools 정의
├── app.py               # Streamlit 웹 기반 사용자 인터페이스
├── .env                 # 환경 변수 및 설정 (DB, API Key 등)
└── requirements.txt     # 파이썬 의존성 패키지
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

## ✅ 구현 단계 (Roadmap)
- [x] Phase 1: 기본 챗봇 기반 마련 및 환경 설정
- [x] Phase 2: LangGraph 멀티 에이전트 아키텍처(Supervisor 패턴) 도입
- [x] Phase 3: 도구(Tool) 기반 응답 체계 구축
- [x] Phase 4: PostgreSQL NL2SQL 및 하이브리드 RAG 연동
- [x] Phase 5: 고급 금융 분석 기능(중도해지, 갈아타기, 이자계산) 구현 및 Streamlit UI 통합
