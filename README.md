# 예적금 상담 멀티 에이전트

LangGraph + LangManus 스타일의 예적금 상담 AI 에이전트입니다.

## 기술 스택
- **LLM**: OpenRouter (langchain-openrouter)
- **오케스트레이션**: LangGraph
- **아키텍처**: LangManus 스타일 Supervisor 패턴
- **Python**: 3.11+

## 설치 및 실행

```bash
# 가상환경 활성화
source venv/bin/activate

# .env 파일에 API 키 설정
# OPENROUTER_API_KEY=your_key_here

# 실행
python main.py
```

## 프로젝트 구조
```
deposit_agent/
├── config/          # 전역 설정
├── agents/          # 에이전트 모듈
├── graph/           # LangGraph 상태 및 빌더
├── prompts/         # 에이전트별 시스템 프롬프트
├── tools/           # Tool 함수 (Phase 2+)
├── crawler/         # 크롤러 (Phase 2+)
├── db/              # 데이터 모듈 (Phase 2+)
├── data/            # 상품/회원 데이터 (Phase 2+)
├── main.py          # CLI 실행 진입점
└── app.py           # Streamlit UI (Phase 3)
```

## 구현 단계
- [x] Phase 1: 기본 챗봇
- [ ] Phase 2: 기본 기능 (상품 조회, 납입 현황, 가입 가능 여부)
- [ ] Phase 3: 고급 기능 (비교/분석/추천/약관)
