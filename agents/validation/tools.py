"""
Validation 에이전트 전용 도구
LLM 기반 결과 검증
"""
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from config.settings import get_llm


VERIFY_PROMPT = """당신은 금융 데이터 검증 전문가입니다.
아래 결과를 검증하고 오류가 있으면 지적해주세요.

## 검증 항목
1. **계산 오류**: 이자, 만기 수령액, 세금 등의 수치가 정확한지
2. **논리 오류**: 비교 분석의 논리가 타당한지
3. **정보 오류**: 상품 정보, 고객 정보가 일관성 있는지
4. **추천 오류**: 추천 상품이 고객 조건에 부합하는지
5. **할루시네이션(Hallucination)**: DB 데이터나 RAG 약관 정보에 근거하지 않는 수치, 우대금리, 가상의 금융 상품을 지어냈는지 여부

## 응답 형식
- 오류가 없으면: "✅ 검증 완료: 오류 없음"
- 오류가 있으면: "⚠️ 검증 결과:" 뒤에 구체적 오류 내용 및 할루시네이션 탐지 내용 기술
"""


@tool
def verify_result(result_text: str, verification_type: str = "일반") -> str:
    """다른 에이전트가 생성한 결과를 검증합니다.
    계산 오류, 논리 오류, 추천 오류 등을 탐지합니다.

    Args:
        result_text: 검증할 결과 텍스트
        verification_type: 검증 유형 ("계산", "추천", "비교", "일반")
    """
    llm = get_llm()

    messages = [
        SystemMessage(content=VERIFY_PROMPT),
        HumanMessage(content=f"[검증 유형: {verification_type}]\n\n{result_text}"),
    ]

    response = llm.invoke(messages)
    return response.content


# 이 에이전트에 바인딩될 도구 목록
VALIDATION_TOOLS = [verify_result]
