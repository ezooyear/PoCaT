"""
Supervisor Final Node
- 각 Agent의 결과를 종합하여 최종 답변 생성
"""
from langchain_core.messages import SystemMessage, AIMessage

from config.settings import get_llm
from graph.state import AgentState


FINAL_SYSTEM_PROMPT = """
당신은 예적금 상담 에이전트 시스템의 최종 응답을 작성하는 Supervisor입니다.

역할:
- 각 Agent가 생성한 결과를 종합합니다.
- 고객에게 자연스럽고 이해하기 쉬운 상담 답변을 제공합니다.
- DB 조회 결과, 계산 결과, 상품 약관/RAG 근거, 추천 결과, 검증 결과를 구분해 반영합니다.
- 검증 결과에 문제가 있으면 단정적으로 추천하지 말고 주의사항을 함께 안내합니다.

답변 원칙:
- 고객 상황을 먼저 요약합니다.
- 추천 또는 분석 결론을 명확히 말합니다.
- 근거를 2~3개로 정리합니다.
- 약관/RAG 기반 주의사항이 있으면 마지막에 안내합니다.
- 모르는 정보는 추측하지 말고 추가 확인이 필요하다고 말합니다.
"""


def supervisor_final_node(state: AgentState) -> dict:
    """
    Agent 결과를 최종 답변으로 종합한다.
    """
    llm = get_llm()

    customer_result = state.get("customer_result")
    calculation_result = state.get("calculation_result")
    product_result = state.get("product_result")
    recommendation_result = state.get("recommendation_result")
    validation_result = state.get("validation_result")

    summary_context = f"""
## Customer Agent 결과
{customer_result}

## Calculation Agent 결과
{calculation_result}

## Product Agent 결과
{product_result}

## Recommend Agent 결과
{recommendation_result}

## Validation Agent 결과
{validation_result}
"""

    messages = [
        SystemMessage(content=FINAL_SYSTEM_PROMPT),
        SystemMessage(content=summary_context),
    ] + list(state.get("messages", []))

    response = llm.invoke(messages)

    return {
        "final_answer": response.content,
        "messages": [AIMessage(content=response.content)],
    }