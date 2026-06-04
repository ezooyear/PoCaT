"""
Supervisor Final Agent
- 각 Agent가 State에 저장한 결과를 종합해 최종 고객 상담 답변 생성
- customer_result, calculation_result, product_result, recommendation_result, validation_result를 사용
"""

import json
from langchain_core.messages import SystemMessage, AIMessage

from config.settings import get_llm
from graph.state import AgentState


SUPERVISOR_FINAL_PROMPT = """
당신은 예적금 상담 시스템의 최종 상담 응답 생성 Agent입니다.

## 역할
앞선 Agent들이 State에 저장한 결과를 종합하여 사용자에게 자연스러운 한국어 상담 답변을 작성합니다.

## 사용 가능한 결과
- customer_result: 고객 기본 정보, 가입 계좌, 납입 이력 조회 결과
- calculation_result: 만기, 납입 횟수, 잔액, 예상 이자 등 계산 결과
- product_result: 상품 DB/RAG 조회 결과
- recommendation_result: 추천 판단 결과
- validation_result: 추천 검증 결과

## 응답 원칙
- 내부 용어(State, Agent, JSON, Tool, customer_result 등)는 사용자에게 말하지 마세요.
- 고객 ID가 있으면 고객 기준으로 답변하세요.
- 고객 조회 결과가 있으면 가입 상품, 잔액, 납입 현황을 요약하세요.
- 계산 결과가 있으면 만기, 납입 횟수, 잔액, 예상 이자 등을 설명하세요.
- 추천 결과가 있으면 추천 상품과 이유를 설명하세요.
- 검증 결과가 있으면 검증 통과/주의사항을 자연스럽게 반영하세요.
- 정보가 부족하면 부족한 정보를 솔직하게 말하세요.
- 단정적인 투자/금융 조언처럼 말하지 말고 상담 보조 답변처럼 말하세요.
- 답변은 너무 길지 않게 작성하되, 사용자가 이해하기 쉽게 항목별로 정리하세요.
"""


def supervisor_final_node(state: AgentState) -> dict:
    """
    Supervisor Final 노드

    역할:
    - State에 저장된 Agent 결과들을 읽는다.
    - 최종 고객 상담 답변을 생성한다.
    - final_answer와 messages에 저장한다.
    """

    llm = get_llm()

    user_query = state.get("user_query") or _get_last_user_text(state.get("messages", []))

    final_context = {
        "user_query": user_query,
        "task_type": state.get("task_type"),
        "customer_id": state.get("customer_id") or state.get("member_id"),
        "customer_result": state.get("customer_result"),
        "calculation_result": state.get("calculation_result"),
        "product_result": state.get("product_result"),
        "recommendation_result": state.get("recommendation_result"),
        "validation_result": state.get("validation_result"),
        "errors": state.get("errors"),
    }

    messages = [
        SystemMessage(content=SUPERVISOR_FINAL_PROMPT),
        SystemMessage(
            content=(
                "아래는 최종 답변 생성을 위한 내부 결과입니다.\n"
                f"{json.dumps(final_context, ensure_ascii=False, default=str)}"
            )
        ),
    ]

    try:
        response = llm.invoke(messages)
        final_answer = response.content

    except Exception as e:
        final_answer = _fallback_final_answer(final_context, error=e)

    final_answer = final_answer.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")

    return {
        "messages": [AIMessage(content=final_answer)],
        "final_answer": final_answer,
        "next": "FINISH",
    }


def _fallback_final_answer(final_context: dict, error: Exception | None = None) -> str:
    """
    LLM 호출 실패 시에도 최소 답변을 반환하기 위한 fallback.
    """

    customer_result = final_context.get("customer_result") or {}
    calculation_result = final_context.get("calculation_result") or {}

    if customer_result.get("ok") and calculation_result.get("ok"):
        customer = (
            customer_result.get("customer_profile")
            or customer_result.get("customer")
            or {}
        )

        customer_id = final_context.get("customer_id")
        customer_name = customer.get("customer_name", "고객")

        summary_lines = [
            f"{customer_name}님 기준으로 조회한 결과입니다.",
            "",
            f"- 고객 ID: {customer_id}",
            f"- 가입 계좌 수: {len(customer_result.get('accounts', []))}개",
            f"- 납입 이력 수: {len(customer_result.get('payment_history', []))}건",
            f"- 총 계좌 잔액: {calculation_result.get('total_balance', 0):,.0f}원",
            f"- 활성 계좌 수: {calculation_result.get('active_account_count', 0)}개",
            f"- 추정 월 저축 여력: {calculation_result.get('available_monthly_saving', 0):,.0f}원",
            "",
            calculation_result.get("calculation_summary", ""),
        ]

        return "\n".join(summary_lines)

    if customer_result.get("ok"):
        customer = (
            customer_result.get("customer_profile")
            or customer_result.get("customer")
            or {}
        )
        customer_name = customer.get("customer_name", "고객")

        return (
            f"{customer_name}님 기준으로 고객 정보를 조회했습니다.\n\n"
            f"- 가입 계좌 수: {len(customer_result.get('accounts', []))}개\n"
            f"- 납입 이력 수: {len(customer_result.get('payment_history', []))}건\n\n"
            f"{customer_result.get('summary', '')}"
        )

    error_text = str(error) if error else "알 수 없는 오류"
    return (
        "죄송합니다. 조회된 정보를 바탕으로 최종 답변을 생성하는 중 문제가 발생했습니다.\n"
        f"오류 내용: {error_text}"
    )


def _get_last_user_text(messages: list) -> str:
    """
    messages에서 마지막 사용자 질문을 추출합니다.
    tuple 형태와 LangChain Message 형태를 모두 처리합니다.
    """

    for msg in reversed(messages):
        if isinstance(msg, tuple) and len(msg) >= 2:
            role, content = msg[0], msg[1]
            if role in ["user", "human"]:
                return str(content)

        if hasattr(msg, "type") and hasattr(msg, "content"):
            if msg.type in ["human", "user"]:
                return str(msg.content)

    return ""