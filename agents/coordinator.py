"""
Coordinator 에이전트
- Phase 1: 기본 챗봇 역할 (직접 응답)
- Phase 2+: 진입점 역할 후 Supervisor에게 위임
- RAG: Vector DB에서 관련 상품 정보를 검색하여 컨텍스트로 활용
"""
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import get_llm
from graph.state import AgentState
from prompts.coordinator_prompt import COORDINATOR_SYSTEM_PROMPT
from db.vectorstore import search_products
from db.customer_db import get_customer, format_customer_info


def coordinator_node(state: AgentState) -> dict:
    """
    Coordinator 노드 함수
    - 사용자 질문으로 Vector DB에서 관련 상품 정보 검색 (RAG)
    - 고객 정보가 있으면 컨텍스트에 추가
    - 시스템 프롬프트와 함께 LLM에 전달하여 응답 생성
    """
    llm = get_llm()

    # 1. 사용자 최신 메시지 추출
    last_message = state["messages"][-1]
    user_message = last_message.content if hasattr(last_message, "content") else str(last_message)

    # 2. Vector DB에서 관련 상품 정보 검색 (RAG)
    relevant_docs = search_products(user_message, k=3)
    if relevant_docs:
        context = "\n\n---\n\n".join([
            f"[출처: {doc.metadata.get('source_file', '알 수 없음')} / p.{doc.metadata.get('page', '?')}]\n{doc.page_content}"
            for doc in relevant_docs
        ])
    else:
        context = "검색된 상품 정보가 없습니다. (Vector DB가 구축되지 않았거나 관련 정보가 없습니다)"

    # 3. 고객 정보 조회 (있는 경우)
    customer_info = ""
    member_id = state.get("member_id")
    if member_id:
        customer = get_customer(member_id)
        if customer:
            customer_info = format_customer_info(customer)

    # 4. 시스템 프롬프트에 컨텍스트 주입
    system_prompt = COORDINATOR_SYSTEM_PROMPT

    system_prompt += f"""

## 참고할 상품 정보 (Vector DB 검색 결과)
{context}
"""

    if customer_info:
        system_prompt += f"""
## 현재 상담 중인 고객 정보
{customer_info}
"""
    else:
        system_prompt += """
## 현재 상담 중인 고객 정보
선택된 고객이 없습니다. 일반 상담 모드로 응답합니다.
"""

    # 5. LLM에 전달하여 응답 생성
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(messages)

    return {"messages": [response]}
