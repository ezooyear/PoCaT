"""
예적금 상담 에이전트 - Streamlit UI
- 고객 ID 선택 UI
- 기본 챗봇 UI
- 선택된 customer_id/member_id를 LangGraph State에 전달
"""

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from graph.builder import build_graph
from db.postgres_db import test_connection


# ─── 페이지 설정 ───
st.set_page_config(
    page_title="🏦 예적금 상담 AI",
    page_icon="🏦",
    layout="centered",
)


# ─── 그래프 캐싱 ───
@st.cache_resource
def get_graph():
    """
    LangGraph 그래프를 한 번만 생성해서 재사용합니다.
    Streamlit rerun 때마다 그래프를 다시 만들지 않기 위한 캐시입니다.
    """
    return build_graph()


# ─── 커스텀 스타일 ───
st.markdown(
    """
<style>
    header {
        visibility: hidden;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    .stDeployButton {
        display: none;
    }

    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }

    .main-header h1 {
        font-size: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }

    .main-header p {
        color: #888;
        font-size: 0.95rem;
    }

    .divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #667eea55, transparent);
        margin: 0.5rem 0 1rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ─── 세션 상태 초기화 ───
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_messages" not in st.session_state:
    st.session_state.conversation_messages = []

# 특정 고객 ID를 기본값으로 넣지 않음
if "selected_customer_id" not in st.session_state:
    st.session_state.selected_customer_id = None


# ─── 사이드바 ───
with st.sidebar:
    st.header("고객 선택")

    customer_id_input = st.text_input(
        "고객 ID를 입력하세요",
        value="" if st.session_state.selected_customer_id is None else str(st.session_state.selected_customer_id),
        placeholder="예: 101",
    )

    if customer_id_input.strip():
        try:
            st.session_state.selected_customer_id = int(customer_id_input.strip())
            st.success(f"현재 선택 고객 ID: {st.session_state.selected_customer_id}")
        except ValueError:
            st.session_state.selected_customer_id = None
            st.error("고객 ID는 숫자로 입력해주세요.")
    else:
        st.session_state.selected_customer_id = None
        st.info("현재 선택된 고객이 없습니다.")

    st.divider()

    st.subheader("시스템 상태")

    try:
        db_ok = test_connection()
        st.write("PostgreSQL:", "✅ 연결됨" if db_ok else "❌ 연결 실패")
    except Exception as e:
        st.write("PostgreSQL: ❌ 확인 실패")
        st.caption(str(e))

    st.write("RAG/ChromaDB:", "⏸️ 시작 속도 문제로 상태 확인 임시 비활성화")

    st.divider()

    if st.button("대화 초기화"):
        st.session_state.messages = []
        st.session_state.conversation_messages = []
        st.rerun()


# ─── 헤더 ───
st.markdown(
    """
<div class="main-header">
    <h1>🏦 예적금 상담 AI</h1>
    <p>고객 DB와 예적금 상품 정보를 바탕으로 맞춤 상담을 제공합니다</p>
</div>
<hr class="divider">
""",
    unsafe_allow_html=True,
)


# ─── 현재 고객 안내 ───
if st.session_state.selected_customer_id is not None:
    st.caption(f"현재 선택된 고객 ID: {st.session_state.selected_customer_id}")
else:
    st.caption("고객 맞춤 상담이 필요한 경우 사이드바에서 고객 ID를 먼저 입력해주세요.")


# ─── 기존 대화 히스토리 표시 ───
for msg in st.session_state.messages:
    with st.chat_message(
        msg["role"],
        avatar="👤" if msg["role"] == "user" else "🤖",
    ):
        st.markdown(msg["content"])


# ─── 사용자 입력 처리 ───
if prompt := st.chat_input("궁금한 점을 입력하세요..."):
    # 사용자 메시지 저장 및 표시
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
    })

    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # LangGraph에 전달할 대화 히스토리
    st.session_state.conversation_messages.append(("user", prompt))

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("답변을 생성하고 있습니다..."):
            try:
                graph = get_graph()

                result = graph.invoke({
                    "messages": st.session_state.conversation_messages,
                    "next": "",
                    "member_id": st.session_state.selected_customer_id,
                    "customer_id": st.session_state.selected_customer_id,
                    "user_query": prompt,
                    "task_type": None,
                    "plan": [],
                    "current_step": 0,
                    "context": {},
                    "customer_result": None,
                    "calculation_result": None,
                    "product_result": None,
                    "recommendation_result": None,
                    "validation_result": None,
                    "final_answer": None,
                    "agent_logs": [],
                    "errors": [],
                })

                # 최종 답변 우선 사용
                if result.get("final_answer"):
                    ai_content = result["final_answer"]
                else:
                    ai_message = result["messages"][-1]
                    ai_content = (
                        ai_message.content
                        if hasattr(ai_message, "content")
                        else str(ai_message)
                    )

                ai_content = (
                    ai_content
                    .replace("<br>", "\n")
                    .replace("<br/>", "\n")
                    .replace("<br />", "\n")
                )

                st.markdown(ai_content)

                # conversation_messages 갱신
                if "messages" in result:
                    new_conversation_messages = []

                    for msg in result["messages"]:
                        if isinstance(msg, tuple) and len(msg) >= 2:
                            new_conversation_messages.append((msg[0], msg[1]))

                        elif hasattr(msg, "type") and hasattr(msg, "content"):
                            role = "user" if msg.type in ["human", "user"] else "assistant"
                            new_conversation_messages.append((role, msg.content))

                        else:
                            new_conversation_messages.append(("assistant", str(msg)))

                    st.session_state.conversation_messages = new_conversation_messages

                # 화면 표시용 메시지 저장
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": ai_content,
                })

            except Exception as e:
                error_msg = f"❌ 오류가 발생했습니다: {e}"
                st.error(error_msg)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })