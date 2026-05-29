"""
예적금 상담 에이전트 - Streamlit UI
- 기본 챗봇 UI (대화 히스토리 유지)
"""
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from graph.builder import build_graph
from db.postgres_db import test_connection
from db.vectorstore import get_vectorstore


# ─── 페이지 설정 ───
st.set_page_config(
    page_title="🏦 예적금 상담 AI",
    page_icon="🏦",
    layout="centered",
)

# ─── 커스텀 스타일 ───
st.markdown("""
<style>

    /* Streamlit 기본 UI 숨기기 */
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

    /* 메인 헤더 스타일 */
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
    /* 구분선 */
    .divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #667eea55, transparent);
        margin: 0.5rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ─── 헤더 ───
st.markdown("""
<div class="main-header">
    <h1>🏦 예적금 상담 AI</h1>
    <p>예적금 관련 궁금한 점을 자유롭게 질문해주세요</p>
</div>
<hr class="divider">
""", unsafe_allow_html=True)

# ─── 세션 상태 초기화 ───
if "messages" not in st.session_state:
    st.session_state.messages = []

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

if "conversation_messages" not in st.session_state:
    st.session_state.conversation_messages = []

# ─── 기존 대화 히스토리 표시 ───
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])

# ─── 사용자 입력 처리 ───
if prompt := st.chat_input("궁금한 점을 입력하세요..."):
    # 사용자 메시지 표시 & 저장
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # 대화 히스토리에 추가
    st.session_state.conversation_messages.append(("user", prompt))

    # AI 응답 생성
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("답변을 생성하고 있습니다..."):
            try:
                result = st.session_state.graph.invoke({
                    "messages": st.session_state.conversation_messages,
                    "next": "",
                    "member_id": None,
                    "context": None,
                })

                # AI 응답 추출
                ai_message = result["messages"][-1]
                ai_content = ai_message.content if hasattr(ai_message, "content") else str(ai_message)

                st.markdown(ai_content)

                # 대화 히스토리 갱신
                st.session_state.conversation_messages = [
                    (msg.type if hasattr(msg, "type") else "user",
                     msg.content if hasattr(msg, "content") else str(msg))
                    for msg in result["messages"]
                ]

                # 화면 표시용 메시지 저장
                st.session_state.messages.append({"role": "assistant", "content": ai_content})

            except Exception as e:
                error_msg = f"❌ 오류가 발생했습니다: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
