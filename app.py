"""
예적금 상담 에이전트 - Streamlit UI
- 기본 챗봇 UI (대화 히스토리 유지)
"""
import streamlit as st
from dotenv import load_dotenv
import re

load_dotenv()

from graph.builder import build_graph


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
    # 이유는 모르겠는데 채팅에서 고객id를 추출하지 못함 -> 수정코드.
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("답변을 생성하고 있습니다..."):
            try:
                # 로그인 UI가 아직 없으므로, 질문 문장에서 고객 ID를 임시 추출
                customer_id = st.session_state.get("customer_id")

                if customer_id is None:
                                    match = re.search(r"고객\s*(\d+)|(\d+)\s*번\s*고객", prompt)
                                    if match:
                                        customer_id = int(match.group(1) or match.group(2))
                                        st.session_state.customer_id = customer_id

                member_id = st.session_state.get("member_id")
                if member_id is None and customer_id is not None:
                    member_id = str(customer_id)
                    st.session_state.member_id = member_id

                result = st.session_state.graph.invoke({
                    "messages": st.session_state.conversation_messages,
                    "next": "",
                    "member_id": member_id,
                    "customer_id": customer_id,
                    "user_query": prompt,
                    "context": None,
                    "plan": [],
                    "current_step": 0,
                    "current_agent": "",
                    "completed_agents": [],
                    "agent_outputs": {},
                    "product_candidates": [],
                    "eligibility_results": [],
                    "financial_results": [],
                    "recommendation_results": [],
                    "customer_profile": None,
                    "customer_accounts": [],
                })

                # AI 응답 추출
                ai_message = result["messages"][-1]
                ai_content = ai_message.content if hasattr(ai_message, "content") else str(ai_message)

                # ⚠️ 챗봇이 프롬프트를 무시하고 <br> 태그를 출력하는 경우를 대비한 강제 전처리
                ai_content = ai_content.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")

                st.markdown(ai_content)

                # 디버그용 상태 출력(확인용으로 추후 삭제 !!)
                with st.expander("DEBUG: 실행 State 확인"):
                    st.write("plan:", result.get("plan"))
                    st.write("completed_agents:", result.get("completed_agents"))
                    st.write("current_agent:", result.get("current_agent"))
                    st.write("current_step:", result.get("current_step"))
                    st.write("agent_outputs keys:", list((result.get("agent_outputs") or {}).keys()))
                    st.write("validation_result:", result.get("validation_result"))
                    st.write("final_answer:", result.get("final_answer"))
                    st.write("product_candidates:", result.get("product_candidates"))
                    st.write("eligibility_results:", result.get("eligibility_results"))
                    st.write("recommendation_results:", result.get("recommendation_results"))
                    st.write("product_result:", result.get("product_result"))
                    st.write("customer_id:", result.get("customer_id"))
                    st.write("member_id:", result.get("member_id"))
                    st.write("customer_result:", result.get("customer_result"))
                    st.write("customer_profile:", result.get("customer_profile"))
                    st.write("customer_accounts:", result.get("customer_accounts"))
                    st.write("agent_outputs.customer_agent:", (result.get("agent_outputs") or {}).get("customer_agent"))


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