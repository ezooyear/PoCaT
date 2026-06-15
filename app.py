"""
Streamlit chat UI for the savings assistant.
"""

from dotenv import load_dotenv
import streamlit as st
from uuid import uuid4

load_dotenv()

from graph.builder import build_graph
from observability.langfuse import (
    flush_langfuse,
    langfuse_observation,
    langfuse_trace_context,
    update_observation,
)


st.set_page_config(
    page_title="KB 예적금 상담 AI",
    page_icon="🏦",
    layout="centered",
)

st.markdown(
    """
<style>
    header { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .stDeployButton { display: none; }
    .main-header { text-align: center; padding: 1rem 0 0.5rem 0; }
    .main-header h1 {
        font-size: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .main-header p { color: #888; font-size: 0.95rem; }
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

st.markdown(
    """
<div class="main-header">
    <h1>🏦 예적금 상담 AI</h1>
    <p>예적금 관련 궁금한 점을 자유롭게 질문해 주세요</p>
</div>
<hr class="divider">
""",
    unsafe_allow_html=True,
)


if "messages" not in st.session_state:
    st.session_state.messages = []

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

if "conversation_messages" not in st.session_state:
    st.session_state.conversation_messages = []

if "langfuse_session_id" not in st.session_state:
    st.session_state.langfuse_session_id = f"streamlit-{uuid4().hex}"


def _build_next_conversation(messages: list, latest_user_prompt: str, latest_ai_content: str) -> list[tuple[str, str]]:
    preserved = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            preserved.append((role, content))

    preserved.append(("user", latest_user_prompt))
    preserved.append(("assistant", latest_ai_content))
    return preserved


for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])


if prompt := st.chat_input("궁금한 점을 입력해 주세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    st.session_state.conversation_messages.append(("user", prompt))

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("답변을 생성하고 있습니다..."):
            try:
                with langfuse_trace_context(
                    trace_name="streamlit-chat-turn",
                    session_id=st.session_state.langfuse_session_id,
                    tags=["streamlit", "pocat"],
                    metadata={
                        "surface": "streamlit",
                        "app": "pocat",
                    },
                ):
                    with langfuse_observation(
                        name="streamlit_chat_turn",
                        as_type="span",
                        input={
                            "prompt": prompt,
                            "conversation_length": len(st.session_state.conversation_messages),
                        },
                        metadata={"surface": "streamlit"},
                    ) as observation:
                        result = st.session_state.graph.invoke(
                            {
                                "messages": st.session_state.conversation_messages,
                                "next": "",
                                "member_id": None,
                                "context": None,
                                "plan": [],
                                "current_step": 0,
                                "agent_outputs": {},
                            }
                        )

                        ai_message = result["messages"][-1]
                        ai_content = ai_message.content if hasattr(ai_message, "content") else str(ai_message)
                        ai_content = ai_content.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")

                        update_observation(
                            observation,
                            output={
                                "response_preview": ai_content[:500],
                                "message_count": len(result.get("messages") or []),
                                "current_agent": result.get("current_agent"),
                                "completed_agents": result.get("completed_agents") or [],
                            },
                        )

                st.markdown(ai_content)
                st.session_state.messages.append({"role": "assistant", "content": ai_content})
                st.session_state.conversation_messages = _build_next_conversation(
                    st.session_state.messages[:-2],
                    prompt,
                    ai_content,
                )
                flush_langfuse()
            except Exception as error:
                error_msg = f"오류가 발생했습니다: {error}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                flush_langfuse()
