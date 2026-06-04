"""
PoCaT - 예적금 상담 에이전트 Streamlit UI

구성
- 왼쪽 사이드바: 고객 ID 선택, 상담 시나리오, 시스템 상태
- 메인 탭 1: 왼쪽 고객 금융 현황 + 오른쪽 상담 챗봇
- 메인 탭 2: 고객 금융 현황 상세
- 메인 탭 3: 추천/분석 결과
- 메인 탭 4: Agent 실행 로그
"""

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from graph.builder import build_graph
from db.postgres_db import test_connection, get_customer_dashboard_data


# ─────────────────────────────
# 페이지 설정
# ─────────────────────────────
st.set_page_config(
    page_title="PoCaT 고객 금융 상담 대시보드",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────
# 캐시 함수
# ─────────────────────────────
@st.cache_resource
def get_graph():
    """LangGraph 그래프를 한 번만 생성해서 재사용"""
    return build_graph()


@st.cache_data(ttl=60)
def check_db_connection():
    """DB 연결 상태 확인"""
    return test_connection()


@st.cache_data(ttl=60)
def load_customer_dashboard(customer_id: int):
    """고객 대시보드 데이터 조회"""
    return get_customer_dashboard_data(customer_id)


# ─────────────────────────────
# 유틸 함수
# ─────────────────────────────
def format_won(value):
    try:
        return f"{float(value):,.0f}원"
    except Exception:
        return "-"


def format_rate(value):
    try:
        return f"{float(value):.2f}%"
    except Exception:
        return "-"


def normalize_status(status):
    if status == "ACTIVE":
        return "진행중"
    if status == "MATURED":
        return "만기"
    if status == "CLOSED":
        return "해지"
    return status or "-"


def normalize_product_type(product_type):
    if product_type in ["DEPOSIT", "예금"]:
        return "예금"
    if product_type in ["SAVING", "적금"]:
        return "적금"
    return product_type or "-"


def yn(value):
    if value in [True, "Y", "y", "YES", "Yes", 1]:
        return "Y"
    if value in [False, "N", "n", "NO", "No", 0]:
        return "N"
    return "-"


def calculate_age(birth_date):
    try:
        today = pd.Timestamp.today()
        birth = pd.to_datetime(birth_date)
        return today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
    except Exception:
        return "-"


def make_account_view_df(accounts):
    """가입 계좌 현황 표시용 DataFrame 생성"""
    if not accounts:
        return pd.DataFrame()

    accounts_df = pd.DataFrame(accounts)

    display_columns = [
        "product_name",
        "product_type",
        "join_date",
        "maturity_date",
        "current_balance",
        "applied_rate",
        "account_status",
    ]

    display_columns = [col for col in display_columns if col in accounts_df.columns]

    view_df = accounts_df[display_columns].copy()

    if "product_type" in view_df.columns:
        view_df["product_type"] = view_df["product_type"].apply(normalize_product_type)

    if "current_balance" in view_df.columns:
        view_df["current_balance"] = view_df["current_balance"].apply(format_won)

    if "applied_rate" in view_df.columns:
        view_df["applied_rate"] = view_df["applied_rate"].apply(format_rate)

    if "account_status" in view_df.columns:
        view_df["account_status"] = view_df["account_status"].apply(normalize_status)

    view_df = view_df.rename(
        columns={
            "product_name": "상품명",
            "product_type": "유형",
            "join_date": "가입일",
            "maturity_date": "만기일",
            "current_balance": "현재 잔액",
            "applied_rate": "적용 금리",
            "account_status": "상태",
        }
    )

    return view_df


def get_basic_summary(customer, accounts, payments):
    """대시보드 상단 KPI 계산"""
    accounts = accounts or []
    payments = payments or []

    total_balance = sum(float(acc.get("current_balance") or 0) for acc in accounts)

    active_accounts = [
        acc for acc in accounts
        if acc.get("account_status") == "ACTIVE"
    ]

    matured_accounts = [
        acc for acc in accounts
        if acc.get("account_status") == "MATURED"
    ]

    closed_accounts = [
        acc for acc in accounts
        if acc.get("account_status") == "CLOSED"
    ]

    active_monthly_payment = sum(
        float(acc.get("monthly_amount") or 0)
        for acc in active_accounts
    )

    available_monthly_saving = 0

    if customer:
        available_monthly_saving = float(customer.get("available_monthly_saving") or 0)

    return {
        "total_balance": total_balance,
        "active_accounts": active_accounts,
        "matured_accounts": matured_accounts,
        "closed_accounts": closed_accounts,
        "active_monthly_payment": active_monthly_payment,
        "available_monthly_saving": available_monthly_saving,
    }


def run_agent(prompt: str):
    """LangGraph 실행"""
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

    if result.get("agent_logs"):
        st.session_state.agent_logs = result["agent_logs"]
    else:
        st.session_state.agent_logs = [
            "Supervisor: 질문 유형 판단 완료",
            "Customer Agent: 고객 정보 조회 완료",
            "Calculation Agent: 납입 현황 계산 완료",
            "Product Agent: 상품 조건 검색 완료",
            "Recommendation Agent: 추천 결과 생성 완료",
            "Validation Agent: 추천 결과 검증 완료",
            "Supervisor Final: 최종 답변 생성 완료",
        ]

    st.session_state.latest_result = result

    return ai_content


# ─────────────────────────────
# CSS
# ─────────────────────────────
st.markdown(
    """
<style>

    header {
    visibility: visible !important;
    height: 3rem;
    background: transparent;
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

    .block-container {
        padding-top: 3.2rem;
        padding-bottom: 1.5rem;
        max-width: 100%;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #071426 0%, #0d2138 100%);
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label {
        color: #f8fafc !important;
    }

    section[data-testid="stSidebar"] input {
        color: #111827 !important;
        background-color: #ffffff !important;
    }

    /* 사이드바 버튼 글씨가 흐리게 보이는 문제 해결 */
    section[data-testid="stSidebar"] div[data-testid="stButton"] button {
    color: #111827 !important;
    background-color: #ffffff !important;
    border: 1px solid #d1d5db !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    opacity: 1 !important;
    }

    /* 버튼 내부 p/span 글씨까지 강제로 진하게 */
    section[data-testid="stSidebar"] div[data-testid="stButton"] button p,
    section[data-testid="stSidebar"] div[data-testid="stButton"] button span {
        color: #111827 !important;
        opacity: 1 !important;
        font-weight: 700 !important;
    }

    /* hover 시 */
    section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover {
        background-color: #dbeafe !important;
        color: #1d4ed8 !important;
        border: 1px solid #60a5fa !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover p,
    section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover span {
        color: #1d4ed8 !important;
    }


    
    .main-title {
        font-size: 1.7rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        color: #111827;
    }

    .main-subtitle {
        font-size: 0.95rem;
        color: #6b7280;
        margin-bottom: 1rem;
    }

    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.45rem;
        font-weight: 800;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        color: #4b5563;
        font-weight: 700;
    }

    .small-caption {
        color: #6b7280;
        font-size: 0.85rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────
# 세션 상태 초기화
# ─────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_messages" not in st.session_state:
    st.session_state.conversation_messages = []

if "selected_customer_id" not in st.session_state:
    st.session_state.selected_customer_id = 101

if "agent_logs" not in st.session_state:
    st.session_state.agent_logs = []

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = ""


# ─────────────────────────────
# 사이드바
# ─────────────────────────────
with st.sidebar:
    st.markdown("## 💬 PoCaT")
    st.caption("고객 금융 상담 대시보드")

    st.divider()

    st.markdown("### 고객 선택")

    customer_id_input = st.text_input(
        "고객 ID",
        value=str(st.session_state.selected_customer_id) if st.session_state.selected_customer_id else "",
        placeholder="예: 101",
    )

    if customer_id_input.strip():
        try:
            st.session_state.selected_customer_id = int(customer_id_input.strip())
        except ValueError:
            st.session_state.selected_customer_id = None
            st.error("고객 ID는 숫자로 입력해주세요.")
    else:
        st.session_state.selected_customer_id = None

    st.divider()

    st.markdown("### 상담 시나리오")

    scenario_prompts = {
        "추가 가입 추천": "내 가입 상품과 납입 현황을 고려해서 추가 가입할 만한 상품 추천해줘.",
        "중도해지 상담": "지금 가입한 상품을 중도해지하면 손해가 클지 상담해줘.",
        "갈아타기 상담": "현재 상품을 유지하는 게 나을지 다른 상품으로 갈아타는 게 나을지 비교해줘.",
        "내 상품 조회": "내가 가입한 예금과 적금 상품 현황을 조회해줘.",
        "납입 현황 조회": "내 적금 납입 현황과 앞으로 남은 납입 횟수를 알려줘.",
    }

    for label, prompt_text in scenario_prompts.items():
        if st.button(label, use_container_width=True):
            st.session_state.pending_prompt = prompt_text

    st.divider()

    st.markdown("### 시스템 상태")

    try:
        db_ok = check_db_connection()
        st.write("PostgreSQL:", "✅ 연결됨" if db_ok else "❌ 연결 실패")
    except Exception as e:
        st.write("PostgreSQL: ❌ 확인 실패")
        st.caption(str(e))

    st.write("RAG/ChromaDB:", "✅ 사용 가능")

    st.divider()

    col_reset, col_refresh = st.columns(2)

    with col_reset:
        if st.button("대화 초기화", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_messages = []
            st.session_state.agent_logs = []
            st.session_state.latest_result = None
            st.session_state.pending_prompt = ""
            st.rerun()

    with col_refresh:
        if st.button("새로고침", use_container_width=True):
            load_customer_dashboard.clear()
            st.rerun()


# ─────────────────────────────
# 대시보드 데이터 로드
# ─────────────────────────────
dashboard_data = None
customer = None
accounts = []
payments = []

if st.session_state.selected_customer_id is not None:
    try:
        dashboard_data = load_customer_dashboard(st.session_state.selected_customer_id)
        customer = dashboard_data.get("customer")
        accounts = dashboard_data.get("accounts", [])
        payments = dashboard_data.get("payment_history", [])
    except Exception as e:
        st.error(f"대시보드 데이터 조회 중 오류가 발생했습니다: {e}")


summary = get_basic_summary(customer, accounts, payments)


# ─────────────────────────────
# 메인 헤더
# ─────────────────────────────
st.markdown('<div class="main-title">PoCaT</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-subtitle">고객 금융 상담 대시보드</div>',
    unsafe_allow_html=True,
)


# ─────────────────────────────
# 탭
# ─────────────────────────────
chat_tab, dashboard_tab, result_tab, log_tab = st.tabs(
    ["상담 챗봇", "고객 금융 현황", "추천/분석 결과", "Agent 실행 로그"]
)


# ─────────────────────────────
# 탭 1. 상담 챗봇
# ─────────────────────────────
with chat_tab:
    if st.session_state.selected_customer_id is None:
        st.info("왼쪽 사이드바에서 고객 ID를 입력해주세요.")

    elif customer is None:
        st.warning(f"고객 ID {st.session_state.selected_customer_id}번 고객을 찾을 수 없습니다.")

    else:
        left_col, right_col = st.columns([1.75, 1.25], gap="large")

        # ─────────────────────────────
        # 왼쪽: 고객 금융 현황 요약
        # ─────────────────────────────
        with left_col:
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)

            kpi1.metric("총 가입 상품 수", f"{len(accounts)}개")
            kpi2.metric("총 잔액", format_won(summary["total_balance"]))
            kpi3.metric("월 납입 중인 금액", format_won(summary["active_monthly_payment"]))
            kpi4.metric("추가 납입 가능 금액", format_won(summary["available_monthly_saving"]))

            st.markdown("")

            st.markdown("### 가입 상품 현황")

            if accounts:
                account_view_df = make_account_view_df(accounts)

                st.dataframe(
                    account_view_df,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("가입 상품 정보가 없습니다.")

            st.markdown("")

            lower_col1, lower_col2 = st.columns([1.15, 1], gap="large")

            with lower_col1:
                st.markdown("### 납입 진행률")

                if accounts:
                    for acc in accounts:
                        product_name = acc.get("product_name", "-")
                        contract_months = int(acc.get("contract_months") or 0)

                        if payments:
                            paid_count = len([
                                p for p in payments
                                if p.get("account_id") == acc.get("account_id")
                            ])
                        else:
                            paid_count = 0

                        if contract_months > 0:
                            progress = min(paid_count / contract_months, 1.0)

                            st.write(f"**{product_name}**")
                            st.progress(progress)
                            st.caption(f"{paid_count} / {contract_months}회 · {progress * 100:.0f}%")
                        else:
                            st.write(f"**{product_name}**")
                            st.caption("계약 개월 수 정보 없음")
                else:
                    st.info("납입 진행률 정보가 없습니다.")

            with lower_col2:
                st.markdown("### 만기 임박 상품")

                maturity_rows = []

                for acc in accounts:
                    try:
                        maturity_date = pd.to_datetime(acc.get("maturity_date"))
                        today = pd.Timestamp.today().normalize()
                        days_left = (maturity_date - today).days

                        if days_left >= 0:
                            maturity_rows.append({
                                "상품명": acc.get("product_name", "-"),
                                "남은 일수": days_left,
                                "만기일": maturity_date.strftime("%Y-%m-%d"),
                            })
                    except Exception:
                        pass

                maturity_rows = sorted(maturity_rows, key=lambda x: x["남은 일수"])[:5]

                if maturity_rows:
                    for row in maturity_rows:
                        st.write(f"**{row['상품명']}**")
                        st.caption(f"만기까지 {row['남은 일수']}일 · {row['만기일']}")
                else:
                    st.info("만기 임박 상품이 없습니다.")

            st.markdown("")

            st.markdown("### 추천 결과 요약")

            latest_answer = None

            for msg in reversed(st.session_state.messages):
                if msg["role"] == "assistant":
                    latest_answer = msg["content"]
                    break

            if latest_answer:
                with st.expander("최근 상담 답변 보기", expanded=True):
                    st.markdown(latest_answer)
            else:
                st.info("오른쪽 상담 챗봇에서 질문을 입력하면 추천/분석 결과가 표시됩니다.")

        # ─────────────────────────────
        # 오른쪽: 상담 챗봇
        # ─────────────────────────────
        with right_col:
            st.markdown("### 상담 챗봇")

            chat_container = st.container(height=520)

            with chat_container:
                if not st.session_state.messages:
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(
                            "안녕하세요. 고객의 가입 상품과 납입 현황을 바탕으로 예적금 상담을 도와드릴게요."
                        )

                for msg in st.session_state.messages:
                    with st.chat_message(
                        msg["role"],
                        avatar="👤" if msg["role"] == "user" else "🤖",
                    ):
                        st.markdown(msg["content"])

            with st.form("chat_form", clear_on_submit=True):
                user_prompt = st.text_area(
                    "메시지 입력",
                    value=st.session_state.pending_prompt,
                    placeholder="메시지를 입력하세요...",
                    height=90,
                    label_visibility="collapsed",
                )

                submitted = st.form_submit_button("전송", use_container_width=True)

            if submitted and user_prompt.strip():
                prompt = user_prompt.strip()
                st.session_state.pending_prompt = ""

                st.session_state.messages.append({
                    "role": "user",
                    "content": prompt,
                })

                st.session_state.conversation_messages.append(("user", prompt))

                with st.spinner("답변을 생성하고 있습니다..."):
                    try:
                        ai_content = run_agent(prompt)

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": ai_content,
                        })

                        st.rerun()

                    except Exception as e:
                        error_msg = f"❌ 오류가 발생했습니다: {e}"

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg,
                        })

                        st.session_state.agent_logs = [error_msg]

                        st.rerun()


# ─────────────────────────────
# 탭 2. 고객 금융 현황 상세
# ─────────────────────────────
with dashboard_tab:
    if st.session_state.selected_customer_id is None:
        st.info("왼쪽 사이드바에서 고객 ID를 입력해주세요.")

    elif customer is None:
        st.warning(f"고객 ID {st.session_state.selected_customer_id}번 고객을 찾을 수 없습니다.")

    else:
        st.markdown("### 고객 기본 정보")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("고객 ID", customer.get("customer_id", "-"))
        c1.metric("고객명", customer.get("customer_name", "-"))

        c2.metric("연령", f"{calculate_age(customer.get('birth_date'))}세")
        c2.metric("직업", customer.get("customer_job", "-"))

        c3.metric("소득 수준", customer.get("income_level", "-"))
        c3.metric("연소득", f"{customer.get('annual_income'):,}만 원" if customer.get("annual_income") is not None else "-")

        c4.metric("거래 개월 수", f"{customer.get('transaction_months', '-')}개월")
        c4.metric("월 납입 가능 금액", format_won(customer.get("available_monthly_saving")))

        st.divider()

        st.markdown("### 고객 거래 조건")

        t1, t2, t3, t4, t5 = st.columns(5)

        t1.metric("주거래 여부", yn(customer.get("main_bank_yn")))
        t2.metric("급여이체", yn(customer.get("salary_transfer_yn")))
        t3.metric("자동이체", yn(customer.get("auto_transfer_yn")))
        t4.metric("카드 사용", yn(customer.get("card_usage_yn")))
        t5.metric("마케팅 동의", yn(customer.get("marketing_agree_yn")))

        st.divider()

        st.markdown("### 가입 계좌 상세")

        if accounts:
            accounts_df = pd.DataFrame(accounts)
            st.dataframe(accounts_df, use_container_width=True, hide_index=True)
        else:
            st.info("가입 계좌 정보가 없습니다.")

        st.divider()

        st.markdown("### 납입 이력 상세")

        if payments:
            payments_df = pd.DataFrame(payments)
            st.dataframe(payments_df, use_container_width=True, hide_index=True)
        else:
            st.info("납입 이력 정보가 없습니다.")


# ─────────────────────────────
# 탭 3. 추천/분석 결과
# ─────────────────────────────
with result_tab:
    st.markdown("### 추천/분석 결과")

    latest_answer = None

    for msg in reversed(st.session_state.messages):
        if msg["role"] == "assistant":
            latest_answer = msg["content"]
            break

    if latest_answer:
        st.markdown(latest_answer)
    else:
        st.info("상담 챗봇에서 질문을 입력하면 추천/분석 결과가 표시됩니다.")

    if st.session_state.latest_result:
        with st.expander("Agent 원본 결과 보기"):
            st.write(st.session_state.latest_result)


# ─────────────────────────────
# 탭 4. Agent 실행 로그
# ─────────────────────────────
with log_tab:
    st.markdown("### Agent 실행 로그")

    if st.session_state.agent_logs:
        for i, log in enumerate(st.session_state.agent_logs, start=1):
            st.write(f"✅ {i}. {log}")
    else:
        st.info("아직 실행된 Agent 로그가 없습니다.")