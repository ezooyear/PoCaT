"""
Customer dashboard and chat UI for the savings assistant.
"""

from datetime import date
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
import streamlit as st

load_dotenv()

from db.postgres_db import get_connection
from graph.builder import build_graph
from observability.langfuse import (
    flush_langfuse,
    langfuse_observation,
    langfuse_trace_context,
)


PDF_DIR = Path(__file__).resolve().parent / "data" / "pdfs"


st.set_page_config(
    page_title="KB 고객 상담 지원",
    page_icon="KB",
    layout="wide",
)

st.markdown(
    """
<style>
    header { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .stDeployButton { display: none; }

    .block-container {
        max-width: 1440px;
        padding-top: 1.35rem;
        padding-bottom: 3rem;
    }

    .hero {
        border-bottom: 1px solid #e6e8ef;
        padding: 0.2rem 0 1.1rem 0;
        margin-bottom: 1.1rem;
    }

    .hero-label {
        color: #6b7280;
        font-size: 0.88rem;
        font-weight: 700;
        letter-spacing: 0;
        margin-bottom: 0.25rem;
    }

    .hero h1 {
        color: #151922;
        font-size: 2.1rem;
        line-height: 1.2;
        margin: 0 0 0.35rem 0;
    }

    .hero p {
        color: #5f6673;
        font-size: 1rem;
        margin: 0;
    }

    .section-title {
        color: #222631;
        font-size: 1rem;
        font-weight: 750;
        margin: 0.35rem 0 0.7rem 0;
    }

    .panel {
        border: 1px solid #e2e6ef;
        border-radius: 8px;
        background: #ffffff;
        padding: 1rem;
        margin-bottom: 0.9rem;
    }

    .profile-name {
        color: #171b24;
        font-size: 1.2rem;
        font-weight: 800;
        margin: 0.45rem 0 0.25rem 0;
    }

    .profile-meta {
        color: #69717f;
        font-size: 0.9rem;
        margin-bottom: 0.9rem;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 0.22rem 0.62rem;
        background: #ecfdf3;
        color: #157347;
        font-size: 0.78rem;
        font-weight: 750;
    }

    .metric-row {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 0.65rem;
    }

    .metric {
        border: 1px solid #e8ebf2;
        border-radius: 8px;
        padding: 0.75rem;
        background: #fafbfe;
    }

    .metric-label {
        color: #737b89;
        font-size: 0.78rem;
        margin-bottom: 0.2rem;
    }

    .metric-value {
        color: #181c25;
        font-size: 1.05rem;
        font-weight: 800;
    }

    .insight-band {
        border: 1px solid #e3e7ef;
        border-radius: 8px;
        background: #fbfcff;
        padding: 0.95rem 1rem;
        margin-bottom: 0.9rem;
    }

    .insight-band strong {
        color: #20242e;
    }

    .insight-band p {
        color: #5f6673;
        margin: 0.25rem 0 0 0;
        line-height: 1.55;
    }

    .journey-step {
        display: flex;
        gap: 0.65rem;
        padding: 0.55rem 0;
        border-bottom: 1px solid #eef1f6;
    }

    .journey-step:last-child {
        border-bottom: none;
    }

    .step-number {
        flex: 0 0 auto;
        width: 1.55rem;
        height: 1.55rem;
        border-radius: 999px;
        background: #ffcc00;
        color: #222;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.82rem;
        font-weight: 800;
    }

    .step-copy strong {
        color: #20242e;
        display: block;
        font-size: 0.9rem;
        line-height: 1.35;
    }

    .step-copy span {
        color: #6d7582;
        display: block;
        font-size: 0.82rem;
        line-height: 1.4;
        margin-top: 0.12rem;
    }

    .empty-dashboard {
        border: 1px dashed #cfd6e4;
        border-radius: 8px;
        background: #fbfcff;
        color: #626b79;
        padding: 1.25rem;
        line-height: 1.55;
    }

    .chat-label {
        color: #69717f;
        font-size: 0.78rem;
        font-weight: 750;
        margin: 0.75rem 0 0.25rem 0;
    }

    .chat-bubble {
        border: 1px solid #e3e7ef;
        border-radius: 8px;
        padding: 0.78rem 0.9rem;
        margin-bottom: 0.45rem;
        line-height: 1.55;
    }

    .chat-bubble.user {
        background: #fff8d8;
        border-color: #f3de8a;
    }

    .chat-bubble.assistant {
        background: #fbfcff;
    }

    .chat-scroll-hint {
        color: #7a8391;
        font-size: 0.78rem;
        margin: -0.2rem 0 0.55rem 0;
    }

    .account-table-header {
        border: 1px solid #e2e6ef;
        border-radius: 8px 8px 0 0;
        background: #f6f8fc;
        padding: 0.55rem 0.75rem;
        margin-top: 0.2rem;
        font-size: 0.78rem;
        font-weight: 800;
        color: #596273;
    }

    .account-row {
        border-left: 1px solid #e2e6ef;
        border-right: 1px solid #e2e6ef;
        border-bottom: 1px solid #e2e6ef;
        background: #ffffff;
        padding: 0.42rem 0.75rem;
        font-size: 0.86rem;
        color: #171b24;
    }

    .account-row.selected {
        background: #fff8d8;
        border-color: #ffcc00;
    }

    .account-row:last-of-type {
        border-radius: 0 0 8px 8px;
    }

    .account-cell-muted {
        color: #69717f;
        font-size: 0.82rem;
    }

    .detail-hero {
        border: 1px solid #e2e6ef;
        border-radius: 8px;
        background: #ffffff;
        padding: 1rem;
        margin: 0.7rem 0 0.75rem 0;
    }

    .detail-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.65rem;
        margin-bottom: 0.85rem;
    }

    .detail-card {
        border: 1px solid #e8ebf2;
        border-radius: 8px;
        background: #fafbfe;
        padding: 0.85rem;
        min-height: 5.6rem;
    }

    .detail-card-label {
        color: #69717f;
        font-size: 0.76rem;
        font-weight: 800;
        margin-bottom: 0.35rem;
    }

    .detail-card-value {
        color: #171b24;
        font-size: 1rem;
        font-weight: 850;
        line-height: 1.35;
    }

    .detail-card-note {
        color: #69717f;
        font-size: 0.78rem;
        line-height: 1.4;
        margin-top: 0.35rem;
    }

    .detail-status-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.55rem;
        margin-top: 0.75rem;
    }

    .detail-status {
        border: 1px solid #e8ebf2;
        border-radius: 8px;
        padding: 0.65rem;
        background: #ffffff;
    }

    .recommend-card {
        border: 1px solid #e2e6ef;
        border-radius: 8px;
        background: #ffffff;
        padding: 0.8rem;
        margin-bottom: 0.55rem;
    }

    .recommend-rank {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 1.55rem;
        height: 1.55rem;
        border-radius: 999px;
        background: #ffcc00;
        color: #171b24;
        font-weight: 850;
        font-size: 0.82rem;
        margin-right: 0.45rem;
    }

    .recommend-title {
        color: #171b24;
        font-size: 0.94rem;
        font-weight: 850;
        line-height: 1.35;
    }

    .recommend-meta {
        color: #69717f;
        font-size: 0.8rem;
        line-height: 1.45;
        margin-top: 0.35rem;
    }

    .recommend-reason {
        color: #171b24;
        font-size: 0.82rem;
        line-height: 1.45;
        margin-top: 0.45rem;
    }

    .stButton > button,
    .stFormSubmitButton > button,
    .stDownloadButton > button {
        width: 100%;
        border-radius: 8px;
        border: 1px solid #d8dde8;
        background: #ffffff;
        color: #222631;
        font-weight: 650;
        min-height: 2.55rem;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover,
    .stDownloadButton > button:hover {
        border-color: #ffcc00;
        color: #171b24;
    }

    [data-testid="stChatInput"] {
        border-top: 1px solid #eef1f6;
        padding-top: 0.75rem;
    }

    @media (max-width: 780px) {
        .metric-row {
            grid-template-columns: 1fr;
        }

        .hero h1 {
            font-size: 1.7rem;
        }

        .detail-grid,
        .detail-status-grid {
            grid-template-columns: 1fr;
        }
    }
</style>
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

if "selected_customer" not in st.session_state:
    st.session_state.selected_customer = None

if "customer_lookup_error" not in st.session_state:
    st.session_state.customer_lookup_error = None

if "selected_account_index" not in st.session_state:
    st.session_state.selected_account_index = None

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False


def _theme_values() -> dict[str, str]:
    if st.session_state.dark_mode:
        return {
            "app_bg": "#0f141c",
            "surface": "#171d27",
            "surface_soft": "#111722",
            "surface_muted": "#202838",
            "border": "#2b3546",
            "border_soft": "#263142",
            "text": "#eef2f7",
            "text_soft": "#b8c0cc",
            "text_muted": "#8f9aab",
            "accent": "#ffcc00",
            "user_bubble": "#3a3020",
            "assistant_bubble": "#151b25",
            "success_bg": "#123525",
            "success_text": "#7ee0a2",
            "input_bg": "#111722",
        }

    return {
        "app_bg": "#ffffff",
        "surface": "#ffffff",
        "surface_soft": "#fbfcff",
        "surface_muted": "#fafbfe",
        "border": "#e2e6ef",
        "border_soft": "#eef1f6",
        "text": "#171b24",
        "text_soft": "#5f6673",
        "text_muted": "#69717f",
        "accent": "#ffcc00",
        "user_bubble": "#fff8d8",
        "assistant_bubble": "#fbfcff",
        "success_bg": "#ecfdf3",
        "success_text": "#157347",
        "input_bg": "#ffffff",
    }


def _apply_theme() -> None:
    theme = _theme_values()
    st.markdown(
        f"""
<style>
    .stApp {{
        background: {theme["app_bg"]};
        color: {theme["text"]};
    }}

    .block-container {{
        color: {theme["text"]} !important;
    }}

    .stApp,
    .stApp p,
    .stApp span,
    .stApp div,
    .stApp label,
    .stApp li,
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4,
    .stApp h5,
    .stApp h6,
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p {{
        color: {theme["text"]};
    }}

    .hero {{
        border-bottom-color: {theme["border"]};
    }}

    .hero-label,
    .profile-meta,
    .metric-label,
    .step-copy span,
    .chat-label,
    .chat-scroll-hint,
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] * {{
        color: {theme["text_muted"]} !important;
    }}

    .hero h1,
    .section-title,
    .profile-name,
    .metric-value,
    .step-copy strong,
    .insight-band strong {{
        color: {theme["text"]};
    }}

    .hero p,
    .insight-band p {{
        color: {theme["text_soft"]};
    }}

    .panel,
    .insight-band {{
        background: {theme["surface"]} !important;
        border-color: {theme["border"]} !important;
    }}

    .metric {{
        background: {theme["surface_muted"]} !important;
        border-color: {theme["border"]} !important;
    }}

    .empty-dashboard {{
        background: {theme["surface_soft"]};
        border-color: {theme["border"]};
        color: {theme["text_soft"]};
    }}

    .account-table-header {{
        background: {theme["surface_muted"]} !important;
        border-color: {theme["border"]} !important;
        color: {theme["text_muted"]} !important;
    }}

    .account-table-header * {{
        color: {theme["text_muted"]} !important;
    }}

    .account-row {{
        background: {theme["surface"]} !important;
        border-color: {theme["border"]} !important;
        color: {theme["text"]} !important;
    }}

    .account-row.selected {{
        background: {theme["user_bubble"]} !important;
        border-color: {theme["accent"]} !important;
    }}

    .account-row *,
    .account-cell-muted {{
        color: {theme["text"]} !important;
    }}

    .detail-hero {{
        background: {theme["surface"]} !important;
        border-color: {theme["border"]} !important;
    }}

    .detail-card {{
        background: {theme["surface_muted"]} !important;
        border-color: {theme["border"]} !important;
    }}

    .detail-status {{
        background: {theme["surface"]} !important;
        border-color: {theme["border"]} !important;
    }}

    .detail-card-label,
    .detail-card-note {{
        color: {theme["text_muted"]} !important;
    }}

    .detail-card-value,
    .detail-card-value * {{
        color: {theme["text"]} !important;
    }}

    .recommend-card {{
        background: {theme["surface"]} !important;
        border-color: {theme["border"]} !important;
    }}

    .recommend-title,
    .recommend-reason {{
        color: {theme["text"]} !important;
    }}

    .recommend-meta {{
        color: {theme["text_muted"]} !important;
    }}

    .journey-step {{
        border-bottom-color: {theme["border_soft"]};
    }}

    .status-pill {{
        background: {theme["success_bg"]};
        color: {theme["success_text"]};
    }}

    .chat-bubble {{
        border-color: {theme["border"]};
        color: {theme["text"]} !important;
    }}

    .chat-bubble * {{
        color: {theme["text"]} !important;
    }}

    .chat-bubble.user {{
        background: {theme["user_bubble"]};
        border-color: {theme["accent"]};
    }}

    .chat-bubble.assistant {{
        background: {theme["assistant_bubble"]};
    }}

    .stButton > button,
    .stFormSubmitButton > button,
    .stDownloadButton > button {{
        background: {theme["surface"]} !important;
        border-color: {theme["border"]} !important;
        color: {theme["text"]} !important;
    }}

    .stButton > button *,
    .stFormSubmitButton > button *,
    .stDownloadButton > button * {{
        color: {theme["text"]} !important;
    }}

    .stButton > button:hover,
    .stFormSubmitButton > button:hover,
    .stDownloadButton > button:hover {{
        border-color: {theme["accent"]};
        color: {theme["text"]};
    }}

    .stDownloadButton > button {{
        background: {theme["accent"]} !important;
        border-color: {theme["accent"]} !important;
        color: #171b24 !important;
        font-weight: 800 !important;
    }}

    .stDownloadButton > button * {{
        color: #171b24 !important;
    }}

    div[data-testid="stForm"],
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: {theme["surface"]};
        border-color: {theme["border"]};
    }}

    input,
    textarea,
    div[data-baseweb="select"] > div {{
        background-color: {theme["input_bg"]} !important;
        border-color: {theme["border"]} !important;
        color: {theme["text"]} !important;
    }}

    input::placeholder,
    textarea::placeholder {{
        color: {theme["text_muted"]} !important;
        opacity: 1 !important;
    }}

    div[data-baseweb="select"] *,
    [data-testid="stTextInput"] *,
    [data-testid="stMetric"] *,
    [data-testid="stForm"] * {{
        color: {theme["text"]};
    }}

    [data-testid="stDataFrame"] {{
        color: {theme["text"]} !important;
        background: {theme["surface"]} !important;
    }}

    [data-testid="stDataFrame"] * {{
        color: {theme["text"]} !important;
    }}
</style>
""",
        unsafe_allow_html=True,
    )


def _format_money(value: object) -> str:
    if value is None:
        return "-"

    try:
        return f"{int(value):,}원"
    except (TypeError, ValueError):
        return str(value)


def _format_rate(value: object) -> str:
    if value is None:
        return "-"

    try:
        return f"{float(value):.2f}%"
    except (TypeError, ValueError):
        return str(value)


def _format_date(value: object) -> str:
    if value is None:
        return "-"

    return str(value)


def _find_product_pdf(account: dict) -> Path | None:
    document_key = str(account.get("rag_document_key") or "").strip()
    candidates = []

    if document_key:
        key_path = Path(document_key)
        if key_path.suffix.lower() == ".pdf":
            candidates.append(PDF_DIR / key_path.name)
        else:
            candidates.append(PDF_DIR / f"{document_key}.pdf")

    product_name = str(account.get("product_name") or "").replace(" ", "")
    if product_name:
        for pdf_path in PDF_DIR.glob("*.pdf"):
            normalized_pdf_name = pdf_path.stem.replace(" ", "")
            if product_name in normalized_pdf_name or normalized_pdf_name in product_name:
                candidates.append(pdf_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def _render_product_detail(account: dict) -> None:
    product_name = account.get("product_name") or "상품명 미확인"
    pdf_path = _find_product_pdf(account)

    amount_range = _format_amount_range(account.get("min_amount"), account.get("max_amount"))
    period_range = _format_period_range(account.get("min_period_months"), account.get("max_period_months"))
    rate_range = _format_rate_range(account.get("base_rate"), account.get("max_rate"))
    age_range = _format_age_range(account.get("age_min"), account.get("age_max"))

    st.markdown(
        f"""
<div class="detail-hero">
    <div class="profile-name">{product_name}</div>
    <div class="profile-meta">
        상품유형 {account.get("product_type", "-")} · 가입계좌 {account.get("account_number", "-")} · 적용금리 {_format_rate(account.get("applied_rate"))}
    </div>
    <div class="detail-grid">
        <div class="detail-card">
            <div class="detail-card-label">가입 가능 금액</div>
            <div class="detail-card-value">{amount_range}</div>
            <div class="detail-card-note">고객 월 저축 가능액과 함께 확인</div>
        </div>
        <div class="detail-card">
            <div class="detail-card-label">가입 가능 기간</div>
            <div class="detail-card-value">{period_range}</div>
            <div class="detail-card-note">현재 계약기간 {account.get("contract_months") or "-"}개월</div>
        </div>
        <div class="detail-card">
            <div class="detail-card-label">금리 범위</div>
            <div class="detail-card-value">{rate_range}</div>
            <div class="detail-card-note">고객 적용금리 {_format_rate(account.get("applied_rate"))}</div>
        </div>
    </div>
    <div class="detail-grid">
        <div class="detail-card">
            <div class="detail-card-label">가입 연령</div>
            <div class="detail-card-value">{age_range}</div>
            <div class="detail-card-note">상품 기본 조건 기준</div>
        </div>
        <div class="detail-card">
            <div class="detail-card-label">계좌 상태</div>
            <div class="detail-card-value">{account.get("account_status", "-")}</div>
            <div class="detail-card-note">만기·유지·추가 가입 상담 기준</div>
        </div>
        <div class="detail-card">
            <div class="detail-card-label">현재 잔액</div>
            <div class="detail-card-value">{_format_money(account.get("current_balance"))}</div>
            <div class="detail-card-note">월 납입 {_format_money(account.get("monthly_amount"))}</div>
        </div>
    </div>
    <div class="detail-status-grid">
        <div class="detail-status">
            <div class="detail-card-label">가입일</div>
            <div class="detail-card-value">{_format_date(account.get("join_date"))}</div>
        </div>
        <div class="detail-status">
            <div class="detail-card-label">만기일</div>
            <div class="detail-card-value">{_format_date(account.get("maturity_date"))}</div>
        </div>
        <div class="detail-status">
            <div class="detail-card-label">납입/예치 금액</div>
            <div class="detail-card-value">{_format_money(account.get("monthly_amount") or account.get("deposit_amount"))}</div>
        </div>
        <div class="detail-status">
            <div class="detail-card-label">상담 활용</div>
            <div class="detail-card-value">만기·우대·추가가입</div>
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    if pdf_path:
        st.download_button(
            "약관 PDF 다운로드",
            data=pdf_path.read_bytes(),
            file_name=pdf_path.name,
            mime="application/pdf",
            type="primary",
            use_container_width=True,
        )
    else:
        st.info("연결된 약관 PDF를 찾지 못했습니다. 상품설명서 파일명을 확인해 주세요.")


def _format_amount_range(min_amount: object, max_amount: object) -> str:
    if min_amount is None and max_amount is None:
        return "-"
    if max_amount is None:
        return f"{_format_money(min_amount)} 이상"
    if min_amount is None:
        return f"{_format_money(max_amount)} 이하"
    return f"{_format_money(min_amount)}~{_format_money(max_amount)}"


def _format_period_range(min_months: object, max_months: object) -> str:
    if min_months is None and max_months is None:
        return "-"
    if max_months is None or max_months == min_months:
        return f"{min_months or max_months}개월"
    return f"{min_months}~{max_months}개월"


def _format_rate_range(base_rate: object, max_rate: object) -> str:
    if base_rate is None and max_rate is None:
        return "-"
    if max_rate is None or max_rate == base_rate:
        return _format_rate(base_rate or max_rate)
    return f"{_format_rate(base_rate)}~{_format_rate(max_rate)}"


def _format_age_range(age_min: object, age_max: object) -> str:
    if age_min is None and age_max is None:
        return "제한 정보 없음"
    parts = []
    if age_min is not None:
        parts.append(f"만 {age_min}세 이상")
    if age_max is not None:
        parts.append(f"만 {age_max}세 이하")
    return " ".join(parts)


def _calculate_age(birth_date: object) -> int | None:
    if birth_date is None:
        return None

    try:
        today = date.today()
        return today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    except AttributeError:
        return None


def _fetch_top_recommendable_products(customer: dict, limit: int = 3) -> list[dict]:
    monthly_saving = customer.get("available_monthly_saving")
    customer_age = _calculate_age(customer.get("birth_date"))
    owned_product_ids = {
        account.get("product_id")
        for account in customer.get("accounts", [])
        if account.get("product_id") is not None and str(account.get("account_status", "")).upper() == "ACTIVE"
    }

    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                product_id,
                product_name,
                product_type,
                min_amount,
                max_amount,
                min_period_months,
                max_period_months,
                base_rate,
                max_rate,
                age_min,
                age_max,
                rag_document_key
            FROM products
            WHERE is_active = TRUE
            ORDER BY COALESCE(max_rate, base_rate, 0) DESC, product_id
            """
        )
        columns = [desc[0] for desc in cursor.description]
        products = [dict(zip(columns, row)) for row in cursor.fetchall()]
        cursor.close()
        conn.close()
    except Exception:
        return []

    ranked = []
    for product in products:
        if product.get("product_id") in owned_product_ids:
            continue

        min_amount = product.get("min_amount")
        max_amount = product.get("max_amount")
        if monthly_saving is not None:
            try:
                saving_amount = int(monthly_saving)
                if min_amount is not None and saving_amount < int(min_amount):
                    continue
                if max_amount is not None and saving_amount > int(max_amount):
                    continue
            except (TypeError, ValueError):
                pass

        if customer_age is not None:
            age_min = product.get("age_min")
            age_max = product.get("age_max")
            if age_min is not None and customer_age < int(age_min):
                continue
            if age_max is not None and customer_age > int(age_max):
                continue

        score = float(product.get("max_rate") or product.get("base_rate") or 0)
        ranked.append((score, product))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [product for _score, product in ranked[:limit]]


def _render_top_recommendations(customer: dict | None) -> None:
    st.markdown('<div class="section-title">추천 가능 상품 TOP 3</div>', unsafe_allow_html=True)

    if not customer:
        st.info("고객번호를 조회하면 추천 가능 상품 TOP 3가 표시됩니다.")
        return

    products = _fetch_top_recommendable_products(customer)
    if not products:
        st.info("현재 조건으로 추천 가능한 상품을 찾지 못했습니다.")
        return

    for rank, product in enumerate(products, 1):
        pdf_path = _find_product_pdf(product)
        st.markdown(
            f"""
<div class="recommend-card">
    <div>
        <span class="recommend-rank">{rank}</span>
        <span class="recommend-title">{product.get("product_name", "-")}</span>
    </div>
    <div class="recommend-meta">
        {product.get("product_type", "-")} · 금리 {_format_rate_range(product.get("base_rate"), product.get("max_rate"))} ·
        기간 {_format_period_range(product.get("min_period_months"), product.get("max_period_months"))}
    </div>
    <div class="recommend-reason">
        월 저축 가능액 기준 가입금액 조건을 충족하는 상품입니다. 상담 시 우대조건 충족 여부를 함께 확인하세요.
    </div>
</div>
""",
            unsafe_allow_html=True,
        )
        if pdf_path:
            st.download_button(
                f"{rank}위 상품 약관 PDF 다운로드",
                data=pdf_path.read_bytes(),
                file_name=pdf_path.name,
                mime="application/pdf",
                type="primary",
                use_container_width=True,
                key=f"recommend_pdf_{product.get('product_id', rank)}",
            )
        else:
            st.caption(f"{product.get('product_name', '추천 상품')}의 연결된 약관 PDF를 찾지 못했습니다.")


def _fetch_customer_dashboard(customer_number: str) -> tuple[dict | None, str | None]:
    normalized = customer_number.strip()
    if not normalized:
        return None, "고객번호를 입력해 주세요."

    if not normalized.isdigit():
        return None, "고객번호는 숫자로 입력해 주세요."

    customer_id = int(normalized)

    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                customer_id,
                customer_name,
                birth_date,
                customer_job,
                annual_income,
                income_level,
                main_bank_yn,
                salary_transfer_yn,
                auto_transfer_yn,
                card_usage_yn,
                marketing_agree_yn,
                transaction_months,
                available_monthly_saving
            FROM customers
            WHERE customer_id = %s;
            """,
            (customer_id,),
        )
        customer_row = cursor.fetchone()

        if customer_row is None:
            cursor.close()
            conn.close()
            return None, f"고객번호 {customer_id}번 고객을 찾을 수 없습니다."

        customer_columns = [desc[0] for desc in cursor.description]
        customer = dict(zip(customer_columns, customer_row))

        cursor.execute(
            """
            SELECT
                ca.account_id,
                ca.account_number,
                ca.product_id,
                COALESCE(p.product_name, '상품명 미확인') AS product_name,
                COALESCE(p.product_type, '유형 미확인') AS product_type,
                p.min_amount,
                p.max_amount,
                p.min_period_months,
                p.max_period_months,
                p.base_rate,
                p.max_rate,
                p.age_min,
                p.age_max,
                p.rag_document_key,
                ca.join_date,
                ca.maturity_date,
                ca.contract_months,
                ca.monthly_amount,
                ca.deposit_amount,
                ca.current_balance,
                ca.applied_rate,
                ca.account_status
            FROM customer_accounts ca
            LEFT JOIN products p ON p.product_id = ca.product_id
            WHERE ca.customer_id = %s
            ORDER BY ca.account_status, ca.maturity_date NULLS LAST, ca.join_date DESC;
            """,
            (customer_id,),
        )
        account_columns = [desc[0] for desc in cursor.description]
        accounts = [dict(zip(account_columns, row)) for row in cursor.fetchall()]

        cursor.close()
        conn.close()

        customer["accounts"] = accounts
        return customer, None
    except Exception as error:
        return None, f"고객 정보를 조회하는 중 오류가 발생했습니다: {error}"


def _build_next_conversation(
    messages: list, latest_user_prompt: str, latest_ai_content: str
) -> list[tuple[str, str]]:
    preserved = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            preserved.append((role, content))

    preserved.append(("user", latest_user_prompt))
    preserved.append(("assistant", latest_ai_content))
    return preserved


def _render_chat_message(role: str, content: str) -> None:
    label = "상담원" if role == "user" else "상담 어시스턴트"
    bubble_class = "user" if role == "user" else "assistant"
    st.markdown(f'<div class="chat-label">{label}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="chat-bubble {bubble_class}">', unsafe_allow_html=True)
    st.markdown(content)
    st.markdown("</div>", unsafe_allow_html=True)


def _run_assistant(prompt: str) -> str:
    selected_customer = st.session_state.selected_customer
    graph_prompt = prompt

    if selected_customer:
        customer_name = selected_customer.get("customer_name")
        customer_id = selected_customer.get("customer_id")
        graph_prompt = (
            f"현재 조회된 고객은 {customer_name}(고객번호 {customer_id})입니다.\n"
            f"고객 맥락을 반영해서 답변해 주세요.\n\n"
            f"사용자 질문: {prompt}"
        )

    st.session_state.conversation_messages.append(("user", graph_prompt))

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
                "prompt": graph_prompt,
                "conversation_length": len(st.session_state.conversation_messages),
            },
            metadata={"surface": "streamlit"},
        ) as observation:
            result = st.session_state.graph.invoke(
                {
                    "messages": st.session_state.conversation_messages,
                    "next": "",
                    "member_id": str(selected_customer.get("customer_id")) if selected_customer else None,
                    "customer_id": selected_customer.get("customer_id") if selected_customer else None,
                    "context": None,
                    "plan": [],
                    "current_step": 0,
                    "agent_outputs": {},
                }
            )

            ai_message = result["messages"][-1]
            ai_content = ai_message.content if hasattr(ai_message, "content") else str(ai_message)
            ai_content = ai_content.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")

            if observation is not None:
                observation.update(
                    output={
                        "response_preview": ai_content[:500],
                        "message_count": len(result.get("messages") or []),
                    }
                )

    flush_langfuse()
    return ai_content


def _handle_prompt(prompt: str) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("고객 상황에 맞는 답변을 준비하고 있습니다."):
        try:
            ai_content = _run_assistant(prompt)
            st.session_state.messages.append({"role": "assistant", "content": ai_content})
            st.session_state.conversation_messages = _build_next_conversation(
                st.session_state.messages[:-2],
                prompt,
                ai_content,
            )
        except Exception as error:
            error_msg = f"답변을 생성하는 중 오류가 발생했습니다: {error}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            flush_langfuse()


_apply_theme()

toolbar_left, toolbar_right = st.columns([0.74, 0.26])
with toolbar_right:
    theme_button_label = "라이트 모드로 전환" if st.session_state.dark_mode else "야간 모드로 전환"
    if st.button(theme_button_label, key="theme_toggle_button"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

st.markdown(
    """
<div class="hero">
    <div class="hero-label">KB 고객 상담 지원</div>
    <h1>고객 상담 지원 대시보드</h1>
    <p>고객 정보와 가입 상품을 확인하고, 상담에 필요한 내용을 빠르게 정리합니다.</p>
</div>
""",
    unsafe_allow_html=True,
)

dashboard_col, chat_col = st.columns([0.62, 0.38], gap="large")

with dashboard_col:
    st.markdown('<div class="section-title">고객 패널</div>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="insight-band">
    <strong>고객 정보와 가입 상품을 확인하는 영역입니다.</strong>
    <p>고객번호를 조회하면 고객 프로필, 가입상품, 선택 상품 상세를 한 화면에서 확인할 수 있습니다.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    with st.form("customer_lookup_form", clear_on_submit=False):
        lookup_col, button_col = st.columns([0.76, 0.24])
        with lookup_col:
            customer_number = st.text_input(
                "고객번호",
                placeholder="예: 1",
                label_visibility="collapsed",
            )
        with button_col:
            lookup_submitted = st.form_submit_button("조회")

    if lookup_submitted:
        customer, error = _fetch_customer_dashboard(customer_number)
        st.session_state.selected_customer = customer
        st.session_state.customer_lookup_error = error

        if customer:
            st.session_state.messages = []
            st.session_state.conversation_messages = []
            st.session_state.selected_account_index = None

    if st.session_state.customer_lookup_error:
        st.error(st.session_state.customer_lookup_error)

    customer = st.session_state.selected_customer

    if not customer:
        st.markdown(
            """
<div class="empty-dashboard">
    고객번호를 조회하면 이 영역에 고객 프로필, 거래 조건, 가입 상품 현황이 표시됩니다.
    상담원 패널은 조회된 고객 맥락을 바탕으로 상담을 지원합니다.
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        accounts = customer.get("accounts") or []
        active_accounts = [
            account
            for account in accounts
            if str(account.get("account_status", "")).upper() == "ACTIVE"
        ]
        total_balance = sum(int(account.get("current_balance") or 0) for account in active_accounts)
        monthly_total = sum(int(account.get("monthly_amount") or 0) for account in active_accounts)

        st.markdown(
            f"""
<div class="panel">
    <span class="status-pill">조회 완료</span>
    <div class="profile-name">{customer.get("customer_name", "-")}</div>
    <div class="profile-meta">
        고객번호 {customer.get("customer_id", "-")} · {customer.get("customer_job", "-")} ·
        소득수준 {customer.get("income_level", "-")}
    </div>
    <div class="metric-row">
        <div class="metric">
            <div class="metric-label">가입 상품</div>
            <div class="metric-value">{len(accounts)}개</div>
        </div>
        <div class="metric">
            <div class="metric-label">활성 계좌 잔액</div>
            <div class="metric-value">{_format_money(total_balance)}</div>
        </div>
        <div class="metric">
            <div class="metric-label">월 납입 합계</div>
            <div class="metric-value">{_format_money(monthly_total)}</div>
        </div>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

        condition_col_1, condition_col_2, condition_col_3 = st.columns(3)
        condition_col_1.metric("거래 개월", f"{customer.get('transaction_months') or 0}개월")
        condition_col_2.metric("연소득", f"{customer.get('annual_income') or 0:,}만원")
        condition_col_3.metric("월 저축 가능액", _format_money(customer.get("available_monthly_saving")))

        st.markdown('<div class="section-title">가입 상품 현황</div>', unsafe_allow_html=True)

        if not accounts:
            st.info("현재 조회된 가입 상품이 없습니다.")
        else:
            st.caption("상품명을 클릭하면 상품 요약과 약관 PDF 다운로드가 표시됩니다.")
            st.markdown(
                """
<div class="account-table-header">
    <div style="display:grid; grid-template-columns: 2.1fr 0.7fr 0.85fr 1fr 1fr 0.9fr 1fr; gap:0.7rem; align-items:center;">
        <div>상품명</div>
        <div>유형</div>
        <div>상태</div>
        <div>가입일</div>
        <div>만기일</div>
        <div>금리</div>
        <div>잔액</div>
    </div>
</div>
""",
                unsafe_allow_html=True,
            )

            for index, account in enumerate(accounts):
                with st.container(border=True):
                    row_cols = st.columns([2.1, 0.7, 0.85, 1, 1, 0.9, 1])
                    with row_cols[0]:
                        if st.button(
                            account.get("product_name", "상품명 미확인"),
                            key=f"account_detail_{account.get('account_id', index)}",
                            use_container_width=True,
                        ):
                            st.session_state.selected_account_index = index
                    row_cols[1].markdown(f"**{account.get('product_type', '-')}**")
                    row_cols[2].markdown(f"**{account.get('account_status', '-')}**")
                    row_cols[3].markdown(_format_date(account.get("join_date")))
                    row_cols[4].markdown(_format_date(account.get("maturity_date")))
                    row_cols[5].markdown(_format_rate(account.get("applied_rate")))
                    row_cols[6].markdown(_format_money(account.get("current_balance")))

            selected_index = st.session_state.selected_account_index
            if selected_index is not None and 0 <= selected_index < len(accounts):
                selected_account = accounts[selected_index]
                _render_product_detail(selected_account)

        st.markdown('<div class="section-title">상담 포인트</div>', unsafe_allow_html=True)
        st.markdown(
            """
<div class="panel">
    <div class="journey-step">
        <div class="step-number">1</div>
        <div class="step-copy">
            <strong>기존 상품 점검</strong>
            <span>만기, 금리, 납입액을 보고 유지 또는 추가 가입 상담을 시작합니다.</span>
        </div>
    </div>
    <div class="journey-step">
        <div class="step-number">2</div>
        <div class="step-copy">
            <strong>우대조건 확인</strong>
            <span>급여이체, 자동이체, 카드사용 등 충족 가능한 조건을 확인합니다.</span>
        </div>
    </div>
    <div class="journey-step">
        <div class="step-number">3</div>
        <div class="step-copy">
            <strong>다음 행동 제안</strong>
            <span>고객 상황에 맞는 추가 납입, 만기 관리, 추천 상품을 상담합니다.</span>
        </div>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

with chat_col:
    selected_name = (
        st.session_state.selected_customer.get("customer_name")
        if st.session_state.selected_customer
        else "고객 미조회"
    )
    st.markdown(f'<div class="section-title">상담원 패널 · {selected_name}</div>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="insight-band">
    <strong>상담원이 고객에게 전달할 내용을 정리하는 영역입니다.</strong>
    <p>고객번호를 먼저 조회하면 상담 어시스턴트가 해당 고객 맥락을 반영해 답변합니다.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    _render_top_recommendations(st.session_state.selected_customer)

    st.markdown('<div class="chat-scroll-hint">상담 내역은 이 영역 안에서 스크롤됩니다.</div>', unsafe_allow_html=True)
    chat_history = st.container(height=460, border=True)

    prompt = None

    with st.form("chat_prompt_form", clear_on_submit=True):
        input_col, submit_col = st.columns([0.86, 0.14])
        with input_col:
            typed_prompt = st.text_input(
                "질문",
                placeholder="예: 이 고객에게 설명할 만기 안내 문장을 만들어줘",
                label_visibility="collapsed",
            )
        with submit_col:
            submitted = st.form_submit_button(">")

    if submitted:
        prompt = typed_prompt.strip()
        if not prompt:
            st.warning("질문을 입력해 주세요.")

    st.caption("상담원 추천 질문")
    suggested_prompts = [
        ("가입상품 요약", "상담원이 고객에게 설명할 수 있게 이 고객의 가입 상품 현황과 핵심 상담 포인트를 요약해줘."),
        ("추가 추천", "이 고객의 기존 가입 상품과 월 저축 가능액을 고려해서 상담원이 제안할 추가 적금 후보를 정리해줘."),
        ("만기 관리", "이 고객의 만기 예정 상품을 확인하고 상담원이 안내할 만기 전 체크포인트를 정리해줘."),
        ("우대조건 확인", "이 고객이 받을 수 있는 우대금리 조건을 기존 거래 조건 중심으로 점검하고 상담 멘트로 정리해줘."),
    ]
    suggestion_cols = st.columns(2)
    for index, (label, suggested_prompt) in enumerate(suggested_prompts):
        with suggestion_cols[index % 2]:
            if st.button(label, key=f"suggested_prompt_{index}", use_container_width=True):
                if st.session_state.selected_customer:
                    prompt = suggested_prompt
                else:
                    st.warning("추천 질문은 고객번호를 먼저 조회한 뒤 사용할 수 있습니다.")

    if prompt:
        _handle_prompt(prompt)

    with chat_history:
        if not st.session_state.messages:
            st.info("상담 내역이 없습니다. 고객을 조회한 뒤 상담 질문을 입력해 주세요.")

        for msg in st.session_state.messages:
            _render_chat_message(msg["role"], msg["content"])
