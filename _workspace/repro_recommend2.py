"""추천 버튼 오류 재현 v2 — app._run_assistant를 최대한 충실히 모사.
실제 고객(고객_123)을 DB에서 불러와 고객 컨텍스트를 붙이고, 추천 추천질문 그대로 invoke."""
import sys, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datetime import date
from db.postgres_db import get_connection
from graph.builder import build_graph

CUSTOMER_ID = 123
SUGGESTED = ("내 가입 상품, 월 저축 가능액, 가입 가능 조건을 종합해서 "
             "나에게 추천할 만한 예적금 상품을 순위와 이유, 주의사항까지 알려줘.")


def fetch_customer(cid):
    conn = get_connection(); cur = conn.cursor()
    cur.execute("""SELECT customer_id, customer_name, birth_date, customer_job, annual_income,
        income_level, main_bank_yn, salary_transfer_yn, auto_transfer_yn, card_usage_yn,
        marketing_agree_yn, transaction_months, available_monthly_saving
        FROM customers WHERE customer_id=%s""", (cid,))
    row = cur.fetchone()
    cols = [d[0] for d in cur.description]
    cust = dict(zip(cols, row))
    cur.execute("""SELECT ca.account_id, ca.account_number, ca.product_id,
        COALESCE(p.product_name,'상품명 미확인') product_name,
        COALESCE(p.product_type,'유형 미확인') product_type,
        ca.applied_rate, ca.maturity_date, ca.current_balance, ca.monthly_amount, ca.account_status
        FROM customer_accounts ca LEFT JOIN products p ON p.product_id=ca.product_id
        WHERE ca.customer_id=%s""", (cid,))
    acols = [d[0] for d in cur.description]
    cust["accounts"] = [dict(zip(acols, r)) for r in cur.fetchall()]
    cur.close(); conn.close()
    return cust


def age_band(bd):
    try:
        t = date.today(); a = t.year-bd.year-((t.month,t.day)<(bd.month,bd.day)); return f"{a//10*10}대"
    except Exception:
        return "나이대 미확인"


def build_context(c):
    acts = [a for a in c.get("accounts",[]) if str(a.get("account_status","")).upper()=="ACTIVE"]
    lines = [f"- {a.get('product_name')} ({a.get('product_type')}, 적용금리 {a.get('applied_rate')}, 만기 {a.get('maturity_date')}, 잔액 {a.get('current_balance')})" for a in acts[:5]] or ["- 현재 활성 가입 상품 없음"]
    def money(v):
        try:
            return f"{int(v):,}원"
        except (TypeError, ValueError):
            return str(v)
    return "\n".join(["[내 계정 요약]", f"- 이름: {c.get('customer_name')}", f"- 나이대: {age_band(c.get('birth_date'))}",
        f"- 직업: {c.get('customer_job')}", f"- 소득수준: {c.get('income_level')}",
        f"- 거래 개월: {c.get('transaction_months')}개월", f"- 월 저축 가능액: {money(c.get('available_monthly_saving'))}",
        "", "[가입 중인 상품]", *lines])


try:
    cust = fetch_customer(CUSTOMER_ID)
    ctx = build_context(cust)
    graph_prompt = (
        f"현재 로그인한 고객은 {cust.get('customer_name')}(테스트 고객번호 {cust.get('customer_id')})입니다.\n"
        f"고객 본인이 이해하기 쉬운 말투로 답변해 주세요.\n\n{ctx}\n\n사용자 질문: {SUGGESTED}"
    )
    g = build_graph()
    result = g.invoke({
        "messages": [("user", graph_prompt)], "next": "",
        "member_id": str(CUSTOMER_ID), "customer_id": CUSTOMER_ID,
        "context": None, "plan": [], "current_step": 0, "agent_outputs": {},
    })
    msg = result["messages"][-1]
    print("=== OK (no crash) ===")
    print(getattr(msg, "content", msg)[:1500])
except Exception:
    print("=== EXCEPTION (this is the bug) ===")
    traceback.print_exc()
