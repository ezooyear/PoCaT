"""
Customer Agent
- 고객 기본 정보, 가입 계좌, 납입 이력 조회 담당
- customer_id 기준 고정 SQL 조회
- 조회 결과를 customer_result에 구조화하여 저장
"""

from graph.state import AgentState
from db.postgres_db import get_customer_dashboard_data


def customer_agent_node(state: AgentState) -> dict:
    """
    Customer Agent 노드

    역할:
    - State의 customer_id 또는 member_id를 기준으로 고객 정보를 조회한다.
    - NL2SQL에 의존하지 않고 고정 SQL 함수를 사용한다.
    - 조회 결과를 customer_result에 저장한다.
    """

    customer_id = state.get("customer_id") or state.get("member_id")

    if customer_id is None:
        return {
            "customer_result": {
                "ok": False,
                "error": "customer_id/member_id가 State에 없습니다.",
            },
            "errors": ["customer_id/member_id가 State에 없습니다."],
        }

    try:
        data = get_customer_dashboard_data(int(customer_id))

        if data.get("customer") is None:
            return {
                "customer_result": {
                    "ok": False,
                    "customer_id": int(customer_id),
                    "error": f"customer_id={customer_id} 고객을 찾을 수 없습니다.",
                },
                "errors": [f"customer_id={customer_id} 고객을 찾을 수 없습니다."],
            }

        customer_result = {
            "ok": True,
            "customer_id": int(customer_id),
            "customer_profile": data.get("customer"),
            "accounts": data.get("accounts", []),
            "payment_history": data.get("payment_history", []),
            "summary": _make_customer_summary(data),
        }

        return {
            "customer_result": customer_result
        }

    except Exception as e:
        return {
            "customer_result": {
                "ok": False,
                "customer_id": customer_id,
                "error": str(e),
            },
            "errors": [str(e)],
        }


def _make_customer_summary(data: dict) -> str:
    """
    조회된 고객 데이터를 간단히 요약한다.
    LLM을 쓰지 않고 rule-based로 요약한다.
    """

    customer = data.get("customer") or {}
    accounts = data.get("accounts") or []
    payment_history = data.get("payment_history") or []

    customer_name = customer.get("customer_name", "해당 고객")
    customer_id = customer.get("customer_id")

    active_accounts = [
        acc for acc in accounts
        if acc.get("account_status") == "ACTIVE"
    ]

    total_balance = sum(
        acc.get("current_balance") or 0
        for acc in accounts
    )

    return (
        f"고객 ID {customer_id}번 {customer_name}님의 정보를 조회했습니다. "
        f"가입 계좌는 총 {len(accounts)}개이며, "
        f"현재 활성 계좌는 {len(active_accounts)}개입니다. "
        f"전체 계좌 잔액 합계는 {total_balance:,.0f}원이고, "
        f"납입 이력은 총 {len(payment_history)}건입니다."
    )