"""
Calculation Agent
- Customer Agent가 조회한 고객/계좌/납입 데이터를 바탕으로 계산 수행
- LLM에 의존하지 않고 customer_result 기반으로 규칙 기반 계산
- 결과를 calculation_result에 구조화하여 저장
"""
"""
기존:
LLM + Tool 호출 + JSON 파싱

수정:
customer_result 직접 사용
→ 납입 이력 집계
→ 만기까지 남은 일수 계산
→ 남은 납입 횟수 계산
→ 단순 예상 이자 계산
→ calculation_result 저장
"""

from datetime import date, datetime
from collections import defaultdict

from graph.state import AgentState


def calculation_agent_node(state: AgentState) -> dict:
    """
    Calculation Agent 노드

    역할:
    - customer_result에 저장된 고객/계좌/납입 데이터를 기준으로 계산한다.
    - 만기까지 남은 일수, 납입 횟수, 남은 납입 횟수, 잔액 합계 등을 계산한다.
    - 계산 결과를 calculation_result에 저장한다.
    """

    customer_result = state.get("customer_result")
    customer_id = state.get("customer_id") or state.get("member_id")
    task_type = state.get("task_type")

    if not customer_result:
        return {
            "calculation_result": {
                "ok": False,
                "customer_id": customer_id,
                "error": "customer_result가 없습니다. Customer Agent를 먼저 실행해야 합니다.",
                "missing_fields": ["customer_result"],
            },
            "errors": ["calculation_agent: customer_result가 없습니다."],
        }

    if not customer_result.get("ok", True):
        return {
            "calculation_result": {
                "ok": False,
                "customer_id": customer_id,
                "error": "Customer Agent 조회 결과가 정상적이지 않습니다.",
                "customer_error": customer_result.get("error"),
                "missing_fields": ["valid_customer_result"],
            },
            "errors": ["calculation_agent: customer_result가 정상적이지 않습니다."],
        }

    customer_profile = (
        customer_result.get("customer_profile")
        or customer_result.get("customer")
        or {}
    )

    accounts = customer_result.get("accounts") or []
    payment_history = customer_result.get("payment_history") or []

    today = date.today()

    # account_id별 납입 이력 집계
    payment_summary_by_account = _summarize_payment_history(payment_history)

    account_calculations = []

    total_balance = 0
    active_account_count = 0
    total_monthly_payment = 0

    for account in accounts:
        account_id = account.get("account_id")
        product_name = account.get("product_name")
        product_type = account.get("product_type")
        account_status = account.get("account_status")

        current_balance = _to_number(account.get("current_balance"))
        deposit_amount = _to_number(account.get("deposit_amount"))
        monthly_amount = _to_number(
            account.get("monthly_amount")
            or account.get("monthly_payment_amount")
        )
        applied_rate = _to_number(account.get("applied_rate"))
        contract_months = _to_int(account.get("contract_months"))

        maturity_date = _to_date(account.get("maturity_date"))
        join_date = _to_date(account.get("join_date"))

        if account_status == "ACTIVE":
            active_account_count += 1

        total_balance += current_balance

        if account_status == "ACTIVE" and monthly_amount:
            total_monthly_payment += monthly_amount

        payment_summary = payment_summary_by_account.get(account_id, {})
        paid_count = payment_summary.get("paid_count", 0)
        total_paid_amount = payment_summary.get("total_paid_amount", 0)
        last_payment_date = payment_summary.get("last_payment_date")

        days_to_maturity = None
        if maturity_date:
            days_to_maturity = (maturity_date - today).days

        remaining_payment_count = None
        if contract_months is not None:
            remaining_payment_count = max(contract_months - paid_count, 0)

        estimated_interest = _estimate_simple_interest(
            product_type=product_type,
            deposit_amount=deposit_amount,
            monthly_amount=monthly_amount,
            current_balance=current_balance,
            applied_rate=applied_rate,
            contract_months=contract_months,
        )

        estimated_maturity_amount = None
        if estimated_interest is not None:
            base_amount = current_balance or deposit_amount or total_paid_amount
            estimated_maturity_amount = base_amount + estimated_interest

        account_calculations.append({
            "account_id": account_id,
            "product_id": account.get("product_id"),
            "product_name": product_name,
            "product_type": product_type,
            "account_status": account_status,
            "join_date": str(join_date) if join_date else None,
            "maturity_date": str(maturity_date) if maturity_date else None,
            "contract_months": contract_months,
            "current_balance": current_balance,
            "deposit_amount": deposit_amount,
            "monthly_amount": monthly_amount,
            "applied_rate": applied_rate,
            "days_to_maturity": days_to_maturity,
            "paid_count": paid_count,
            "total_paid_amount": total_paid_amount,
            "last_payment_date": str(last_payment_date) if last_payment_date else None,
            "remaining_payment_count": remaining_payment_count,
            "estimated_interest": estimated_interest,
            "estimated_maturity_amount": estimated_maturity_amount,
            "early_termination_loss": None,
        })

    available_monthly_saving = _to_number(
        customer_profile.get("available_monthly_saving")
    )

    # 사용자가 직접 희망 납입액을 말했는지 간단히 추출
    user_query = state.get("user_query") or _get_last_user_text(state.get("messages", []))
    requested_monthly_amount = _extract_monthly_amount(user_query)

    if requested_monthly_amount is not None:
        saving_reference = requested_monthly_amount
        saving_reference_type = "user_requested_amount"
    else:
        saving_reference = available_monthly_saving
        saving_reference_type = "estimated_monthly_saving_capacity"

    monthly_burden_after_existing_payments = None
    if saving_reference is not None:
        monthly_burden_after_existing_payments = saving_reference - total_monthly_payment

    calculation_result = {
        "ok": True,
        "customer_id": customer_id,
        "task_type": task_type,
        "total_balance": total_balance,
        "active_account_count": active_account_count,
        "total_monthly_payment": total_monthly_payment,
        "available_monthly_saving": available_monthly_saving,
        "requested_monthly_amount": requested_monthly_amount,
        "saving_reference": saving_reference,
        "saving_reference_type": saving_reference_type,
        "monthly_burden_after_existing_payments": monthly_burden_after_existing_payments,
        "account_calculations": account_calculations,
        "missing_fields": [],
        "calculation_summary": _make_calculation_summary(
            customer_id=customer_id,
            total_balance=total_balance,
            active_account_count=active_account_count,
            total_monthly_payment=total_monthly_payment,
            available_monthly_saving=available_monthly_saving,
            requested_monthly_amount=requested_monthly_amount,
            account_calculations=account_calculations,
        ),
    }

    return {
        "calculation_result": calculation_result
    }


def _summarize_payment_history(payment_history: list[dict]) -> dict:
    """account_id별 납입 횟수, 총 납입액, 마지막 납입일 집계"""

    summary = defaultdict(lambda: {
        "paid_count": 0,
        "total_paid_amount": 0,
        "last_payment_date": None,
    })

    for payment in payment_history:
        account_id = payment.get("account_id")
        if account_id is None:
            continue

        payment_amount = _to_number(payment.get("payment_amount"))
        payment_date = _to_date(payment.get("payment_date"))

        summary[account_id]["paid_count"] += 1
        summary[account_id]["total_paid_amount"] += payment_amount

        last_date = summary[account_id]["last_payment_date"]
        if payment_date and (last_date is None or payment_date > last_date):
            summary[account_id]["last_payment_date"] = payment_date

    return dict(summary)


def _estimate_simple_interest(
    product_type: str | None,
    deposit_amount: float,
    monthly_amount: float,
    current_balance: float,
    applied_rate: float,
    contract_months: int | None,
) -> float | None:
    """
    단순 예상 이자 계산.
    정확한 은행 이자 계산식이 아니라 상담용 추정치다.

    예금:
    원금 × 연이율 × 기간/12

    적금:
    월납입액 × 납입개월 × 연이율 × (납입개월+1)/(2*12)
    """

    if applied_rate is None:
        return None

    annual_rate = applied_rate / 100

    if not contract_months:
        return None

    product_type_text = str(product_type or "")

    if "예금" in product_type_text:
        principal = deposit_amount or current_balance
        if not principal:
            return None

        interest = principal * annual_rate * (contract_months / 12)
        return round(interest)

    if "적금" in product_type_text or "부금" in product_type_text:
        if not monthly_amount:
            return None

        # 적금 평균 잔액 방식 단순 추정
        interest = monthly_amount * contract_months * annual_rate * ((contract_months + 1) / (2 * 12))
        return round(interest)

    # 상품 유형이 애매하면 현재 잔액 기준 단순 계산
    principal = current_balance or deposit_amount
    if principal:
        interest = principal * annual_rate * (contract_months / 12)
        return round(interest)

    return None


def _make_calculation_summary(
    customer_id,
    total_balance,
    active_account_count,
    total_monthly_payment,
    available_monthly_saving,
    requested_monthly_amount,
    account_calculations,
) -> str:
    """계산 결과 요약 문장 생성"""

    if requested_monthly_amount is not None:
        saving_text = (
            f"사용자가 희망한 월 납입액은 {requested_monthly_amount:,.0f}원입니다. "
            f"DB상 추정 월 저축 여력은 {available_monthly_saving:,.0f}원입니다."
            if available_monthly_saving is not None
            else f"사용자가 희망한 월 납입액은 {requested_monthly_amount:,.0f}원입니다."
        )
    else:
        saving_text = (
            f"추정 월 저축 여력은 {available_monthly_saving:,.0f}원입니다."
            if available_monthly_saving is not None
            else "추정 월 저축 여력 정보는 없습니다."
        )

    return (
        f"고객 ID {customer_id}번의 계좌 계산을 완료했습니다. "
        f"총 계좌 잔액은 {total_balance:,.0f}원이고, "
        f"활성 계좌는 {active_account_count}개입니다. "
        f"현재 활성 계좌 기준 월 납입액 합계는 {total_monthly_payment:,.0f}원입니다. "
        f"{saving_text} "
        f"계좌별 만기일, 납입 횟수, 남은 납입 횟수, 예상 이자를 계산했습니다."
    )


def _extract_monthly_amount(text: str | None) -> int | None:
    """
    사용자 질문에서 '월 30만 원', '월 300000원' 같은 희망 납입액을 간단히 추출.
    복잡한 자연어 처리는 추후 Recommend Agent에서 보완 가능.
    """

    if not text:
        return None

    import re

    text = text.replace(",", "").replace(" ", "")

    # 월30만원, 월30만, 매월30만원
    match = re.search(r"(월|매월)(\d+)만원", text)
    if match:
        return int(match.group(2)) * 10000

    match = re.search(r"(월|매월)(\d+)만", text)
    if match:
        return int(match.group(2)) * 10000

    # 월300000원, 매월300000원
    match = re.search(r"(월|매월)(\d+)원", text)
    if match:
        return int(match.group(2))

    return None


def _get_last_user_text(messages: list) -> str:
    """messages에서 마지막 사용자 질문 추출"""

    for msg in reversed(messages):
        if isinstance(msg, tuple) and len(msg) >= 2:
            role, content = msg[0], msg[1]
            if role in ["user", "human"]:
                return str(content)

        if hasattr(msg, "type") and hasattr(msg, "content"):
            if msg.type in ["human", "user"]:
                return str(msg.content)

    return ""


def _to_date(value):
    """date/datetime/string 값을 date로 변환"""

    if value is None:
        return None

    if isinstance(value, date) and not isinstance(value, datetime):
        return value

    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except Exception:
            return None

    return None


def _to_number(value) -> float:
    """숫자 변환"""

    if value is None:
        return 0

    try:
        return float(value)
    except Exception:
        return 0


def _to_int(value) -> int | None:
    """정수 변환"""

    if value is None:
        return None

    try:
        return int(value)
    except Exception:
        return None