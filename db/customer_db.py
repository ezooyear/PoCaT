"""
고객 데이터 관리 모듈
- JSON 파일에서 고객 더미데이터를 로드/조회
- 고객 정보를 LLM 컨텍스트용 텍스트로 변환
"""
import json
import os
from datetime import datetime, date

# ─── 경로 설정 ───
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUSTOMERS_FILE = os.path.join(BASE_DIR, "data", "customers.json")


def load_customers() -> list:
    """고객 데이터를 JSON 파일에서 로드합니다."""
    if not os.path.exists(CUSTOMERS_FILE):
        print(f"⚠️  고객 데이터 파일이 없습니다: {CUSTOMERS_FILE}")
        return []

    with open(CUSTOMERS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("customers", [])


def get_customer(customer_id: str) -> dict | None:
    """고객 ID로 고객 정보를 조회합니다."""
    customers = load_customers()
    for customer in customers:
        if customer["id"] == customer_id:
            return customer
    return None


def get_customer_list() -> list[dict]:
    """고객 목록을 (id, name) 형태로 반환합니다."""
    customers = load_customers()
    return [{"id": c["id"], "name": c["name"]} for c in customers]


def get_customer_products(customer_id: str) -> list:
    """고객의 가입 상품 목록을 반환합니다."""
    customer = get_customer(customer_id)
    if customer is None:
        return []
    return customer.get("products", [])


def format_customer_info(customer: dict) -> str:
    """
    고객 정보를 LLM에 전달할 텍스트로 변환합니다.
    
    Args:
        customer: 고객 데이터 딕셔너리
    
    Returns:
        str: 포맷된 고객 정보 텍스트
    """
    if not customer:
        return "고객 정보 없음"

    today = date.today()
    lines = []
    lines.append(f"### 고객 기본 정보")
    lines.append(f"- 이름: {customer['name']}")
    lines.append(f"- 나이: {customer['age']}세")
    lines.append(f"- 직업: {customer['job']}")
    lines.append(f"- 연락처: {customer.get('phone', '미등록')}")
    lines.append("")

    products = customer.get("products", [])
    if products:
        lines.append(f"### 가입 상품 ({len(products)}건)")
        for i, prod in enumerate(products, 1):
            lines.append(f"\n**{i}. {prod['name']}** ({prod['type']})")

            if prod["type"] == "정기예금":
                lines.append(f"   - 예치금액: {prod['principal']:,.0f}원")
            else:
                lines.append(f"   - 월 납입액: {prod['monthly_payment']:,.0f}원")
                if prod.get("paid_count") is not None and prod.get("total_count") is not None:
                    lines.append(f"   - 납입 현황: {prod['paid_count']}/{prod['total_count']}회")
                    remaining = prod["total_count"] - prod["paid_count"]
                    lines.append(f"   - 남은 납입 횟수: {remaining}회")

            lines.append(f"   - 기본금리: {prod['rate']}%")
            if prod.get("bonus_rate", 0) > 0:
                lines.append(f"   - 우대금리: +{prod['bonus_rate']}%")
                lines.append(f"   - 적용금리: {prod['rate'] + prod['bonus_rate']}%")

            lines.append(f"   - 가입일: {prod['start_date']}")
            lines.append(f"   - 만기일: {prod['end_date']}")

            # 만기까지 남은 일수 계산
            try:
                end_date = datetime.strptime(prod["end_date"], "%Y-%m-%d").date()
                days_left = (end_date - today).days
                if days_left > 0:
                    lines.append(f"   - 만기까지 남은 기간: {days_left}일")
                elif days_left == 0:
                    lines.append(f"   - ⚠️ 오늘 만기!")
                else:
                    lines.append(f"   - ✅ 이미 만기됨 ({abs(days_left)}일 전)")
            except ValueError:
                pass
    else:
        lines.append("### 가입 상품: 없음")

    return "\n".join(lines)
