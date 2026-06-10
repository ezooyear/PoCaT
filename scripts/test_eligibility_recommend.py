"""Lightweight local tests for eligibility/recommend flow.

Run:
python scripts/test_eligibility_recommend.py
"""
from copy import deepcopy
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.eligibility.agent import eligibility_agent_node
from agents.recommend.agent import recommend_agent_node


def make_base_state() -> dict:
    customer_output = """
고객명: 고객_테스트
나이: 32
직업: 공무원
월 가용 저축액: 300000
급여이체 있음
자동이체 있음
카드 사용 있음
주거래 고객
""".strip()

    product_output = """
장병내일적금
군인 전용 상품
가입 가능 연령: 만 18세 이상 만 34세 이하
월 최소 납입: 10000
월 최대 납입: 400000
판매중

---

일반정기적금
가입 가능 연령: 만 19세 이상 만 60세 이하
월 최소 납입: 10000
월 최대 납입: 500000
급여이체 우대
자동이체 우대
카드 우대
판매중

---

판매종료적금
가입 가능 연령: 만 19세 이상 만 60세 이하
월 최소 납입: 10000
판매 종료
""".strip()

    return {
        "messages": [],
        "next": "",
        "member_id": None,
        "context": None,
        "plan": [],
        "current_step": 0,
        "agent_outputs": {
            "customer_agent": customer_output,
            "product_agent": product_output,
        },
        "customer_profile": None,
        "customer_accounts": None,
        "payment_history": None,
        "product_candidates": None,
        "eligibility_results": None,
        "financial_results": None,
        "recommendation_results": None,
        "validation_results": None,
    }


def merge_state(base: dict, update: dict) -> dict:
    merged = deepcopy(base)
    merged.update(update)
    return merged


def find_result(results: list[dict], keyword: str) -> dict:
    for item in results:
        if keyword in item.get("product_name", ""):
            return item
    raise AssertionError(f"Result containing '{keyword}' not found")


def test_basic_flow() -> None:
    state = make_base_state()

    eligibility_update = eligibility_agent_node(state)
    state = merge_state(state, eligibility_update)

    eligibility_results = state.get("eligibility_results") or []
    assert eligibility_results, "eligibility_results should not be empty"

    military_result = find_result(eligibility_results, "장병내일적금")
    general_result = find_result(eligibility_results, "일반정기적금")
    closed_result = find_result(eligibility_results, "판매종료적금")

    assert military_result["eligible"] is False
    assert general_result["eligible"] is True
    assert closed_result["eligible"] is False

    recommend_update = recommend_agent_node(state)
    state = merge_state(state, recommend_update)

    recommendation_results = state.get("recommendation_results") or []
    recommended_names = [item.get("product_name", "") for item in recommendation_results]

    assert not any("장병내일적금" in name for name in recommended_names)
    assert any("일반정기적금" in name for name in recommended_names)
    assert not any("판매종료적금" in name for name in recommended_names)


def main() -> None:
    test_basic_flow()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
