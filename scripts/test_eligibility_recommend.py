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
고객명 고객_테스트
나이: 32
직업: 직장인
월 가용 저축액: 300000
소득: 4200000
급여이체 있음
자동이체 있음
카드 사용 있음
주거래 고객
""".strip()

    product_output = """
나라사랑적금
군인 전용 상품
가입 대상 연령: 만 18세 이상 만 34세 이하
월 최소 납입: 10000
월 최대 납입: 400000
판매중

---

일반정기적금
가입 대상 연령: 만 19세 이상 만 60세 이하
월 최소 납입: 10000
월 최대 납입: 500000
급여이체 우대
자동이체 우대
카드 우대
판매중
가입기간 12개월 24개월 36개월

---

판매종료적금
가입 대상 연령: 만 19세 이상 만 60세 이하
월 최소 납입: 10000
판매 종료
""".strip()

    return {
        "messages": [],
        "user_query": "월 30만원씩 24개월 적금 추천해줘",
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

    military_result = find_result(eligibility_results, "나라사랑적금")
    general_result = find_result(eligibility_results, "일반정기적금")
    closed_result = find_result(eligibility_results, "판매종료적금")

    assert military_result["status"] == "rejected"
    assert general_result["status"] == "eligible"
    assert closed_result["status"] == "rejected"

    state["financial_results"] = [
        {
            "product_name": "일반정기적금",
            "estimated_interest": 180000,
            "maturity_amount": 7380000,
        }
    ]

    recommend_update = recommend_agent_node(state)
    state = merge_state(state, recommend_update)

    recommend_result = state.get("recommend_result") or {}
    recommend_payload = recommend_result.get("result", {})
    recommendation_results = state.get("recommendation_results") or []
    recommended_names = [item.get("product_name", "") for item in recommendation_results]

    assert recommend_payload.get("status") == "recommended"
    assert any("일반정기적금" in name for name in recommended_names)
    assert not any("나라사랑적금" in name for name in recommended_names)
    assert not any("판매종료적금" in name for name in recommended_names)


def test_customer_profile_missing_blocks_eligible_true() -> None:
    state = make_base_state()
    state["agent_outputs"]["customer_agent"] = "고객명 고객_테스트"

    eligibility_update = eligibility_agent_node(state)
    results = eligibility_update.get("eligibility_results") or []

    assert results, "eligibility_results should not be empty"
    assert all(item.get("status") != "eligible" for item in results)
    assert any("고객 나이 정보" in item.get("missing_fields", []) for item in results)


def test_invalid_product_name_becomes_invalid_product() -> None:
    state = make_base_state()
    state["product_candidates"] = [
        {
            "product_name": "적용조건 우대이율 신규가입일 영업점 안내 문장입니다",
            "raw_text": "적용조건 우대이율 신규가입일 영업점 안내 문장입니다",
        }
    ]

    eligibility_update = eligibility_agent_node(state)
    results = eligibility_update.get("eligibility_results") or []

    assert len(results) == 1
    assert results[0]["status"] == "invalid_product"
    assert results[0]["eligible"] is False


def test_financial_results_missing_defers_recommendation() -> None:
    state = make_base_state()
    eligibility_update = eligibility_agent_node(state)
    state = merge_state(state, eligibility_update)
    state["financial_results"] = []

    recommend_update = recommend_agent_node(state)
    recommend_payload = recommend_update.get("recommend_result", {}).get("result", {})

    assert recommend_payload.get("status") == "recommendation_deferred"
    assert recommend_payload.get("fallback_reason") == "financial_results_missing"
    assert not recommend_payload.get("recommendations")


def test_no_eligible_product_does_not_force_recommendation() -> None:
    state = make_base_state()
    state["eligibility_results"] = [
        {
            "product_name": "판매종료적금",
            "eligible": False,
            "status": "rejected",
            "reasons": ["판매 종료"],
            "missing_fields": [],
            "source_agent": "eligibility_agent",
        }
    ]
    state["product_candidates"] = [{"product_name": "판매종료적금", "raw_text": "판매 종료"}]
    state["financial_results"] = [{"product_name": "판매종료적금", "estimated_interest": 1000}]

    recommend_update = recommend_agent_node(state)
    recommend_payload = recommend_update.get("recommend_result", {}).get("result", {})

    assert recommend_payload.get("status") == "no_eligible_product"
    assert recommend_payload.get("fallback_reason") == "no_eligible_product"
    assert not recommend_payload.get("recommendations")


def main() -> None:
    test_basic_flow()
    test_customer_profile_missing_blocks_eligible_true()
    test_invalid_product_name_becomes_invalid_product()
    test_financial_results_missing_defers_recommendation()
    test_no_eligible_product_does_not_force_recommendation()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
