"""Fixture-based local tests for eligibility/recommend flow.

Run:
python scripts/test_with_product_fixtures.py
"""
from copy import deepcopy
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.eligibility.agent import eligibility_agent_node
from agents.recommend.agent import recommend_agent_node

FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "product_outputs"


def load_product_output() -> str:
    fixture_files = [
        "jangbyeongnaeil.txt",
        "kb_miso_dream.txt",
        "general_regular_savings.txt",
        "discontinued_product.txt",
    ]
    return "\n\n---\n\n".join(
        (FIXTURE_DIR / name).read_text(encoding="utf-8").strip() for name in fixture_files
    )


def make_state(customer_output: str) -> dict:
    return {
        "messages": [],
        "next": "",
        "member_id": None,
        "context": None,
        "plan": [],
        "current_step": 0,
        "agent_outputs": {
            "customer_agent": customer_output.strip(),
            "product_agent": load_product_output(),
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


def run_flow(customer_output: str) -> dict:
    state = make_state(customer_output)
    eligibility_update = eligibility_agent_node(state)
    state = merge_state(state, eligibility_update)
    recommend_update = recommend_agent_node(state)
    state = merge_state(state, recommend_update)
    return state


def find_result(results: list[dict], keyword: str) -> dict:
    for item in results:
        if keyword in item.get("product_name", ""):
            return item
    raise AssertionError(f"Result containing '{keyword}' not found")


def recommended_names(state: dict) -> list[str]:
    return [item.get("product_name", "") for item in (state.get("recommendation_results") or [])]


def test_civil_servant_customer() -> None:
    state = run_flow(
        """
고객명: 공무원고객
나이: 32
직업: 공무원
월 가용 저축액: 300000
급여이체 있음
자동이체 있음
카드 사용 있음
주거래 고객
"""
    )
    results = state["eligibility_results"]
    military = find_result(results, "장병내일적금")
    discontinued = find_result(results, "판매종료적금")
    general = find_result(results, "일반정기적금")

    assert military["eligible"] is False
    assert discontinued["eligible"] is False
    assert general["eligible"] is True

    names = recommended_names(state)
    assert "장병내일적금" not in "".join(names)
    assert "판매종료적금" not in "".join(names)
    assert any("일반정기적금" in name for name in names)


def test_soldier_customer() -> None:
    state = run_flow(
        """
고객명: 군인고객
나이: 25
직업: 직업군인
월 가용 저축액: 250000
급여이체 있음
"""
    )
    military = find_result(state["eligibility_results"], "장병내일적금")
    assert military["eligible"] is True
    assert military["check_required"] == []
    assert any("장병내일적금" in name for name in recommended_names(state))


def test_miso_unknown_customer() -> None:
    state = run_flow(
        """
고객명: 확인필요고객
나이: 29
직업: 일반 직장인
월 가용 저축액: 200000
자동이체 있음
"""
    )
    miso = find_result(state["eligibility_results"], "KB미소드림적금")
    assert "미소드림적금 대상 조건 충족 여부" in miso["check_required"]
    assert not any("KB미소드림적금" in name for name in recommended_names(state))


def test_general_regular_savings_recommended() -> None:
    state = run_flow(
        """
고객명: 직장인고객
나이: 35
직업: 일반 직장인
월 가용 저축액: 400000
급여이체 있음
자동이체 있음
카드 사용 있음
"""
    )
    general = find_result(state["eligibility_results"], "일반정기적금")
    assert general["eligible"] is True
    assert general["check_required"] == []
    assert any("일반정기적금" in name for name in recommended_names(state))


def test_only_recommendable_products_are_ranked() -> None:
    state = run_flow(
        """
고객명: 일반고객
나이: 35
직업: 일반 직장인
월 가용 저축액: 200000
"""
    )
    eligibility_results = state["eligibility_results"]
    recommended = set(recommended_names(state))
    for item in eligibility_results:
        if item["eligible"] is True and not item["check_required"]:
            assert item["product_name"] in recommended
        else:
            assert item["product_name"] not in recommended


def main() -> None:
    test_civil_servant_customer()
    test_soldier_customer()
    test_miso_unknown_customer()
    test_general_regular_savings_recommended()
    test_only_recommendable_products_are_ranked()
    print("ALL FIXTURE TESTS PASSED")


if __name__ == "__main__":
    main()
