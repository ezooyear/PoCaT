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

from agents.base import make_agent_result
from agents.eligibility.agent import eligibility_agent_node
from agents.financial.agent import _finalize_financial_result_schema
from agents.product.agent import _build_structured_product_result
from agents.recommend.agent import recommend_agent_node
from agents.validation.agent import validation_agent_node


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
    payload = eligibility_update.get("eligibility_result", {}).get("result", {})

    assert results, "eligibility_results should not be empty"
    assert all(item.get("status") != "eligible" for item in results)
    assert payload.get("customer_profile_source") == "parsed_summary"
    assert "customer_profile_incomplete" in str(payload.get("fallback_reason"))
    assert "고객 나이 정보" in (payload.get("missing_fields") or [])


def test_summary_profile_fields_are_parsed_without_missing_core_fields() -> None:
    state = make_base_state()
    state["customer_profile"] = None
    state["agent_outputs"]["customer_agent"] = """
고객명: 고객_223
현재 연령: 21세
직업: 군인
연간 소득: 2,900만원
월 가용 저축액: 100,000원
급여이체 여부: 예
자동이체 여부: 아니오
카드 사용 여부: 아니오
""".strip()

    eligibility_update = eligibility_agent_node(state)
    payload = eligibility_update.get("eligibility_result", {}).get("result", {})
    customer_profile = payload.get("customer_profile", {})
    missing_fields = payload.get("missing_fields") or []

    assert payload.get("customer_profile_source") == "parsed_summary"
    assert customer_profile.get("age") == 21
    assert customer_profile.get("income") == 29000000
    assert customer_profile.get("monthly_saving_amount") == 100000
    assert customer_profile.get("job") == "군인"
    assert customer_profile.get("salary_transfer") is True
    assert customer_profile.get("auto_transfer") is False
    assert customer_profile.get("card_usage") is False
    assert "고객 나이 정보" not in missing_fields
    assert "고객 월 가용 저축액" not in missing_fields


def test_customer_result_summary_maps_customer_fields_without_missing() -> None:
    state = make_base_state()
    state["agent_outputs"]["customer_agent"] = {}
    state["customer_result"] = {
        "result": {
            "summary": """
고객명: 고객_223
현재 연령: 21세
직업: 군인
연간 소득: 2,900만원
월 가용 저축액: 100,000원
급여이체 여부: 예
자동이체 여부: 아니오
카드 사용 여부: 아니오
""".strip()
        }
    }

    eligibility_update = eligibility_agent_node(state)
    payload = eligibility_update.get("eligibility_result", {}).get("result", {})
    customer_profile = payload.get("customer_profile", {})
    missing_fields = payload.get("missing_fields") or []

    assert payload.get("customer_profile_source") == "customer_result"
    assert customer_profile.get("age") == 21
    assert customer_profile.get("income") == 29000000
    assert customer_profile.get("monthly_saving_amount") == 100000
    assert customer_profile.get("salary_transfer") is True
    assert customer_profile.get("auto_transfer") is False
    assert "age" in (payload.get("parsed_customer_fields") or [])
    assert "income" in (payload.get("parsed_customer_fields") or [])
    assert "monthly_saving_amount" in (payload.get("parsed_customer_fields") or [])
    assert "salary_transfer" in (payload.get("parsed_customer_fields") or [])
    assert "auto_transfer" in (payload.get("parsed_customer_fields") or [])
    assert "age" not in missing_fields
    assert "monthly_saving_amount" not in missing_fields
    assert "income" not in missing_fields
    assert "salary_transfer" not in missing_fields
    assert "auto_transfer" not in missing_fields


def test_birth_date_and_age_line_parses_age() -> None:
    state = make_base_state()
    state["agent_outputs"]["customer_agent"] = "\uc0dd\ub144\uc6d4\uc77c / \uc5f0\ub839: 2004-08-20 (22\uc138)"

    eligibility_update = eligibility_agent_node(state)
    payload = eligibility_update.get("eligibility_result", {}).get("result", {})
    customer_profile = payload.get("customer_profile", {})

    assert customer_profile.get("age") == 22
    assert payload.get("customer_profile_source") == "parsed_summary"
    assert payload.get("parsed_customer_values", {}).get("age") == 22


def test_simple_age_line_parses_age() -> None:
    state = make_base_state()
    state["agent_outputs"]["customer_agent"] = "연령: 22세"

    eligibility_update = eligibility_agent_node(state)
    payload = eligibility_update.get("eligibility_result", {}).get("result", {})
    customer_profile = payload.get("customer_profile", {})

    assert customer_profile.get("age") == 22
    assert payload.get("parsed_customer_values", {}).get("age") == 22
    assert "age" not in (payload.get("missing_fields") or [])


def test_birth_date_with_current_age_parses_age() -> None:
    state = make_base_state()
    state["agent_outputs"]["customer_agent"] = "생년월일: 2004-08-20 (현재 연령: 22세)"

    eligibility_update = eligibility_agent_node(state)
    payload = eligibility_update.get("eligibility_result", {}).get("result", {})
    customer_profile = payload.get("customer_profile", {})

    assert customer_profile.get("age") == 22
    assert payload.get("parsed_customer_values", {}).get("age") == 22


def test_birth_date_only_computes_age() -> None:
    state = make_base_state()
    state["agent_outputs"]["customer_agent"] = "생년월일: 2004-08-20"

    eligibility_update = eligibility_agent_node(state)
    payload = eligibility_update.get("eligibility_result", {}).get("result", {})
    customer_profile = payload.get("customer_profile", {})

    assert customer_profile.get("age") is not None
    assert payload.get("parsed_customer_values", {}).get("age") == customer_profile.get("age")


def test_monthly_saving_amount_parses_won_value() -> None:
    state = make_base_state()
    state["agent_outputs"]["customer_agent"] = "월 가용 저축액: 100,000원"

    eligibility_update = eligibility_agent_node(state)
    payload = eligibility_update.get("eligibility_result", {}).get("result", {})
    customer_profile = payload.get("customer_profile", {})

    assert customer_profile.get("monthly_saving_amount") == 100000
    assert payload.get("parsed_customer_values", {}).get("monthly_saving_amount") == 100000


def test_monthly_saving_amount_parses_bold_won_value() -> None:
    state = make_base_state()
    state["agent_outputs"]["customer_agent"] = "월 가용 저축액: **100,000원**"

    eligibility_update = eligibility_agent_node(state)
    payload = eligibility_update.get("eligibility_result", {}).get("result", {})
    customer_profile = payload.get("customer_profile", {})

    assert customer_profile.get("monthly_saving_amount") == 100000
    assert payload.get("parsed_customer_values", {}).get("monthly_saving_amount") == 100000


def test_monthly_saving_amount_parses_manwon_value() -> None:
    state = make_base_state()
    state["agent_outputs"]["customer_agent"] = "월 저축액: 10만원"

    eligibility_update = eligibility_agent_node(state)
    payload = eligibility_update.get("eligibility_result", {}).get("result", {})
    customer_profile = payload.get("customer_profile", {})

    assert customer_profile.get("monthly_saving_amount") == 100000
    assert payload.get("parsed_customer_values", {}).get("monthly_saving_amount") == 100000


def test_monthly_saving_amount_parses_spaced_manwon_value() -> None:
    state = make_base_state()
    state["agent_outputs"]["customer_agent"] = "월 저축 가능액: 10 만원"

    eligibility_update = eligibility_agent_node(state)
    payload = eligibility_update.get("eligibility_result", {}).get("result", {})
    customer_profile = payload.get("customer_profile", {})

    assert customer_profile.get("monthly_saving_amount") == 100000
    assert payload.get("parsed_customer_values", {}).get("monthly_saving_amount") == 100000


def test_income_parses_manwon_value() -> None:
    state = make_base_state()
    state["agent_outputs"]["customer_agent"] = "연간 소득: 2,900만원"

    eligibility_update = eligibility_agent_node(state)
    payload = eligibility_update.get("eligibility_result", {}).get("result", {})
    customer_profile = payload.get("customer_profile", {})

    assert customer_profile.get("income") == 29000000
    assert payload.get("parsed_customer_values", {}).get("income") == 29000000


def test_income_parses_spaced_manwon_value() -> None:
    state = make_base_state()
    state["agent_outputs"]["customer_agent"] = "연간 소득: 2,900 만원"

    eligibility_update = eligibility_agent_node(state)
    payload = eligibility_update.get("eligibility_result", {}).get("result", {})
    customer_profile = payload.get("customer_profile", {})

    assert customer_profile.get("income") == 29000000
    assert payload.get("parsed_customer_values", {}).get("income") == 29000000


def test_income_parses_bold_manwon_value() -> None:
    state = make_base_state()
    state["agent_outputs"]["customer_agent"] = "연간 소득: **2,900만원**"

    eligibility_update = eligibility_agent_node(state)
    payload = eligibility_update.get("eligibility_result", {}).get("result", {})
    customer_profile = payload.get("customer_profile", {})

    assert customer_profile.get("income") == 29000000
    assert payload.get("parsed_customer_values", {}).get("income") == 29000000


def test_structured_customer_profile_overrides_summary_value() -> None:
    state = make_base_state()
    state["customer_profile"] = {
        "age": 25,
        "job": "군인",
        "income": 50000000,
        "monthly_saving_amount": 250000,
        "salary_transfer": True,
        "auto_transfer": False,
        "card_usage": False,
        "raw_text": "구조화 프로필",
    }
    state["agent_outputs"]["customer_agent"] = """
현재 연령: 21세
연간 소득: 2,900만원
월 가용 저축액: 100,000원
""".strip()

    eligibility_update = eligibility_agent_node(state)
    payload = eligibility_update.get("eligibility_result", {}).get("result", {})
    customer_profile = payload.get("customer_profile", {})

    assert payload.get("customer_profile_source") == "structured_state"
    assert customer_profile.get("age") == 25
    assert customer_profile.get("income") == 50000000
    assert customer_profile.get("monthly_saving_amount") == 250000


def test_job_markdown_prefix_is_cleaned() -> None:
    state = make_base_state()
    state["agent_outputs"]["customer_agent"] = "\uc9c1\uc5c5: ** \uad70\uc778"

    eligibility_update = eligibility_agent_node(state)
    payload = eligibility_update.get("eligibility_result", {}).get("result", {})
    customer_profile = payload.get("customer_profile", {})

    assert customer_profile.get("job") == "군인"
    assert payload.get("parsed_customer_values", {}).get("job") == "군인"


def test_no_values_are_treated_as_false_not_missing() -> None:
    state = make_base_state()
    state["agent_outputs"]["customer_agent"] = """
고객명: 고객_223
현재 연령: 21세
직업: 군인
연간 소득: 2,900만원
월 가용 저축액: 100,000원
급여이체 여부: 예
자동이체 여부: 아니오
카드 사용 여부: 아니오
""".strip()

    eligibility_update = eligibility_agent_node(state)
    payload = eligibility_update.get("eligibility_result", {}).get("result", {})
    customer_profile = payload.get("customer_profile", {})
    missing_fields = payload.get("missing_fields") or []

    assert customer_profile.get("auto_transfer") is False
    assert customer_profile.get("card_usage") is False
    assert "auto_transfer" not in missing_fields
    assert "card_usage" not in missing_fields


def test_customer_profile_incomplete_when_customer_info_is_really_missing() -> None:
    state = make_base_state()
    state["customer_profile"] = None
    state["agent_outputs"]["customer_agent"] = "고객명: 고객_223"

    eligibility_update = eligibility_agent_node(state)
    payload = eligibility_update.get("eligibility_result", {}).get("result", {})

    assert "customer_profile_incomplete" in str(payload.get("fallback_reason"))
    assert "고객 나이 정보" in (payload.get("missing_fields") or [])
    assert "고객 월 가용 저축액" in (payload.get("missing_fields") or [])


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


def test_product_schema_contains_products() -> None:
    base_product_result = make_agent_result(
        status="success",
        result={
            "summary": "추천 후보 상품을 조회했습니다.",
            "tool_results": [],
        },
        evidence=[],
        error=None,
    )
    products = [
        {
            "product_name": "KB 테스트 적금",
            "product_type": "적금",
            "base_rate": 2.0,
            "max_rate": 2.5,
        }
    ]
    product_result = _build_structured_product_result(
        base_product_result,
        summary="추천 후보 상품을 조회했습니다.",
        products=products,
        product_candidates=products,
        performance={"tool_count": 1},
    )

    assert "products" in product_result
    assert isinstance(product_result["products"], list)
    assert product_result["structured_product_count"] == len(product_result["products"])


def test_eligibility_schema_contains_grouped_keys() -> None:
    state = make_base_state()
    eligibility_update = eligibility_agent_node(state)
    eligibility_result = eligibility_update.get("eligibility_result", {})

    assert "results" in eligibility_result
    assert "eligible_products" in eligibility_result
    assert "needs_check_products" in eligibility_result
    assert "rejected_products" in eligibility_result


def test_financial_schema_contains_calculations() -> None:
    result = {
        "financial_result": make_agent_result(
            status="success",
            result={
                "summary": "상품별 예상 이자와 만기금액을 계산했습니다.",
                "tool_results": [],
                "calculations": [
                    {
                        "product_name": "KB 테스트 적금",
                        "product_type": "적금",
                        "monthly_amount": 100000,
                        "term_months": 12,
                        "payment_count": 12,
                        "applied_rate": 2.5,
                        "principal": 1200000,
                        "estimated_interest_before_tax": 10000,
                        "estimated_maturity_amount": 1210000,
                    }
                ],
            },
            evidence=[],
            error=None,
        ),
        "agent_outputs": {"financial_agent": {}},
    }

    finalized = _finalize_financial_result_schema(result)
    financial_result = finalized.get("financial_result", {})

    assert "calculations" in financial_result
    assert isinstance(financial_result["calculations"], list)


def test_recommend_schema_contains_recommendations_and_matches() -> None:
    state = make_base_state()
    eligibility_update = eligibility_agent_node(state)
    state = merge_state(state, eligibility_update)
    state["financial_results"] = [
        {
            "product_name": "일반정기적금",
            "estimated_interest": 180000,
            "maturity_amount": 7380000,
        }
    ]

    recommend_update = recommend_agent_node(state)
    recommend_result = recommend_update.get("recommend_result", {})

    assert "recommendations" in recommend_result
    assert "matched_products" in recommend_result


def test_validation_schema_contains_expected_keys() -> None:
    state = {
        "task_type": "product_info",
        "plan": ["product_agent", "validation_agent"],
        "completed_agents": ["product_agent"],
        "current_step": 1,
        "agent_outputs": {},
        "product_result": {
            "status": "success",
            "summary": "추천 후보 상품을 조회했습니다.",
            "result": {
                "summary": "추천 후보 상품을 조회했습니다.",
                "products": [{"product_name": "KB 테스트 적금"}],
            },
            "products": [{"product_name": "KB 테스트 적금"}],
            "evidence": [],
            "error": None,
        },
    }

    validation_update = validation_agent_node(state)
    validation_result = validation_update.get("validation_result", {})

    assert "is_valid" in validation_result
    assert "failure_reasons" in validation_result
    assert "warnings" in validation_result
    assert "checks" in validation_result


def test_end_to_end_schema_paths_exist() -> None:
    state = make_base_state()
    state["customer_result"] = {
        "status": "success",
        "summary": "고객 정보를 조회했습니다.",
        "result": {
            "summary": "고객 정보를 조회했습니다.",
            "customer_profile": {
                "customer_id": "customer_223",
                "age": 21,
                "job": "군인",
                "income": 29000000,
                "monthly_saving_amount": 100000,
                "salary_transfer": True,
                "auto_transfer": False,
            },
        },
        "customer_profile": {
            "customer_id": "customer_223",
            "age": 21,
            "job": "군인",
            "income": 29000000,
            "monthly_saving_amount": 100000,
            "salary_transfer": True,
            "auto_transfer": False,
        },
        "evidence": [],
        "error": None,
    }
    state["product_result"] = {
        "status": "success",
        "summary": "추천 후보 상품을 조회했습니다.",
        "result": {
            "summary": "추천 후보 상품을 조회했습니다.",
            "products": [{"product_name": "일반정기적금", "product_type": "적금"}],
        },
        "products": [{"product_name": "일반정기적금", "product_type": "적금"}],
        "evidence": [],
        "error": None,
    }

    eligibility_update = eligibility_agent_node(state)
    state = merge_state(state, eligibility_update)
    state["financial_result"] = {
        "status": "success",
        "summary": "상품별 예상 이자와 만기금액을 계산했습니다.",
        "result": {
            "summary": "상품별 예상 이자와 만기금액을 계산했습니다.",
            "calculations": [
                {
                    "product_name": "일반정기적금",
                    "product_type": "적금",
                    "monthly_amount": 100000,
                    "term_months": 24,
                    "payment_count": 24,
                    "applied_rate": 2.5,
                    "principal": 2400000,
                    "estimated_interest_before_tax": 50000,
                    "estimated_maturity_amount": 2450000,
                }
            ],
        },
        "calculations": [
            {
                "product_name": "일반정기적금",
                "product_type": "적금",
                "monthly_amount": 100000,
                "term_months": 24,
                "payment_count": 24,
                "applied_rate": 2.5,
                "principal": 2400000,
                "estimated_interest_before_tax": 50000,
                "estimated_maturity_amount": 2450000,
            }
        ],
        "evidence": [],
        "error": None,
    }
    state["financial_results"] = state["financial_result"]["calculations"]

    recommend_update = recommend_agent_node(state)
    state = merge_state(state, recommend_update)
    validation_update = validation_agent_node(state)
    state = merge_state(state, validation_update)

    assert state["customer_result"]["customer_profile"]
    assert state["product_result"]["products"]
    assert state["eligibility_result"]["eligible_products"] is not None
    assert state["financial_result"]["calculations"] is not None
    assert state["recommend_result"]["recommendations"] is not None
    assert state["validation_result"] is not None


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
    test_summary_profile_fields_are_parsed_without_missing_core_fields()
    test_customer_result_summary_maps_customer_fields_without_missing()
    test_birth_date_and_age_line_parses_age()
    test_simple_age_line_parses_age()
    test_birth_date_with_current_age_parses_age()
    test_birth_date_only_computes_age()
    test_monthly_saving_amount_parses_won_value()
    test_monthly_saving_amount_parses_bold_won_value()
    test_monthly_saving_amount_parses_manwon_value()
    test_monthly_saving_amount_parses_spaced_manwon_value()
    test_income_parses_manwon_value()
    test_income_parses_spaced_manwon_value()
    test_income_parses_bold_manwon_value()
    test_structured_customer_profile_overrides_summary_value()
    test_job_markdown_prefix_is_cleaned()
    test_no_values_are_treated_as_false_not_missing()
    test_customer_profile_incomplete_when_customer_info_is_really_missing()
    test_invalid_product_name_becomes_invalid_product()
    test_product_schema_contains_products()
    test_eligibility_schema_contains_grouped_keys()
    test_financial_schema_contains_calculations()
    test_recommend_schema_contains_recommendations_and_matches()
    test_validation_schema_contains_expected_keys()
    test_end_to_end_schema_paths_exist()
    test_financial_results_missing_defers_recommendation()
    test_no_eligible_product_does_not_force_recommendation()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
