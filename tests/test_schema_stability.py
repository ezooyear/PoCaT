import unittest
from unittest.mock import patch

from agents.base import (
    get_customer_profile,
    get_financial_calculations,
    get_product_candidates,
    get_recommendations,
    get_validation_result,
    make_agent_result,
)
from agents.product import agent as product_agent
from agents.eligibility import agent as eligibility_agent
from agents.financial import agent as financial_agent
from agents.recommend import agent as recommend_agent
from agents.validation import agent as validation_agent


class SchemaStabilityTest(unittest.TestCase):
    def test_product_schema_contains_products_and_structured_counts(self):
        fake_search_text = "KB 스타 적금(30만원 이하 자유적립식)\n기본 금리 연 2.0%\n월 최소 납입 100,000원\n월 최대 납입 300,000원"
        fake_result = {
            "product_result": {
                "status": "success",
                "summary": "제품 후보를 검색했습니다.",
                "result": {
                    "summary": "제품 후보를 검색했습니다.",
                    "tool_results": [
                        {
                            "tool_name": "search_terms",
                            "tool_args": {"query": "적금 조건"},
                            "tool_result": fake_search_text,
                        }
                    ]
                },
                "evidence": [],
                "error": None,
            },
            "agent_outputs": {
                "product_agent": {
                    "status": "success",
                    "summary": "제품 후보를 검색했습니다.",
                    "result": {
                        "summary": "제품 후보를 검색했습니다.",
                        "tool_results": [
                            {
                                "tool_name": "search_terms",
                                "tool_args": {"query": "적금 조건"},
                                "tool_result": fake_search_text,
                            }
                        ]
                    },
                    "evidence": [],
                    "error": None,
                }
            },
            "messages": [],
        }

        with patch.object(product_agent, "run_agent_loop", return_value=fake_result):
            result = product_agent.product_agent_node({"messages": [], "agent_outputs": {}, "completed_agents": [], "current_step": 0})

        product_result = result["product_result"]
        self.assertIn("products", product_result)
        self.assertIsInstance(product_result["products"], list)
        self.assertEqual(product_result["structured_product_count"], len(product_result["products"]))

    def test_eligibility_schema_contains_required_keys(self):
        customer_profile = {
            "customer_id": "customer_223",
            "age": 21,
            "job": "군인",
            "income": 2900,
            "monthly_saving_amount": 100000,
            "transaction_months": 7,
            "salary_transfer": True,
            "auto_transfer": False,
            "card_usage": False,
            "main_bank": False,
            "marketing_agree": False,
            "is_soldier": True,
        }
        state = {
            "messages": [],
            "agent_outputs": {},
            "completed_agents": [],
            "current_step": 0,
            "customer_result": make_agent_result(result={"summary": "고객 정보", "customer_profile": customer_profile}),
            "product_result": make_agent_result(result={"summary": "상품 후보", "products": [{"product_name": "KB 스타 적금(30만원 이하 자유적립식)", "raw_text": "KB 스타 적금\n월 최소 납입 100,000원\n월 최대 납입 300,000원"}]}),
        }

        result = eligibility_agent.eligibility_agent_node(state)
        eligibility_result = result["eligibility_result"]
        self.assertIn("results", eligibility_result)
        self.assertIn("eligible_products", eligibility_result)
        self.assertIn("needs_check_products", eligibility_result)
        self.assertIn("rejected_products", eligibility_result)

    def test_financial_schema_contains_calculations(self):
        fake_result = {
            "financial_result": make_agent_result(result={
                "summary": "수치 계산 결과",
                "tool_results": [
                    {
                        "tool_name": "calculate_interest",
                        "tool_args": {"principal": 1200000, "annual_rate": 2.5, "months": 12, "monthly_payment": 100000},
                        "tool_result": "정기예금 이자 계산 결과 (단리)\n- 만기 예상 수령액: 1,213,747원\n- 세전 이자: 16,250원\n- 세후 이자: 13,747원",
                    }
                ]
            }),
            "agent_outputs": {"financial_agent": {"result": {"summary": "수치 계산 결과", "tool_results": []}}},
        }

        with patch.object(financial_agent, "run_agent_loop", return_value=fake_result):
            result = financial_agent.financial_agent_node({"messages": [], "agent_outputs": {}, "completed_agents": [], "current_step": 0})

        financial_result = result["financial_result"]
        self.assertIn("calculations", financial_result["result"])
        self.assertIsInstance(financial_result["result"]["calculations"], list)

    def test_recommend_schema_contains_recommendations_and_matched_products(self):
        state = {
            "messages": [],
            "agent_outputs": {},
            "completed_agents": [],
            "current_step": 0,
            "eligibility_result": make_agent_result(result={"results": [{"product_name": "KB 스타 적금(30만원 이하 자유적립식)", "eligible": True, "status": "eligible", "reasons": []}] ,"eligible_products":[{"product_name":"KB 스타 적금(30만원 이하 자유적립식)","eligible":True,"status":"eligible"}]}),
            "product_result": make_agent_result(result={"products": [{"product_name": "KB 스타 적금(30만원 이하 자유적립식)", "raw_text": "KB 스타 적금"}]}),
            "financial_result": make_agent_result(result={"calculations": [{"product_name": "KB 스타 적금(30만원 이하 자유적립식)", "estimated_interest_before_tax": 16250, "estimated_maturity_amount": 1213747}]}),
        }

        result = recommend_agent.recommend_agent_node(state)
        recommend_result = result["recommend_result"]
        self.assertIn("recommendations", recommend_result)
        self.assertIn("matched_products", recommend_result)

    def test_validation_schema_contains_required_keys(self):
        state = {
            "messages": [],
            "plan": ["customer_agent", "product_agent", "eligibility_agent", "financial_agent", "recommend_agent", "validation_agent"],
            "completed_agents": ["customer_agent", "product_agent", "eligibility_agent", "financial_agent", "recommend_agent", "validation_agent"],
            "current_step": 6,
            "task_type": "recommendation",
            "agent_outputs": {},
            "customer_result": make_agent_result(result={"summary": "고객 정보", "customer_profile": {"customer_id": "customer_223"}}),
            "product_result": make_agent_result(result={"summary": "상품 후보", "products": []}),
            "eligibility_result": make_agent_result(result={"summary": "eligibility", "results": [], "eligible_products": [], "needs_check_products": [], "rejected_products": []}),
            "financial_result": make_agent_result(result={"summary": "financial", "calculations": []}),
            "recommend_result": make_agent_result(result={"summary": "recommend", "recommendations": [], "matched_products": [], "unmatched_products": []}),
        }

        result = validation_agent.validation_agent_node(state)
        validation_result = result["validation_result"]
        self.assertIn("is_valid", validation_result)
        self.assertIn("failure_reasons", validation_result)
        self.assertIn("warnings", validation_result)
        self.assertIn("checks", validation_result)

    def test_end_to_end_read_paths_work_across_state(self):
        state = {
            "customer_result": make_agent_result(result={"summary": "고객 정보", "customer_profile": {"customer_id": "customer_223"}}),
            "product_result": make_agent_result(result={"summary": "상품 후보", "products": [{"product_name": "KB 스타 적금", "raw_text": "KB 스타 적금"}]}),
            "eligibility_result": make_agent_result(result={"summary": "eligible", "eligible_products": [{"product_name": "KB 스타 적금", "eligible": True}], "results": []}),
            "financial_result": make_agent_result(result={"summary": "financial", "calculations": [{"product_name": "KB 스타 적금", "estimated_maturity_amount": 1213747}]}),
            "recommend_result": make_agent_result(result={"summary": "recommend", "recommendations": [{"product_name": "KB 스타 적금"}]}),
            "validation_result": make_agent_result(result={"summary": "validation", "is_valid": True, "failure_reasons": [], "warnings": [], "checks": {"has_product_result": True}}),
        }

        customer_profile = get_customer_profile(state)
        product_candidates = get_product_candidates(state)
        financial_calculations = get_financial_calculations(state)
        recommendations = get_recommendations(state)
        validation_payload = get_validation_result(state)

        self.assertEqual(customer_profile["source"], "customer_result.customer_profile")
        self.assertEqual(product_candidates["source"], "product_result.products")
        self.assertEqual(financial_calculations["source"], "financial_result.calculations")
        self.assertEqual(recommendations["source"], "recommend_result.recommendations")
        self.assertEqual(validation_payload["source"], "validation_result")
        self.assertIsInstance(customer_profile["data"], dict)
        self.assertIsInstance(product_candidates["data"], list)
        self.assertIsInstance(financial_calculations["data"], list)
        self.assertIsInstance(recommendations["data"], list)
        self.assertIsInstance(validation_payload["data"], dict)


if __name__ == "__main__":
    unittest.main()
