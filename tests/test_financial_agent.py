import unittest
from unittest.mock import patch

from agents.financial import agent as financial_agent


class FinancialAgentRoleBoundaryTest(unittest.TestCase):
    def test_financial_agent_estimates_active_account_maturity_when_data_is_available(self):
        accounts_text = "\n".join(
            [
                "account_number | product_name | product_type | maturity_date | monthly_amount | current_balance | applied_rate | account_status",
                "---------------",
                "280-910044-00596 | 일반정기적금 | 적금 | 2026-11-15 | 219256 | 4385120 | 2.27 | ACTIVE",
            ]
        )
        fake_result = {
            "financial_result": {
                "status": "success",
                "result": {
                    "summary": "정확한 만기 예상 수령액 계산을 위해 추가 정보가 필요합니다.",
                    "tool_results": [],
                },
                "evidence": [],
                "error": None,
            },
            "agent_outputs": {"financial_agent": {"result": {"summary": ""}}},
        }
        state = {
            "user_query": "현재 진행 중인 계좌 만기 예상 수령액을 계산해줘.",
            "customer_result": {
                "result": {
                    "tool_results": [
                        {
                            "tool_name": "get_customer_accounts",
                            "tool_result": accounts_text,
                        }
                    ]
                }
            },
        }

        with patch.object(financial_agent, "run_agent_loop", return_value=fake_result):
            result = financial_agent.financial_agent_node(state)

        payload = result["financial_result"]["result"]
        tool_results = payload["tool_results"]

        self.assertTrue(
            any(item["tool_name"] == "estimate_active_account_maturity" for item in tool_results)
        )
        self.assertIn("만기 예상 수령액", payload["summary"])
        self.assertIn("4,385,120원", payload["summary"])
        self.assertIn("219,256원", payload["summary"])

    def test_financial_agent_estimates_maturity_interest_query(self):
        accounts_text = "\n".join(
            [
                "account_number | product_name | product_type | maturity_date | monthly_amount | current_balance | applied_rate | account_status",
                "---------------",
                "280-910044-00596 | 일반정기적금 | 적금 | 2026-11-15 | 219256 | 4385120 | 2.27 | ACTIVE",
            ]
        )
        fake_result = {
            "financial_result": {
                "status": "success",
                "result": {
                    "summary": "현재 시스템에서는 정확한 만기 이자 금액을 직접 계산할 수 없습니다.",
                    "tool_results": [],
                },
                "evidence": [],
                "error": None,
            },
            "agent_outputs": {"financial_agent": {"result": {"summary": ""}}},
        }
        state = {
            "user_query": "만기 시 이자를 알려줘",
            "customer_result": {
                "result": {
                    "tool_results": [
                        {
                            "tool_name": "get_customer_accounts",
                            "tool_result": accounts_text,
                        }
                    ]
                }
            },
        }

        with patch.object(financial_agent, "run_agent_loop", return_value=fake_result):
            result = financial_agent.financial_agent_node(state)

        payload = result["financial_result"]["result"]

        self.assertIn("세전 예상 이자", payload["summary"])
        self.assertIn("세후 예상 이자", payload["summary"])
        self.assertTrue(
            any(item["tool_name"] == "estimate_active_account_maturity" for item in payload["tool_results"])
        )

    def test_financial_agent_replaces_latex_maturity_formula(self):
        accounts_text = "\n".join(
            [
                "account_number | product_name | product_type | maturity_date | monthly_amount | current_balance | applied_rate | account_status",
                "---------------",
                "280-910044-00596 | 일반정기적금 | 적금 | 2026-11-15 | 219256 | 4385120 | 2.27 | ACTIVE",
            ]
        )
        latex_summary = (
            r"[ \text{예상 이자} = \text{잔액} \times \text{연이율} "
            r"\times \frac{\text{남은 일수}}{365} ]"
        )
        fake_result = {
            "financial_result": {
                "status": "success",
                "result": {
                    "summary": latex_summary,
                    "tool_results": [],
                },
                "evidence": [],
                "error": None,
            },
            "agent_outputs": {"financial_agent": {"result": {"summary": latex_summary}}},
        }
        state = {
            "user_query": "만기 시 이자를 알려줘",
            "customer_result": {
                "result": {
                    "tool_results": [
                        {
                            "tool_name": "get_customer_accounts",
                            "tool_result": accounts_text,
                        }
                    ]
                }
            },
        }

        with patch.object(financial_agent, "run_agent_loop", return_value=fake_result):
            result = financial_agent.financial_agent_node(state)

        summary = result["financial_result"]["result"]["summary"]

        self.assertNotIn(r"\text", summary)
        self.assertNotIn(r"\times", summary)
        self.assertNotIn(r"\frac", summary)
        self.assertIn("세전 예상 이자", summary)
        self.assertIn("만기 예상 수령액", summary)

    def test_financial_agent_removes_recommendation_claims(self):
        fake_tool_result = "비교 결과\n- 일반정기예금 만기 예상 수령액: 10,300,000원\n- KBStar정기예금 만기 예상 수령액: 10,250,000원"
        fake_result = {
            "financial_result": {
                "status": "success",
                "result": {
                    "summary": "가장 잘 맞는 상품은 일반정기예금입니다.",
                    "tool_results": [
                        {
                            "tool_name": "compare_products",
                            "tool_args": {"products_info": "..."},
                            "tool_result": fake_tool_result,
                        }
                    ],
                },
                "evidence": [],
                "error": None,
            },
            "agent_outputs": {
                "financial_agent": {
                    "status": "success",
                    "result": {
                        "summary": "가장 잘 맞는 상품은 일반정기예금입니다.",
                        "tool_results": [],
                    },
                    "evidence": [],
                    "error": None,
                }
            },
        }

        with patch.object(financial_agent, "run_agent_loop", return_value=fake_result):
            result = financial_agent.financial_agent_node({"messages": []})

        payload = result["financial_result"]["result"]
        summary = payload["summary"]

        self.assertTrue(payload["role_boundary_enforced"])
        self.assertNotIn("가장 잘 맞는 상품", summary)
        self.assertNotIn("추천합니다", summary)
        self.assertIn(fake_tool_result, summary)
        self.assertEqual(result["agent_outputs"]["financial_agent"]["result"], payload)

    def test_financial_agent_keeps_calculation_only_summary(self):
        fake_summary = "금액 기준으로 A가 B보다 50,000원 높게 계산됩니다."
        fake_result = {
            "financial_result": {
                "status": "success",
                "result": {
                    "summary": fake_summary,
                    "tool_results": [],
                },
                "evidence": [],
                "error": None,
            },
            "agent_outputs": {"financial_agent": {"result": {"summary": fake_summary}}},
        }

        with patch.object(financial_agent, "run_agent_loop", return_value=fake_result):
            result = financial_agent.financial_agent_node({"messages": []})

        payload = result["financial_result"]["result"]

        self.assertEqual(payload["summary"], fake_summary)
        self.assertNotIn("role_boundary_enforced", payload)


if __name__ == "__main__":
    unittest.main()
