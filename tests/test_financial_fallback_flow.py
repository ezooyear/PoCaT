import asyncio
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from agents.base import make_agent_result
from agents.financial import agent as financial_agent
from agents.validation import agent as validation_agent


class FinancialFallbackFlowTest(unittest.TestCase):
    def test_financial_agent_marks_outer_observation_as_fallback_success(self):
        state = {
            "task_type": "recommendation",
            "user_query": "가입 가능한 적금 중 월 30만원 기준으로 추천해줘",
            "customer_profile": {
                "customer_id": "customer_223",
                "monthly_saving_amount": 300000,
            },
            "product_candidates": [
                {
                    "product_name": "KB Star 적금",
                    "product_type": "적금",
                    "base_rate": 2.5,
                    "max_rate": 3.2,
                    "min_period_months": 12,
                    "max_period_months": 12,
                    "max_monthly_amount": 300000,
                }
            ],
            "eligibility_result": make_agent_result(
                result={
                    "results": [
                        {
                            "product_name": "KB Star 적금",
                            "eligible": True,
                            "status": "eligible",
                        }
                    ]
                }
            ),
            "agent_outputs": {},
            "completed_agents": ["customer_agent", "product_agent", "eligibility_agent"],
            "current_step": 3,
        }

        class FakeObservation:
            def __init__(self):
                self.updates = []

            def update(self, **kwargs):
                self.updates.append(kwargs)

        observations = []

        @contextmanager
        def fake_langfuse_observation(**kwargs):
            observation = FakeObservation()
            observations.append({"kwargs": kwargs, "observation": observation})
            yield observation

        async def raise_timeout(*args, **kwargs):
            raise financial_agent.LLMInvocationTimeoutError("LLM ainvoke timeout after 20.0 seconds")

        with patch.object(financial_agent, "langfuse_observation", side_effect=fake_langfuse_observation), patch.object(financial_agent, "run_agent_loop_async", side_effect=raise_timeout):
            result = asyncio.run(financial_agent.financial_agent_node(state))

        self.assertEqual(result["financial_result"]["result"]["status"], "fallback_success")
        self.assertEqual(len(observations), 1)
        update_payload = observations[0]["observation"].updates[-1]
        self.assertEqual(update_payload["metadata"]["fallback_used"], True)
        self.assertEqual(update_payload["metadata"]["fallback_success"], True)
        self.assertEqual(update_payload["metadata"]["llm_error_type"], "LLMInvocationTimeoutError")
        self.assertEqual(update_payload["metadata"]["financial_results_count"], 1)
        self.assertEqual(update_payload["metadata"]["status"], "fallback_success")
        self.assertEqual(update_payload["status_message"], "financial_fallback_success")

    def test_financial_agent_builds_rule_based_fallback_on_llm_failure(self):
        state = {
            "task_type": "recommendation",
            "user_query": "가입 가능한 예적금 중 월 30만원 기준 추천해줘",
            "customer_profile": {
                "customer_id": "customer_223",
                "monthly_saving_amount": 300000,
            },
            "product_candidates": [
                {
                    "product_name": "KB Star 적금",
                    "product_type": "적금",
                    "base_rate": 2.5,
                    "max_rate": 3.2,
                    "min_period_months": 12,
                    "max_period_months": 12,
                    "max_monthly_amount": 300000,
                }
            ],
            "eligibility_result": make_agent_result(
                result={
                    "results": [
                        {
                            "product_name": "KB Star 적금",
                            "eligible": True,
                            "status": "eligible",
                            "reasons": [],
                            "bonus_conditions_met": ["salary_transfer"],
                            "bonus_conditions_missing": [],
                        }
                    ]
                }
            ),
            "agent_outputs": {},
            "completed_agents": ["customer_agent", "product_agent", "eligibility_agent"],
            "current_step": 3,
        }

        async def raise_timeout(*args, **kwargs):
            raise TimeoutError("LLM ainvoke timeout after 20.0 seconds")

        with patch("builtins.print") as mock_print, patch.object(financial_agent, "run_agent_loop_async", side_effect=raise_timeout):
            result = asyncio.run(financial_agent.financial_agent_node(state))

        payload = result["financial_result"]["result"]
        self.assertEqual(payload["status"], "fallback_success")
        self.assertEqual(payload["calculated_monthly_saving"], 300000)
        self.assertIn("KB Star 적금", payload["affordable_products"])
        self.assertTrue(payload["fallback_applied"])
        self.assertIn("expected_interest_summary", payload)
        self.assertTrue(result["errors"][0]["recoverable"])
        self.assertTrue(any("resolved_model=" in str(call) for call in mock_print.call_args_list))

    def test_validation_treats_recoverable_financial_fallback_as_valid(self):
        state = {
            "messages": [],
            "plan": ["customer_agent", "product_agent", "eligibility_agent", "financial_agent", "recommend_agent", "validation_agent"],
            "completed_agents": ["customer_agent", "product_agent", "eligibility_agent", "financial_agent", "recommend_agent"],
            "current_step": 5,
            "task_type": "recommendation",
            "agent_outputs": {},
            "customer_result": make_agent_result(result={"summary": "customer", "customer_profile": {"customer_id": "customer_223"}}),
            "product_result": make_agent_result(result={"summary": "product", "products": [{"product_name": "KB Star 적금"}]}),
            "eligibility_result": make_agent_result(result={"summary": "eligibility", "results": [{"product_name": "KB Star 적금", "eligible": True, "status": "eligible"}], "eligible_products": [{"product_name": "KB Star 적금", "eligible": True, "status": "eligible"}], "needs_check_products": [], "rejected_products": []}),
            "financial_result": make_agent_result(result={"summary": "financial", "status": "fallback_success", "calculations": [{"product_name": "KB Star 적금", "status": "calculated", "estimated_maturity_amount": 1213747, "estimated_interest": 13747}], "financial_results": [{"product_name": "KB Star 적금", "status": "calculated", "estimated_maturity_amount": 1213747, "estimated_interest": 13747}], "fallback_applied": True}),
            "recommend_result": make_agent_result(result={"summary": "recommend", "recommendations": [{"product_name": "KB Star 적금"}], "matched_products": [{"product_name": "KB Star 적금"}], "unmatched_products": []}),
            "errors": [{"agent": "financial_agent", "error": "Provider returned error", "recoverable": True, "user_visible": False, "fallback_applied": True}],
        }

        with patch.object(validation_agent, "_should_run_llm_validation", return_value=False):
            result = asyncio.run(validation_agent.validation_agent_node(state))
        validation_result = result["validation_result"]["result"]

        self.assertTrue(validation_result["is_valid"])
        self.assertNotIn("복구되지 않은 오류", " ".join(validation_result.get("failure_reasons") or []))


if __name__ == "__main__":
    unittest.main()
