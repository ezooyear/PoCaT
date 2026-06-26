import unittest

from agents.base import make_agent_result
from agents.supervisor import agent as supervisor_agent
from agents.validation import tools as validation_tools


class FinalRecommendationRecoveryTest(unittest.TestCase):
    def _build_state(self):
        return {
            "task_type": "recommendation",
            "customer_result": make_agent_result(
                status="failed",
                result={
                    "status": "success",
                    "summary": "customer loaded",
                    "customer_profile": {"customer_id": "customer_223", "monthly_saving_amount": 300000},
                },
                error="Provider returned error",
            ),
            "eligibility_result": make_agent_result(
                result={
                    "results": [
                        {"product_name": "KB Star 적금", "eligible": True, "status": "eligible"},
                    ],
                    "eligible_products": [
                        {"product_name": "KB Star 적금", "eligible": True, "status": "eligible"},
                    ],
                }
            ),
            "financial_result": make_agent_result(
                result={
                    "status": "fallback_success",
                    "summary": "financial recovered",
                    "financial_results": [
                        {
                            "product_name": "KB Star 적금",
                            "status": "calculated",
                            "estimated_interest": 13747,
                            "estimated_maturity_amount": 1213747,
                        }
                    ],
                    "calculations": [
                        {
                            "product_name": "KB Star 적금",
                            "status": "calculated",
                            "estimated_interest": 13747,
                            "estimated_maturity_amount": 1213747,
                        }
                    ],
                }
            ),
            "recommend_result": make_agent_result(
                result={
                    "status": "recommended",
                    "summary": "1위 KB Star 적금 - 예상 세후 이자 13,747원, 예상 만기금액 1,213,747원",
                    "recommendations": [
                        {
                            "product_name": "KB Star 적금",
                            "rank": 1,
                            "estimated_interest_after_tax": 13747,
                            "estimated_maturity_amount": 1213747,
                            "reason": "가입 가능하고 계산 결과가 가장 안정적입니다.",
                        }
                    ],
                }
            ),
            "errors": [
                {
                    "agent": "financial_agent",
                    "error": "LLM ainvoke timeout after 20.0 seconds",
                    "recoverable": True,
                    "user_visible": False,
                    "fallback_applied": True,
                }
            ],
        }

    def test_validation_tools_treat_recovered_state_as_actionable(self):
        state = self._build_state()
        self.assertTrue(validation_tools.has_actionable_recommendation_state(state))

    def test_supervisor_does_not_block_recovered_recommendation(self):
        state = self._build_state()
        final_context = {
            **state,
            "validation_result": make_agent_result(
                status="failed",
                result={
                    "is_valid": False,
                    "revision_required": True,
                    "summary": "old failure",
                    "blocking_issues": ["customer_result status is failed."],
                },
            ),
        }

        self.assertFalse(supervisor_agent._should_block_final_recommendation(final_context))

    def test_supervisor_fallback_answer_prefers_current_recommendation(self):
        state = self._build_state()
        final_context = {
            **state,
            "validation_result": make_agent_result(
                status="failed",
                result={
                    "is_valid": False,
                    "revision_required": True,
                    "summary": "old failure",
                },
            ),
        }

        answer = supervisor_agent._fallback_final_answer(final_context, error=RuntimeError("Provider returned error"))
        self.assertIn("KB Star 적금", answer)
        self.assertNotIn("Provider returned error", answer)
        self.assertNotIn("timeout", answer.lower())

    def test_validation_retry_update_does_not_restart_financial_when_state_is_actionable(self):
        state = {
            **self._build_state(),
            "user_query": "추천해줘",
            "retry_history": [],
            "validation_retry_count": 0,
            "max_validation_retries": 1,
        }

        retry_update = __import__("agents.validation.agent", fromlist=["_build_validation_retry_update"])._build_validation_retry_update(
            state=state,
            verify_result={"is_valid": False, "revision_required": True, "issues": []},
            validation_summary={
                "awaiting_user_input": False,
                "failure_type": "agent_output_error",
                "blocking_issues": ["과거 오류"],
            },
        )

        self.assertIsNone(retry_update["retry_start_agent"])
        self.assertEqual(retry_update["validation_retry_count"], 0)


if __name__ == "__main__":
    unittest.main()
