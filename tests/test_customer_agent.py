import asyncio
import unittest
from unittest.mock import patch

from agents.customer import agent as customer_agent


class FakeTool:
    def __init__(self, name: str):
        self.name = name
        self.calls = []

    def invoke(self, args: dict):
        self.calls.append(args)
        return f"{self.name} result for {args['customer_name']}"

    async def ainvoke(self, args: dict):
        self.calls.append(args)
        return f"{self.name} result for {args['customer_name']}"


class CustomerAgentTest(unittest.TestCase):
    def test_customer_agent_uses_required_tools_when_customer_id_is_in_state(self):
        fake_tools = [
            FakeTool("get_customer_profile"),
            FakeTool("get_customer_accounts"),
            FakeTool("get_payment_history"),
        ]

        with patch.object(customer_agent, "CUSTOMER_TOOLS", fake_tools):
            result = asyncio.run(
                customer_agent.customer_agent_node(
                    {
                        "customer_id": 98,
                        "messages": [],
                        "agent_outputs": {},
                        "completed_agents": [],
                        "current_step": 0,
                    }
                )
            )

        payload = result["customer_result"]["result"]

        self.assertEqual(
            [tool.calls for tool in fake_tools],
            [
                [{"customer_name": "怨좉컼_098"}],
                [{"customer_name": "怨좉컼_098"}],
                [{"customer_name": "怨좉컼_098"}],
            ],
        )
        self.assertEqual(
            [item["tool_name"] for item in payload["tool_results"]],
            [
                "get_customer_profile",
                "get_customer_accounts",
                "get_payment_history",
            ],
        )
        self.assertIn("怨좉컼_098", payload["summary"])
        self.assertEqual(result["agent_outputs"]["customer_agent"], result["customer_result"])

    def test_customer_agent_extracts_customer_id_from_message_text(self):
        fake_tools = [
            FakeTool("get_customer_profile"),
            FakeTool("get_customer_accounts"),
            FakeTool("get_payment_history"),
        ]

        with patch.object(customer_agent, "CUSTOMER_TOOLS", fake_tools):
            result = asyncio.run(
                customer_agent.customer_agent_node(
                    {
                        "messages": [("user", "怨좉컼 ID 98踰?媛???곹뭹 議고쉶?댁쨾")],
                        "agent_outputs": {},
                        "completed_agents": [],
                        "current_step": 0,
                    }
                )
            )

        self.assertTrue(result["customer_result"]["result"]["tool_results"])
        self.assertEqual(fake_tools[0].calls, [{"customer_name": "怨좉컼_098"}])

    def test_customer_agent_preserves_structured_monthly_saving_amount(self):
        fake_tools = [
            FakeTool("get_customer_profile"),
            FakeTool("get_customer_accounts"),
            FakeTool("get_payment_history"),
        ]

        with patch.object(customer_agent, "CUSTOMER_TOOLS", fake_tools), patch.object(
            customer_agent,
            "_extract_customer_profile",
            return_value={"job": "직장인"},
        ):
            result = asyncio.run(
                customer_agent.customer_agent_node(
                    {
                        "customer_id": 223,
                        "messages": [],
                        "agent_outputs": {},
                        "completed_agents": [],
                        "current_step": 0,
                        "customer_profile": {
                            "customer_id": 223,
                            "monthly_saving_amount": 100000,
                            "available_monthly_saving": 100000,
                            "customer_profile_source": "streamlit_dashboard",
                        },
                    }
                )
            )

        profile = result["customer_profile"]
        payload_profile = result["customer_result"]["result"]["customer_profile"]

        self.assertEqual(profile.get("monthly_saving_amount"), 100000)
        self.assertEqual(profile.get("available_monthly_saving"), 100000)
        self.assertEqual(payload_profile.get("monthly_saving_amount"), 100000)
        self.assertEqual(payload_profile.get("customer_profile_source"), "streamlit_dashboard")


if __name__ == "__main__":
    unittest.main()
