import unittest
import sys
import types
from unittest.mock import patch

mcp_module = types.ModuleType("mcp")
mcp_module.ClientSession = object
mcp_module.types = types.SimpleNamespace()
client_module = types.ModuleType("mcp.client")
streamable_http_module = types.ModuleType("mcp.client.streamable_http")
streamable_http_module.streamable_http_client = object()

sys.modules.setdefault("mcp", mcp_module)
sys.modules.setdefault("mcp.client", client_module)
sys.modules.setdefault("mcp.client.streamable_http", streamable_http_module)

from agents.customer import agent as customer_agent


class FakeTool:
    def __init__(self, name: str):
        self.name = name
        self.calls = []

    def invoke(self, args: dict):
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
            result = customer_agent.customer_agent_node(
                {
                    "customer_id": 98,
                    "messages": [],
                    "agent_outputs": {},
                    "completed_agents": [],
                    "current_step": 0,
                }
            )

        payload = result["customer_result"]["result"]

        self.assertEqual(
            [tool.calls for tool in fake_tools],
            [
                [{"customer_name": "고객_098"}],
                [{"customer_name": "고객_098"}],
                [{"customer_name": "고객_098"}],
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
        self.assertIn("고객_098", payload["summary"])
        self.assertEqual(result["agent_outputs"]["customer_agent"], result["customer_result"])

    def test_customer_agent_extracts_customer_id_from_message_text(self):
        fake_tools = [
            FakeTool("get_customer_profile"),
            FakeTool("get_customer_accounts"),
            FakeTool("get_payment_history"),
        ]

        with patch.object(customer_agent, "CUSTOMER_TOOLS", fake_tools):
            result = customer_agent.customer_agent_node(
                {
                    "messages": [("user", "고객 ID 98번 가입 상품 조회해줘")],
                    "agent_outputs": {},
                    "completed_agents": [],
                    "current_step": 0,
                }
            )

        self.assertTrue(result["customer_result"]["result"]["tool_results"])
        self.assertEqual(fake_tools[0].calls, [{"customer_name": "고객_098"}])


if __name__ == "__main__":
    unittest.main()
