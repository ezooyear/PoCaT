"""
Eligibility agent.
"""
from langchain_core.messages import AIMessage

from agents.base import make_agent_result
from agents.eligibility.prompts import ELIGIBILITY_SYSTEM_PROMPT
from agents.eligibility.tools import (
    build_eligibility_summary,
    evaluate_product_eligibility,
    extract_product_candidates,
    parse_customer_accounts,
    parse_customer_profile,
)
from graph.state import AgentState
from observability.langfuse import langfuse_observation


def eligibility_agent_node(state: AgentState) -> dict:
    with langfuse_observation(
        name="eligibility_agent",
        as_type="span",
        input={"task_type": state.get("task_type")},
        metadata={"agent": "eligibility_agent"},
    ):
        agent_outputs = dict(state.get("agent_outputs") or {})
        customer_result = state.get("customer_result") or {}
        product_result = state.get("product_result") or {}

        customer_profile = state.get("customer_profile")
        if not customer_profile:
            customer_profile_raw = _extract_tool_result(customer_result, "get_customer_profile")
            if not customer_profile_raw:
                customer_profile_raw = _extract_summary(agent_outputs.get("customer_agent"))
            customer_profile = parse_customer_profile(customer_profile_raw)

        customer_accounts = state.get("customer_accounts")
        if customer_accounts is None:
            customer_accounts = []
        if not customer_accounts:
            customer_accounts_raw = _extract_tool_result(customer_result, "get_customer_accounts")
            if not customer_accounts_raw:
                customer_accounts_raw = _extract_summary(agent_outputs.get("customer_agent"))
            customer_accounts = parse_customer_accounts(customer_accounts_raw)

        product_candidates = state.get("product_candidates")
        if not product_candidates:
            product_candidates = _extract_product_candidates_from_result(product_result)
        if not product_candidates:
            product_candidates = extract_product_candidates(_extract_summary(agent_outputs.get("product_agent")))
        else:
            product_candidates = extract_product_candidates(product_candidates)

        results = [
            evaluate_product_eligibility(customer_profile, customer_accounts, product)
            for product in product_candidates
        ]
        summary = build_eligibility_summary(results)

        eligible_products = [
            item for item in results
            if item.get("eligible") is True and not item.get("check_required")
        ]
        needs_check_products = [
            item for item in results
            if item.get("eligible") is True and item.get("check_required")
        ]
        rejected_products = [
            item for item in results
            if item.get("eligible") is not True
        ]

        eligibility_result = make_agent_result(
            status="success",
            result={
                "summary": summary,
                "results": results,
                "eligible_products": eligible_products,
                "needs_check_products": needs_check_products,
                "rejected_products": rejected_products,
                "result_count": len(results),
                "recommendable_count": len(eligible_products),
                "needs_check_count": len(needs_check_products),
                "rejected_count": len(rejected_products),
                "customer_profile": customer_profile,
                "customer_accounts": customer_accounts,
                "product_candidates": product_candidates,
            },
            evidence=results,
            error=None,
        )
        agent_outputs["eligibility_agent"] = eligibility_result

        completed_agents = list(state.get("completed_agents") or [])
        if "eligibility_agent" not in completed_agents:
            completed_agents.append("eligibility_agent")

        return {
            "messages": [AIMessage(content=summary)],
            "agent_outputs": agent_outputs,
            "current_step": (state.get("current_step") or 0) + 1,
            "current_agent": "eligibility_agent",
            "completed_agents": completed_agents,
            "customer_profile": customer_profile,
            "customer_accounts": customer_accounts,
            "product_candidates": product_candidates,
            "eligibility_results": results,
            "eligibility_result": eligibility_result,
            "context": {
                **(state.get("context") or {}),
                "eligibility_prompt": ELIGIBILITY_SYSTEM_PROMPT,
            },
        }


def _extract_summary(value) -> str:
    if isinstance(value, dict):
        result = value.get("result")
        if isinstance(result, dict) and isinstance(result.get("summary"), str):
            return result["summary"]
        if isinstance(value.get("summary"), str):
            return value["summary"]
    if isinstance(value, str):
        return value
    return ""


def _extract_tool_result(result_container, tool_name: str) -> str:
    if not isinstance(result_container, dict):
        return ""
    result = result_container.get("result", {})
    if not isinstance(result, dict):
        return ""
    tool_results = result.get("tool_results", [])
    if not isinstance(tool_results, list):
        return ""
    for item in tool_results:
        if not isinstance(item, dict):
            continue
        if item.get("tool_name") == tool_name and isinstance(item.get("tool_result"), str):
            return item["tool_result"]
    return ""


def _extract_product_candidates_from_result(product_result) -> list[dict]:
    if not isinstance(product_result, dict):
        return []
    result = product_result.get("result", {})
    if not isinstance(result, dict):
        return []
    product_candidates = result.get("product_candidates", [])
    if isinstance(product_candidates, list):
        return product_candidates
    return []
