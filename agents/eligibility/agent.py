"""
Eligibility agent.
"""
import time

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
from observability.langfuse import langfuse_observation, update_observation, safe_jsonable


def eligibility_agent_node(state: AgentState) -> dict:
    start_time = time.time()

    with langfuse_observation(
        name="eligibility_agent",
        as_type="span",
        input={
            "task_type": state.get("task_type"),
            "user_query": str(state.get("user_query") or "")[:200],
        },
        metadata={
            "agent_name": "eligibility_agent",
            "current_step": (state.get("current_step") or 0) + 1,
            "result_key": "eligibility_result",
            "status": "running",
            "plan": safe_jsonable(state.get("plan") or []),
            "completed_agents": list(state.get("completed_agents") or []),
        },
    ) as observation:
        status = "error"
        summary = ""
        product_candidate_source = "none"
        product_candidate_count = 0
        fallback_reason = None
        results: list = []
        eligible_products: list = []
        needs_check_products: list = []
        rejected_products: list = []
        product_candidates_structured: list = []
        customer_profile: dict = {}
        customer_accounts: list = []
        eligibility_result: dict = {}

        try:
            agent_outputs = dict(state.get("agent_outputs") or {})
            customer_result = state.get("customer_result") or {}
            product_result = state.get("product_result") or {}

            customer_profile = state.get("customer_profile")
            if not customer_profile:
                customer_profile_raw = _extract_tool_result(
                    customer_result, "get_customer_profile"
                )
                if not customer_profile_raw:
                    customer_profile_raw = _extract_summary(
                        agent_outputs.get("customer_agent")
                    )
                customer_profile = parse_customer_profile(customer_profile_raw)

            customer_accounts = state.get("customer_accounts")
            if customer_accounts is None:
                customer_accounts = []
            if not customer_accounts:
                customer_accounts_raw = _extract_tool_result(
                    customer_result, "get_customer_accounts"
                )
                if not customer_accounts_raw:
                    customer_accounts_raw = _extract_summary(
                        agent_outputs.get("customer_agent")
                    )
                customer_accounts = parse_customer_accounts(customer_accounts_raw)

            # --- product_candidates 수신 경로 추적 -------------------------
            product_candidates = state.get("product_candidates")

            if product_candidates:
                product_candidate_source = "state.product_candidates"
            else:
                # product_result.result.product_candidates
                from_result = _extract_product_candidates_from_result(product_result)
                if from_result:
                    product_candidates = from_result
                    product_candidate_source = "product_result.result.product_candidates"
                else:
                    # product_agent summary fallback
                    from_summary = extract_product_candidates(
                        _extract_summary(agent_outputs.get("product_agent"))
                    )
                    if from_summary:
                        product_candidates = from_summary
                        product_candidate_source = "product_agent.summary_fallback"
                    else:
                        product_candidates = []
                        product_candidate_source = "none"
                        fallback_reason = "empty_product_result"
            # pass through extract_product_candidates for structured format
            if product_candidates and not isinstance(product_candidates[0], str):
                product_candidates_structured = extract_product_candidates(product_candidates)
            else:
                product_candidates_structured = extract_product_candidates(product_candidates)

            product_candidate_count = len(product_candidates_structured)

            results = [
                evaluate_product_eligibility(
                    customer_profile, customer_accounts, product
                )
                for product in product_candidates_structured
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
                    "product_candidates": product_candidates_structured,
                },
                evidence=results,
                error=None,
            )
            agent_outputs["eligibility_agent"] = eligibility_result

            status = "success" if results else "fallback"
            if not results:
                fallback_reason = fallback_reason or "no_product_candidates"

        except Exception as exc:
            status = "error"
            fallback_reason = "exception"
            eligibility_result = make_agent_result(status="error", error=str(exc))
            agent_outputs = dict(state.get("agent_outputs") or {})
            raise

        finally:
            duration_ms = int((time.time() - start_time) * 1000)
            update_observation(
                observation,
                output={
                    "status": status,
                    "product_candidate_source": product_candidate_source,
                    "product_candidate_count": product_candidate_count,
                    "product_candidates_preview": [
                        c.get("product_name", str(c))[:80]
                        for c in product_candidates_structured[:5]
                    ],
                    "result_count": len(results),
                    "eligible_count": len(eligible_products),
                    "fallback_reason": fallback_reason,
                    "summary_preview": summary[:300],
                    "duration_ms": duration_ms,
                },
                metadata=safe_jsonable({
                    "agent_name": "eligibility_agent",
                    "current_step": (state.get("current_step") or 0) + 1,
                    "result_key": "eligibility_result",
                    "status": status,
                    "product_candidate_source": product_candidate_source,
                    "product_candidate_count": product_candidate_count,
                    "fallback_reason": fallback_reason,
                    "duration_ms": duration_ms,
                }),
            )

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
            "product_candidates": product_candidates_structured,
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
