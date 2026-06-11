"""
Eligibility Agent
- Customer/Product 결과를 받아 상품별 가입 가능 여부를 구조화합니다.
"""
from langchain_core.messages import AIMessage

from agents.eligibility.prompts import ELIGIBILITY_SYSTEM_PROMPT
from agents.eligibility.tools import (
    build_eligibility_summary,
    evaluate_product_eligibility,
    extract_product_candidates,
    parse_customer_accounts,
    parse_customer_profile,
)
from graph.state import AgentState


def eligibility_agent_node(state: AgentState) -> dict:
    # 고객 정보와 상품 정보를 이어 받아 eligibility_results를 만드는 진입 함수입니다.
    agent_outputs = dict(state.get("agent_outputs") or {})

    customer_profile_raw = state.get("customer_profile")
    if not customer_profile_raw:
        customer_profile_raw = agent_outputs.get("customer_agent", "")
    customer_profile = parse_customer_profile(customer_profile_raw)

    customer_accounts_raw = state.get("customer_accounts")
    if customer_accounts_raw is None:
        customer_accounts_raw = ""
    if not customer_accounts_raw:
        customer_accounts_raw = agent_outputs.get("customer_agent", "")
    customer_accounts = parse_customer_accounts(customer_accounts_raw)

    product_candidates = state.get("product_candidates")
    if not product_candidates:
        product_candidates = extract_product_candidates(agent_outputs.get("product_agent", ""))

    results = [
        evaluate_product_eligibility(customer_profile, customer_accounts, product)
        for product in product_candidates
    ]
    summary = build_eligibility_summary(results)
    agent_outputs["eligibility_agent"] = summary

    return {
        "messages": [AIMessage(content=summary)],
        "agent_outputs": agent_outputs,
        "current_step": (state.get("current_step") or 0) + 1,
        "customer_profile": customer_profile,
        "customer_accounts": customer_accounts,
        "product_candidates": product_candidates,
        "eligibility_results": results,
        "context": {
            **(state.get("context") or {}),
            "eligibility_prompt": ELIGIBILITY_SYSTEM_PROMPT,
        },
    }
