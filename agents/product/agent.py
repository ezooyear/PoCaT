"""
Product agent wrapper.
"""
from agents.base import run_agent_loop
from agents.product.prompts import PRODUCT_SYSTEM_PROMPT
from agents.product.tools import PRODUCT_TOOLS, extract_product_candidates_from_search_results
from graph.state import AgentState


def product_agent_node(state: AgentState) -> dict:
    result = run_agent_loop(
        state=state,
        system_prompt=PRODUCT_SYSTEM_PROMPT,
        tools=PRODUCT_TOOLS,
        output_key="product_agent",
        result_key="product_result",
        max_iterations=3,
    )

    product_result = result.get("product_result") or {}
    payload = product_result.get("result", {}) if isinstance(product_result, dict) else {}
    tool_results = payload.get("tool_results", []) if isinstance(payload, dict) else []

    raw_search_results = [
        item.get("tool_result")
        for item in tool_results
        if isinstance(item, dict) and isinstance(item.get("tool_result"), str)
    ]
    product_candidates = extract_product_candidates_from_search_results(raw_search_results)

    # 비즈니스 Fallback: 검색된 상품 후보군이 없는 경우 에러 상태로 세팅
    if not product_candidates:
        if isinstance(product_result, dict):
            product_result["status"] = "failed"
            product_result["error"] = "검색 조건에 맞는 금융 상품을 발견하지 못했습니다. 질문 내 키워드를 완화하거나 다른 조건으로 다시 시도해 주세요."
            if isinstance(payload, dict):
                payload["summary"] = "검색 조건에 부합하는 예적금 상품을 찾을 수 없습니다."

    if isinstance(product_result, dict) and isinstance(payload, dict):
        payload["product_candidates"] = product_candidates
        payload["searched_products"] = [item["product_name"] for item in product_candidates]
        product_result["result"] = payload
        result["product_result"] = product_result

    agent_outputs = dict(result.get("agent_outputs") or {})
    if isinstance(agent_outputs.get("product_agent"), dict):
        agent_outputs["product_agent"]["result"] = product_result.get("result", {})
    result["agent_outputs"] = agent_outputs
    result["product_candidates"] = product_candidates
    return result
