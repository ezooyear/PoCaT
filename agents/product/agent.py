"""
Product agent wrapper.
"""

from __future__ import annotations

from copy import deepcopy
from time import perf_counter
from typing import Any

from agents.base import build_agent_trace_input, build_agent_trace_output, run_agent_loop
from agents.product.prompts import PRODUCT_SYSTEM_PROMPT
from agents.product.tools import (
    PRODUCT_TOOLS,
    extract_product_candidates_from_search_results,
    extract_products_from_product_summary,
    extract_structured_products_from_search_results,
    get_product_perf_stats,
    reset_product_perf_stats,
)
from graph.state import AgentState
from observability.langfuse import flush_langfuse, langfuse_observation, safe_jsonable, update_observation


def _clone_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return deepcopy(value)


def _build_product_schema_diagnostics(product_result: dict[str, Any], structured_product_names: list[str]) -> dict[str, Any]:
    payload = product_result.get("result", {}) if isinstance(product_result, dict) else {}
    payload_keys = sorted(product_result.keys()) if isinstance(product_result, dict) else []
    result_keys = sorted(payload.keys()) if isinstance(payload, dict) else []
    top_level_products = product_result.get("products") if isinstance(product_result, dict) else None
    nested_products = payload.get("products") if isinstance(payload, dict) else None
    return {
        "product_result_keys": payload_keys,
        "product_result_result_keys": result_keys,
        "has_product_result_products": isinstance(top_level_products, list) and len(top_level_products) > 0,
        "has_product_result_result_products": isinstance(nested_products, list) and len(nested_products) > 0,
        "structured_product_count": len(structured_product_names),
        "structured_product_names": structured_product_names,
    }


def _build_structured_product_result(
    base_product_result: dict[str, Any],
    *,
    summary: str,
    products: list[dict[str, Any]],
    product_candidates: list[dict[str, Any]],
    performance: dict[str, Any],
) -> dict[str, Any]:
    product_result = _clone_dict(base_product_result)
    payload = _clone_dict(product_result.get("result"))
    structured_product_names = [
        item.get("product_name")
        for item in products
        if isinstance(item, dict) and item.get("product_name")
    ]

    payload["summary"] = summary
    payload["products"] = deepcopy(products)
    payload["product_candidates"] = deepcopy(product_candidates)
    payload["searched_products"] = [item.get("product_name") for item in product_candidates if isinstance(item, dict)]
    payload["performance"] = deepcopy(performance)
    payload["structured_product_count"] = len(products)
    payload["structured_product_names"] = list(structured_product_names)
    payload["status"] = product_result.get("status", "success")

    product_result["status"] = product_result.get("status", "success")
    product_result["summary"] = summary
    product_result["products"] = deepcopy(products)
    product_result["product_candidates"] = deepcopy(product_candidates)
    product_result["searched_products"] = [item.get("product_name") for item in product_candidates if isinstance(item, dict)]
    product_result["performance"] = deepcopy(performance)
    product_result["structured_product_count"] = len(products)
    product_result["structured_product_names"] = list(structured_product_names)
    product_result["result"] = payload

    schema_diagnostics = _build_product_schema_diagnostics(product_result, structured_product_names)
    product_result["schema_diagnostics"] = schema_diagnostics
    product_result["result"]["schema_diagnostics"] = schema_diagnostics
    return product_result


def product_agent_node(state: AgentState) -> dict[str, Any]:
    reset_product_perf_stats()
    started_at = perf_counter()

    raw_result = run_agent_loop(
        state=state,
        system_prompt=PRODUCT_SYSTEM_PROMPT,
        tools=PRODUCT_TOOLS,
        output_key="product_agent",
        result_key="product_result",
        max_iterations=3,
    )

    result = _clone_dict(raw_result)
    base_product_result = _clone_dict(raw_result.get("product_result"))
    base_payload = _clone_dict(base_product_result.get("result"))
    tool_results = base_payload.get("tool_results", []) if isinstance(base_payload.get("tool_results"), list) else []
    summary = str(base_payload.get("summary") or base_product_result.get("summary") or "")

    raw_search_results = [
        item.get("tool_result")
        for item in tool_results
        if isinstance(item, dict) and isinstance(item.get("tool_result"), str)
    ]

    products = extract_products_from_product_summary(summary)
    if not products:
        products = extract_structured_products_from_search_results(raw_search_results)

    product_candidates = extract_product_candidates_from_search_results(raw_search_results)
    if not product_candidates and products:
        product_candidates = deepcopy(products)

    response_message = raw_result.get("messages", [None])[0] if isinstance(raw_result.get("messages"), list) else None
    usage_metadata = getattr(response_message, "usage_metadata", None)
    performance = {
        **get_product_perf_stats(),
        "total_duration_ms": round((perf_counter() - started_at) * 1000, 2),
        "tool_count": len(tool_results),
        "tool_names": [item.get("tool_name") for item in tool_results if isinstance(item, dict)],
        "raw_search_result_count": len(raw_search_results),
        "product_candidate_count": len(product_candidates),
        "final_products_count": len(products),
        "max_iterations": 3,
        "retry_used": len(tool_results) > 1,
        "iterations_used": max(1, len(tool_results)) if tool_results else 1,
        "llm_usage": safe_jsonable(usage_metadata),
    }

    if not product_candidates:
        base_product_result["status"] = "failed"
        base_product_result["error"] = "검색 조건에 맞는 금융 상품을 발견하지 못했습니다. 질문을 더 구체화하거나 다른 조건으로 다시 시도해 주세요."
        if isinstance(base_payload, dict):
            base_payload["summary"] = "검색 조건에 부합하는 예적금 상품을 찾을 수 없습니다."
        summary = str(base_payload.get("summary") or summary)

    product_result = _build_structured_product_result(
        base_product_result,
        summary=summary,
        products=products,
        product_candidates=product_candidates,
        performance=performance,
    )
    if base_product_result.get("error"):
        product_result["error"] = base_product_result.get("error")

    agent_outputs = _clone_dict(raw_result.get("agent_outputs"))
    product_agent_output = _clone_dict(agent_outputs.get("product_agent"))
    product_agent_output["status"] = product_result.get("status")
    product_agent_output["error"] = product_result.get("error")
    product_agent_output["summary"] = product_result.get("summary")
    product_agent_output["products"] = deepcopy(product_result.get("products", []))
    product_agent_output["structured_product_count"] = product_result.get("structured_product_count", 0)
    product_agent_output["structured_product_names"] = list(product_result.get("structured_product_names", []))
    product_agent_output["schema_diagnostics"] = deepcopy(product_result.get("schema_diagnostics", {}))
    product_agent_output["result"] = deepcopy(product_result.get("result", {}))
    product_agent_output["product_result"] = deepcopy(product_result)
    agent_outputs["product_agent"] = product_agent_output

    agent_logs = list(raw_result.get("agent_logs") or state.get("agent_logs") or [])
    agent_logs.append({"agent": "product_agent", "performance": deepcopy(performance)})

    result["product_result"] = deepcopy(product_result)
    result["agent_outputs"] = agent_outputs
    result["agent_logs"] = agent_logs
    result["product_candidates"] = deepcopy(product_candidates)
    result["product_performance"] = deepcopy(performance)
    result["products"] = deepcopy(products)
    result["structured_product_count"] = product_result.get("structured_product_count", 0)
    result["structured_product_names"] = list(product_result.get("structured_product_names", []))
    result["product_schema_diagnostics"] = deepcopy(product_result.get("schema_diagnostics", {}))
    return safe_jsonable(result)


_product_agent_node_impl = product_agent_node


def product_agent_node(state: AgentState) -> dict[str, Any]:
    try:
        with langfuse_observation(
            name="product_agent",
            as_type="span",
            input=build_agent_trace_input(
                state,
                agent_name="product_agent",
                result_key="product_result",
                max_iterations=3,
            ),
            metadata={"agent": "product_agent", "phase": "finalize"},
        ) as observation:
            result = _product_agent_node_impl(state)
            product_result = result.get("product_result", {})
            payload = product_result.get("result", {}) if isinstance(product_result, dict) else {}
            update_observation(
                observation,
                output=build_agent_trace_output(
                    product_result,
                    agent_name="product_agent",
                    state=state,
                    result_key="product_result",
                    extra_output={
                        "products": payload.get("products"),
                        "product_candidates": payload.get("product_candidates"),
                        "performance": payload.get("performance"),
                        "structured_product_count": payload.get("structured_product_count"),
                        "structured_product_names": payload.get("structured_product_names"),
                        "schema_diagnostics": payload.get("schema_diagnostics") or product_result.get("schema_diagnostics"),
                    },
                ),
                metadata={
                    "agent": "product_agent",
                    "status": product_result.get("status") if isinstance(product_result, dict) else None,
                    "product_count": len(payload.get("products") or []) if isinstance(payload, dict) else 0,
                    "structured_product_count": payload.get("structured_product_count") if isinstance(payload, dict) else 0,
                    "structured_product_names": ",".join(payload.get("structured_product_names") or [])[:500] if isinstance(payload, dict) else None,
                    "product_result_keys": ",".join((payload.get("schema_diagnostics") or product_result.get("schema_diagnostics") or {}).get("product_result_keys", []))[:500]
                    if isinstance(product_result, dict)
                    else None,
                    "product_result_result_keys": ",".join((payload.get("schema_diagnostics") or product_result.get("schema_diagnostics") or {}).get("product_result_result_keys", []))[:500]
                    if isinstance(product_result, dict)
                    else None,
                    "has_product_result_products": (payload.get("schema_diagnostics") or product_result.get("schema_diagnostics") or {}).get("has_product_result_products")
                    if isinstance(product_result, dict)
                    else None,
                    "has_product_result_result_products": (payload.get("schema_diagnostics") or product_result.get("schema_diagnostics") or {}).get("has_product_result_result_products")
                    if isinstance(product_result, dict)
                    else None,
                },
            )
            return result
    finally:
        flush_langfuse()
