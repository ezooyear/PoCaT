"""
Product agent wrapper — Langfuse full instrumentation.

span tree:
  product_agent (outer)
    ├── product_agent.prepare_inputs
    ├── product_agent.llm_loop        (via run_agent_loop)
    ├── product_agent.search_terms    (retroactive — summarises tool call results)
    ├── product_agent.parse_products
    └── product_agent.finalize
"""

import time
from typing import Any

from agents.base import run_agent_loop
from agents.product.prompts import PRODUCT_SYSTEM_PROMPT
from agents.product.tools import PRODUCT_TOOLS, extract_product_candidates_from_search_results
from graph.state import AgentState
from observability.langfuse import langfuse_observation, update_observation, safe_jsonable


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _tool_names_in_registry() -> list[str]:
    return [t.name for t in PRODUCT_TOOLS]


def _extract_tool_results(result: dict) -> list[dict]:
    product_result = result.get("product_result") or {}
    payload = product_result.get("result", {}) if isinstance(product_result, dict) else {}
    tool_results = payload.get("tool_results", []) if isinstance(payload, dict) else []
    return [t for t in tool_results if isinstance(t, dict)]


def _analyze_search_calls(tool_results: list[dict]) -> dict[str, Any]:
    """search_terms 도구 호출 분석 결과 반환."""
    calls = [t for t in tool_results if t.get("tool_name") == "search_terms"]
    error_calls = [
        t for t in tool_results
        if str(t.get("tool_result", "")).startswith("Tool execution error")
    ]

    if not calls:
        return {
            "called": False,
            "call_count": 0,
            "search_query": None,
            "result_count": 0,
            "is_empty": True,
            "result_preview": "",
        }

    first = calls[0]
    query = (first.get("tool_args") or {}).get("query", "")
    result_text = str(first.get("tool_result", ""))
    is_empty = (
        "검색된 약관 정보가 없습니다" in result_text
        or not result_text.strip()
    )
    # rough count: number of "[N]" markers in output
    result_count = result_text.count("\n\n---\n\n") + 1 if not is_empty else 0

    return {
        "called": True,
        "call_count": len(calls),
        "search_query": query[:300] if query else None,
        "result_count": result_count,
        "is_empty": is_empty,
        "result_preview": result_text[:400],
        "error_call_count": len(error_calls),
    }


def _determine_fallback_reason(
    search_enabled: bool,
    search_info: dict,
    product_candidates: list,
    tool_error_count: int,
) -> tuple[str | None, str]:
    """(fallback_reason, status) 결정."""
    if tool_error_count > 0 and not search_info["called"]:
        return "tool_error", "error"

    if not search_enabled:
        return "rag_disabled", "fallback"

    if not search_info["called"]:
        return "tool_not_called", "fallback"

    if search_info["is_empty"]:
        return "no_search_results", "fallback"

    if not product_candidates:
        return "empty_product_candidates", "fallback"

    return None, "success"


def _analyze_product_result_storage(result: dict) -> dict[str, Any]:
    """product_result 저장 구조 분석 (keys/preview 만, state 전체 제외)."""
    product_result = result.get("product_result") or {}
    pr_keys = list(product_result.keys()) if isinstance(product_result, dict) else []

    pr_inner = product_result.get("result", {}) if isinstance(product_result, dict) else {}
    pr_inner_keys = list(pr_inner.keys()) if isinstance(pr_inner, dict) else []

    product_candidates = pr_inner.get("product_candidates", []) if isinstance(pr_inner, dict) else []
    products_direct = pr_inner.get("products", []) if isinstance(pr_inner, dict) else []
    summary = pr_inner.get("summary", "") if isinstance(pr_inner, dict) else ""
    pr_status = product_result.get("status") if isinstance(product_result, dict) else None

    structured_count = len(product_candidates) if isinstance(product_candidates, list) else 0
    structured_names = [
        c.get("product_name", "") for c in (product_candidates or [])[:10]
        if isinstance(c, dict)
    ]

    return {
        "product_result_keys": pr_keys,
        "product_result_result_keys": pr_inner_keys,
        "has_product_result": bool(product_result),
        "has_product_result_products": bool(products_direct),
        "has_product_result_candidates": bool(product_candidates),
        "product_result_status": pr_status,
        "product_result_summary_preview": str(summary)[:200] if summary else "",
        "structured_product_count": structured_count,
        "structured_product_names": structured_names,
    }


# ---------------------------------------------------------------------------
# main node
# ---------------------------------------------------------------------------

def product_agent_node(state: AgentState) -> dict:
    start_time = time.time()

    available_tool_names = _tool_names_in_registry()
    search_enabled = "search_terms" in available_tool_names
    rag_enabled = search_enabled
    product_agent_mode = "rag" if rag_enabled else "no_rag"

    outer_meta: dict[str, Any] = {
        "agent_name": "product_agent",
        "current_step": (state.get("current_step") or 0) + 1,
        "result_key": "product_result",
        "status": "running",
        "plan": safe_jsonable(state.get("plan") or []),
        "completed_agents": list(state.get("completed_agents") or []),
        "product_agent_mode": product_agent_mode,
        "search_enabled": search_enabled,
        "rag_enabled": rag_enabled,
        "tool_count": 0,
        "tool_names": [],
        "tool_error_count": 0,
        "search_query": None,
        "search_result_count": 0,
        "rag_result_count": 0,
        "product_count": 0,
        "structured_product_count": 0,
        "structured_product_names": [],
        "fallback_reason": None,
        "input_preview": str(state.get("user_query") or "")[:200],
        "output_preview": "",
        "duration_ms": 0,
    }

    result: dict = {}
    product_candidates: list = []

    with langfuse_observation(
        name="product_agent",
        as_type="span",
        input={
            "user_query": str(state.get("user_query") or "")[:300],
            "task_type": state.get("task_type"),
            "search_enabled": search_enabled,
            "rag_enabled": rag_enabled,
            "available_tools": available_tool_names,
        },
        metadata=outer_meta,
    ) as outer_span:
        try:
            # ---- sub-span 1: prepare_inputs --------------------------------
            ps = time.time()
            with langfuse_observation(
                name="product_agent.prepare_inputs",
                as_type="span",
            ) as prep_span:
                prior_keys = list((state.get("agent_outputs") or {}).keys())
                update_observation(
                    prep_span,
                    output={
                        "prior_agent_keys": prior_keys,
                        "has_customer_result": bool(state.get("customer_result")),
                        "has_product_result_already": bool(state.get("product_result")),
                        "user_query": str(state.get("user_query") or "")[:200],
                        "task_type": state.get("task_type"),
                        "tool_names_available": available_tool_names,
                    },
                    metadata={
                        "step": "product_agent.prepare_inputs",
                        "duration_ms": int((time.time() - ps) * 1000),
                    },
                )

            # ---- sub-span 2: llm_loop (via run_agent_loop) -----------------
            result = run_agent_loop(
                state=state,
                system_prompt=PRODUCT_SYSTEM_PROMPT,
                tools=PRODUCT_TOOLS,
                output_key="product_agent",
                result_key="product_result",
                max_iterations=3,
                span_name="product_agent.llm_loop",
            )

            # extract raw tool call records from the completed loop
            tool_results_list = _extract_tool_results(result)
            tool_error_count = sum(
                1 for t in tool_results_list
                if str(t.get("tool_result", "")).startswith("Tool execution error")
            )
            search_info = _analyze_search_calls(tool_results_list)

            outer_meta["tool_count"] = len(tool_results_list)
            outer_meta["tool_names"] = [t.get("tool_name") for t in tool_results_list[:10]]
            outer_meta["tool_error_count"] = tool_error_count

            if search_info["called"]:
                outer_meta["search_query"] = search_info.get("search_query")
                outer_meta["search_result_count"] = search_info.get("result_count", 0)
                outer_meta["rag_result_count"] = search_info.get("result_count", 0)

            # ---- sub-span 3: search_terms analysis -------------------------
            ss = time.time()
            with langfuse_observation(
                name="product_agent.search_terms",
                as_type="span",
            ) as search_span:
                if search_info["called"]:
                    update_observation(
                        search_span,
                        output={
                            "query": search_info.get("search_query"),
                            "result_count": search_info.get("result_count", 0),
                            "is_empty": search_info.get("is_empty"),
                            "call_count": search_info.get("call_count", 0),
                            "result_preview": search_info.get("result_preview", ""),
                        },
                        metadata={
                            "step": "product_agent.search_terms",
                            "enabled": search_enabled,
                            "duration_ms": int((time.time() - ss) * 1000),
                            "error": None,
                        },
                    )
                else:
                    skip_reason = (
                        "tool_not_called_by_llm"
                        if search_enabled
                        else "speed_optimized_branch_disabled_rag"
                    )
                    update_observation(
                        search_span,
                        output={"call_count": 0, "enabled": search_enabled},
                        metadata={
                            "step": "product_agent.search_terms",
                            "enabled": search_enabled,
                            "skip_reason": skip_reason,
                            "duration_ms": 0,
                            "error": None,
                        },
                    )

            # ---- sub-span 4: parse_products --------------------------------
            ps2 = time.time()
            with langfuse_observation(
                name="product_agent.parse_products",
                as_type="span",
            ) as parse_span:
                raw_search_results = [
                    item.get("tool_result")
                    for item in tool_results_list
                    if isinstance(item.get("tool_result"), str)
                ]
                product_candidates = extract_product_candidates_from_search_results(
                    raw_search_results
                )

                outer_meta["product_count"] = len(product_candidates)
                outer_meta["structured_product_count"] = len(product_candidates)
                outer_meta["structured_product_names"] = [
                    c.get("product_name", "") for c in product_candidates[:10]
                ]

                update_observation(
                    parse_span,
                    output={
                        "product_count": len(product_candidates),
                        "product_names": outer_meta["structured_product_names"],
                        "raw_result_count": len(raw_search_results),
                    },
                    metadata={
                        "step": "product_agent.parse_products",
                        "duration_ms": int((time.time() - ps2) * 1000),
                        "error": None,
                    },
                )

            # determine fallback_reason and final status
            fallback_reason, status = _determine_fallback_reason(
                search_enabled=search_enabled,
                search_info=search_info,
                product_candidates=product_candidates,
                tool_error_count=tool_error_count,
            )
            outer_meta["fallback_reason"] = fallback_reason
            outer_meta["status"] = status

            # mutate product_result if no candidates found (business fallback)
            product_result_raw = result.get("product_result") or {}
            payload = (
                product_result_raw.get("result", {})
                if isinstance(product_result_raw, dict)
                else {}
            )

            if not product_candidates:
                if isinstance(product_result_raw, dict):
                    product_result_raw["status"] = "failed"
                    product_result_raw["error"] = (
                        "검색 조건에 맞는 금융 상품을 발견하지 못했습니다. "
                        "질문 내 키워드를 완화하거나 다른 조건으로 다시 시도해 주세요."
                    )
                    if isinstance(payload, dict):
                        payload["summary"] = (
                            "검색 조건에 부합하는 예적금 상품을 찾을 수 없습니다."
                        )

            if isinstance(product_result_raw, dict) and isinstance(payload, dict):
                payload["product_candidates"] = product_candidates
                payload["searched_products"] = [
                    item["product_name"] for item in product_candidates
                ]
                product_result_raw["result"] = payload
                result["product_result"] = product_result_raw

            agent_outputs = dict(result.get("agent_outputs") or {})
            if isinstance(agent_outputs.get("product_agent"), dict):
                agent_outputs["product_agent"]["result"] = product_result_raw.get("result", {})
            result["agent_outputs"] = agent_outputs
            result["product_candidates"] = product_candidates

            # ---- sub-span 5: finalize --------------------------------------
            fs = time.time()
            storage_info = _analyze_product_result_storage(result)
            with langfuse_observation(
                name="product_agent.finalize",
                as_type="span",
            ) as finalize_span:
                update_observation(
                    finalize_span,
                    output={
                        "status": status,
                        "fallback_reason": fallback_reason,
                        **storage_info,
                    },
                    metadata={
                        "step": "product_agent.finalize",
                        "duration_ms": int((time.time() - fs) * 1000),
                    },
                )

        except Exception as exc:
            outer_meta["status"] = "error"
            outer_meta["fallback_reason"] = "exception"
            outer_meta["error"] = str(exc)[:500]
            raise

        finally:
            duration_ms = int((time.time() - start_time) * 1000)
            outer_meta["duration_ms"] = duration_ms

            storage_info = _analyze_product_result_storage(result)

            # summary_preview from LLM response
            pr = result.get("product_result") or {}
            summary_text = (
                pr.get("result", {}).get("summary", "")
                if isinstance(pr, dict) and isinstance(pr.get("result"), dict)
                else ""
            )
            outer_meta["output_preview"] = str(summary_text)[:200]

            update_observation(
                outer_span,
                output=safe_jsonable({
                    "status": outer_meta["status"],
                    "product_agent_mode": product_agent_mode,
                    "search_enabled": search_enabled,
                    "rag_enabled": rag_enabled,
                    "tool_count": outer_meta["tool_count"],
                    "tool_names": outer_meta["tool_names"],
                    "tool_error_count": outer_meta["tool_error_count"],
                    "search_query": outer_meta.get("search_query"),
                    "search_result_count": outer_meta.get("search_result_count", 0),
                    "rag_result_count": outer_meta.get("rag_result_count", 0),
                    "structured_product_count": outer_meta["structured_product_count"],
                    "structured_product_names": outer_meta["structured_product_names"],
                    "fallback_reason": outer_meta.get("fallback_reason"),
                    "duration_ms": duration_ms,
                    **storage_info,
                }),
                metadata=safe_jsonable(outer_meta),
            )

    return result
