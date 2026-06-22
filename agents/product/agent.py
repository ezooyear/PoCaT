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

import re
import time
from typing import Any

from agents.base import make_agent_result, run_agent_loop, run_agent_loop_async
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


def _normalize_product_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    raw_text = str(candidate.get("raw_text") or "")
    name = str(candidate.get("product_name") or "미확인 상품").strip()
    bank = _extract_bank(raw_text)
    product_type = _extract_product_type(raw_text)
    base_rate, max_rate = _extract_rates(raw_text)
    min_amount, max_amount = _extract_amount_bounds(raw_text)
    term_options = _extract_term_options(raw_text)
    preferential_conditions = _extract_preferential_conditions(raw_text)

    source_file = candidate.get("source_file")
    page = candidate.get("page")
    source_pages = candidate.get("source_pages") or ([page] if page else [])

    evidence = candidate.get("evidence") or [
        {
            "field": "raw_text",
            "value": raw_text[:500],
            "source": "rag_search",
            "source_file": source_file,
            "page": page,
            "pages": source_pages,
            "text": raw_text[:500],
            "confidence": "medium",
        }
    ]
    return {
        "product_id": candidate.get("product_id"),
        "product_name": name,
        "bank": bank,
        "product_type": product_type,
        "base_rate": base_rate,
        "max_rate": max_rate,
        "base_rate_text": f"연 {base_rate:.2f}%" if base_rate is not None else None,
        "max_rate_text": f"연 {max_rate:.2f}%" if max_rate is not None else None,
        "min_monthly_amount": min_amount,
        "max_monthly_amount": max_amount,
        "term_options_months": term_options,
        "eligibility_text": _extract_eligibility_text(raw_text),
        "preferential_conditions_text": _extract_preferential_conditions_text(raw_text),
        "preferential_conditions": preferential_conditions,
        "evidence": [
            {   "source_file": source_file,
                "page": page,
                "source_pages": source_pages,
                "evidence": evidence,
            }
        ],
        "source": "rag_search",
        "raw_text": raw_text,
    }


def _extract_bank(text: str) -> str | None:
    lowered = text.lower()
    if "국민은행" in lowered:
        return "KB국민은행"
    if "우리은행" in lowered:
        return "우리은행"
    if "신한은행" in lowered:
        return "신한은행"
    if "하나은행" in lowered:
        return "하나은행"
    if "카카오뱅크" in lowered:
        return "카카오뱅크"
    return None


def _extract_product_type(text: str) -> str | None:
    lowered = text.lower()
    if "적금" in lowered:
        return "적금"
    if "예금" in lowered:
        return "예금"
    if "부금" in lowered:
        return "부금"
    return None


def _extract_rates(text: str) -> tuple[float | None, float | None]:
    matches = []
    for m in re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*%", text):
        val = float(m.replace(",", ""))
        # 일반적인 예적금 금리 범위(0.1% ~ 12.0%) 내의 값만 수용 (담보대출 95% 비율 등 오인 차단)
        if 0.1 <= val <= 12.0:
            matches.append(val)
            
    if not matches:
        return None, None
    if len(matches) == 1:
        return matches[0], matches[0]
    return matches[0], max(matches)


def _extract_amount_bounds(text: str) -> tuple[int | None, int | None]:
    min_vals = [int(re.sub(r"[^0-9]", "", m)) for m in re.findall(r"최소\s*가입\s*금액[:\s]*([0-9,]+)원", text)]
    max_vals = [int(re.sub(r"[^0-9]", "", m)) for m in re.findall(r"최대\s*가입\s*금액[:\s]*([0-9,]+)원", text)]
    if not min_vals:
        min_vals = [int(re.sub(r"[^0-9]", "", m)) for m in re.findall(r"월\s*최소\s*납입[:\s]*([0-9,]+)원", text)]
    if not max_vals:
        max_vals = [int(re.sub(r"[^0-9]", "", m)) for m in re.findall(r"월\s*최대\s*납입[:\s]*([0-9,]+)원", text)]
    return (min_vals[0] if min_vals else None, max_vals[0] if max_vals else None)


def _extract_term_options(text: str) -> list[int]:
    months = [int(m) for m in re.findall(r"([0-9]{1,2})\s*개월", text)]
    return sorted(set(months))


def _extract_eligibility_text(text: str) -> str | None:
    match = re.search(r"가입\s*조건[:\s]*(.+?)(?:\n|$)", text)
    if match:
        return match.group(1).strip()
    return None


def _extract_preferential_conditions_text(text: str) -> str | None:
    match = re.search(r"우대.*금리[:\s]*(.+?)(?:\n|$)", text)
    if match:
        return match.group(1).strip()
    return None


def _extract_preferential_conditions(text: str) -> list[dict[str, Any]]:
    # 텍스트에 우대 혜택을 언급하는 맥락이 없는 경우 우대 조건 목록 생성을 차단합니다.
    if not any(w in text for w in ["우대", "최고이율", "최고금리", "혜택"]):
        return []

    conditions = []
    for keyword in ["급여이체", "자동이체", "카드", "주거래", "마케팅"]:
        if keyword in text:
            # 키워드 주변 150자 이내에 우대/가산/이율/금리/+/% 등 실질적인 이율 우대 문맥이 있을 때만 추가
            idx = text.find(keyword)
            context = text[max(0, idx-50):min(len(text), idx+150)]
            if any(w in context for w in ["우대", "가산", "더해", "추가", "이율", "금리", "+", "%"]):
                conditions.append(
                    {
                        "name": keyword,
                        "condition": f"{keyword} 조건을 만족하면 우대금리 적용 가능",
                        "rate": None,
                    }
                )
    return conditions


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

async def product_agent_node(state: AgentState) -> dict:
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
            user_query = str(state.get("user_query") or "")
            is_comparative = any(
                keyword in user_query
                for keyword in ["비교", "차이", "모두", "목록", "공통", "다른점", "차이점", "추천", "순위", "종합", "맞는", "적합한", "예적금"]
            )

            if is_comparative:
                # LLM 루프를 거치지 않고 고객 정보를 반영하여 LLM(Temp=0)으로 RAG 쿼리 재정형 후 직접 실행
                from agents.product.tools import search_terms
                from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
                from config.settings import get_llm

                customer_profile = state.get("customer_profile") or {}
                profile_desc = ""
                if customer_profile:
                    profile_desc = (
                        f"고객 프로필 정보:\n"
                        f"- 나이: {customer_profile.get('age') or '알 수 없음'}\n"
                        f"- 직업: {customer_profile.get('job') or '알 수 없음'}\n"
                        f"- 월 가용 저축액: {customer_profile.get('monthly_saving_amount') or '알 수 없음'}\n"
                    )

                system_prompt = (
                    "당신은 금융 상품 약관 RAG 검색에 최적화된 검색 키워드 정형화 전문가입니다.\n"
                    "제공된 고객 프로필 정보와 사용자 질문을 기반으로, "
                    "ChromaDB에서 가장 관련성 높은 예적금 상품 약관을 검색하기 위한 검색 키워드 조합 한 줄만을 생성하십시오.\n\n"
                    "## 지침:\n"
                    "1. 설명이나 부연설명 없이 오직 RAG 검색에 유용한 키워드들로만 구성된 최적화된 검색 쿼리 단 하나만 한 줄로 출력해야 합니다. 따옴표도 붙이지 마십시오.\n"
                    "2. 고객의 나이, 직업(예: 군인 등) 정보가 검색에 필요한 상품 범주와 관련이 있다면 검색 키워드에 반드시 포함시키십시오.\n"
                    "3. 사용자가 전체 상품 추천 및 비교를 요청했을 때 고객의 직업이 '군인'인 경우, 군인 대상 특수 상품('KB나라사랑적금(직업군인용)', 'KB장병내일준비적금', '장기간부 적금')이 모두 누락 없이 RAG 상위에 검색되도록 'KB국민은행 군인 장병 나라사랑 장기간부 예적금 상품 종류 금리 가입대상 조건 우대이율' 키워드 조합을 적극 포함하여 재구성하십시오.\n"
                    "4. 만약 특별한 조건이나 타겟 고객 정보가 없다면 기본 검색어 조합인 'KB국민은행 예적금 상품 종류 금리 가입대상 조건 우대이율' 형태로 정리하십시오.\n"
                )

                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=f"{profile_desc}사용자 질문: '{user_query}'"),
                ]

                try:
                    llm = get_llm(temperature=0)
                    response = await llm.ainvoke(messages)
                    reformed_query = str(getattr(response, "content", response) or "").strip()
                    reformed_query = reformed_query.replace("'", "").replace('"', "")
                    if not reformed_query:
                        reformed_query = "KB국민은행 예적금 상품 종류 금리 가입대상 조건 우대이율"
                except Exception:
                    reformed_query = "KB국민은행 예적금 상품 종류 금리 가입대상 조건 우대이율"

                tool_result_str = await search_terms.ainvoke({"query": reformed_query})

                tool_results_list = [{
                    "tool_name": "search_terms",
                    "tool_args": {"query": reformed_query},
                    "tool_result": tool_result_str,
                }]

                summary = "RAG 검색을 통해 KB국민은행의 주요 예적금 상품 목록과 가입 요건을 일관되게 추출하였습니다."

                structured_result = make_agent_result(
                    status="success",
                    result={
                        "summary": summary,
                        "tool_results": tool_results_list,
                    },
                    evidence=tool_results_list,
                    error=None,
                )

                outputs = dict(state.get("agent_outputs") or {})
                outputs["product_agent"] = structured_result

                completed_agents = list(state.get("completed_agents") or [])
                if "product_agent" not in completed_agents:
                    completed_agents.append("product_agent")

                result = {
                    "messages": [AIMessage(content=summary)],
                    "agent_outputs": outputs,
                    "current_step": (state.get("current_step") or 0) + 1,
                    "current_agent": "product_agent",
                    "completed_agents": completed_agents,
                    "product_result": structured_result,
                }
            else:
                result = await run_agent_loop_async(
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
                products = [_normalize_product_candidate(candidate) for candidate in product_candidates]

                # 군인 자격 필터링 (고객 직업이 '군인'이 아닌 경우 군인 전용 상품 3종 사전 제외)
                from agents.base import get_customer_profile as base_get_customer_profile
                customer_profile = state.get("customer_profile")
                if not customer_profile:
                    profile_res = base_get_customer_profile(state)
                    customer_profile = profile_res.get("data") or {}

                is_soldier = str(customer_profile.get("job") or "").strip() == "군인"
                if not is_soldier:
                    military_product_names = {"KB나라사랑적금(직업군인용)", "KB장병내일준비적금", "장기간부 적금"}
                    products = [p for p in products if p.get("product_name") not in military_product_names]
                    product_candidates = [c for c in product_candidates if c.get("product_name") not in military_product_names]

                outer_meta["product_count"] = len(products)
                outer_meta["structured_product_count"] = len(products)
                outer_meta["structured_product_names"] = [
                    c.get("product_name", "") for c in products[:10]
                ]

                update_observation(
                    parse_span,
                    output={
                        "product_count": len(products),
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
                product_candidates=products,
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

            if not products:
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
                payload["products"] = products
                payload["product_candidates"] = product_candidates
                payload["structured_product_count"] = len(products)
                payload["structured_product_names"] = [
                    item["product_name"] for item in products
                ]
                payload["searched_products"] = [
                    item["product_name"] for item in products
                ]
                product_result_raw["result"] = payload

                product_evidence = []
                for product in products:
                    if isinstance(product, dict):
                        product_evidence.extend(product.get("evidence") or [])

                product_result_raw["evidence"] = product_evidence

                product_result_wrapped = make_agent_result(
                    status=product_result_raw.get("status", "success"),
                    result=payload,
                    evidence=product_evidence,
                    error=product_result_raw.get("error"),
                )
                result["product_result"] = product_result_wrapped
            else:
                result["product_result"] = product_result_raw

            agent_outputs = dict(result.get("agent_outputs") or {})
            if isinstance(agent_outputs.get("product_agent"), dict):
                agent_outputs["product_agent"] = make_agent_result(
                    status=product_result_raw.get("status", "success"),
                    result=payload if isinstance(payload, dict) else {},
                    evidence=product_result_raw.get("evidence", []),
                    error=product_result_raw.get("error"),
                )
            result["agent_outputs"] = agent_outputs
            result["product_candidates"] = products
            result["products"] = products

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
