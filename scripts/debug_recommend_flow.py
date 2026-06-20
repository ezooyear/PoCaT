# scripts/debug_recommend_flow.py
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from graph.builder import build_graph
from pprint import pprint
# prompt = "내 가입 상품, 월 저축 가능액, 가입 가능 조건을 종합해서 나에게 추천할 만한 예적금 상품을 순위와 이유, 주의사항까지 알려줘."
prompt = "월 30만원씩 2년 동안 저축하려고 해. 내 가입 상품과 월 저축 가능액을 고려해서 가입 가능한 예적금 상품을 추천해줘."
graph = build_graph()

result = graph.invoke({
    "messages": [("user", prompt)],
    "user_query": prompt,
    "next": "",
    "member_id": "123",
    "customer_id": 123,
    "context": None,
    "plan": [],
    "current_step": 0,
    "agent_outputs": {},
})

print("\n=== BASIC ===")
for k in ["user_query", "task_type", "plan", "completed_agents", "current_agent", "current_step"]:
    print(k, "=", result.get(k))

print("\n=== RESULT STATUS ===")
for k in ["customer_result", "product_result", "eligibility_result", "financial_result", "recommend_result", "validation_result"]:
    v = result.get(k)
    if isinstance(v, dict):
        print(k, "status=", v.get("status"), "summary=", str(v.get("summary") or v.get("result", {}).get("summary"))[:300])
    else:
        print(k, "=", type(v), v)

print("\n=== COUNTS ===")
print("product_candidates:", len(result.get("product_candidates") or []))
print("eligibility_results:", len(result.get("eligibility_results") or []))
print("financial_results:", len(result.get("financial_results") or []))
print("recommendation_results:", len(result.get("recommendation_results") or []))



print("\n=== ERRORS ===")
pprint(result.get("errors"))

print("\n=== PRODUCT DEBUG ===")

product_result = result.get("product_result") or {}
print("top-level product_candidates:", len(result.get("product_candidates") or []))

if isinstance(product_result, dict):
    print("product_result keys:", list(product_result.keys()))
    print("product_result status:", product_result.get("status"))
    print("product_result summary preview:", str(product_result.get("summary"))[:500])

    payload = product_result.get("result") or {}
    print("product_result.result type:", type(payload))
    if isinstance(payload, dict):
        print("product_result.result keys:", list(payload.keys()))
        print("payload products count:", len(payload.get("products") or []))
        print("payload product_candidates count:", len(payload.get("product_candidates") or []))
        print("payload searched_products count:", len(payload.get("searched_products") or []))
        print("payload structured_product_count:", payload.get("structured_product_count"))
        print("payload structured_product_names:", payload.get("structured_product_names"))

        tool_results = payload.get("tool_results") or []
        print("payload tool_results count:", len(tool_results))

        for i, item in enumerate(tool_results[:3], 1):
            print(f"\n[PRODUCT TOOL {i}]")
            print("type:", type(item))
            if isinstance(item, dict):
                print("keys:", list(item.keys()))
                print("tool_name:", item.get("tool_name") or item.get("name"))
                print("tool_args:", item.get("tool_args") or item.get("args"))
                print("tool_result preview:", str(item.get("tool_result") or item.get("result"))[:1000])
            else:
                print("preview:", str(item)[:1000])

agent_outputs = result.get("agent_outputs") or {}
print("\n=== AGENT OUTPUT PRODUCT ===")
print("agent_outputs keys:", list(agent_outputs.keys()))

product_agent_output = agent_outputs.get("product_agent") or {}
print("product_agent_output type:", type(product_agent_output))
if isinstance(product_agent_output, dict):
    print("product_agent_output keys:", list(product_agent_output.keys()))
    product_payload = product_agent_output.get("result") or {}
    print("product_agent_output.result type:", type(product_payload))
    if isinstance(product_payload, dict):
        print("product_agent_output.result keys:", list(product_payload.keys()))
        print("agent output product_candidates count:", len(product_payload.get("product_candidates") or []))
        print("agent output products count:", len(product_payload.get("products") or []))

print("\n=== CUSTOMER / ELIGIBILITY / FINANCIAL DEBUG ===")

customer_result = result.get("customer_result") or {}
print("top-level customer_profile:", result.get("customer_profile"))
print("top-level customer_accounts count:", len(result.get("customer_accounts") or []))

if isinstance(customer_result, dict):
    print("customer_result keys:", list(customer_result.keys()))
    customer_payload = customer_result.get("result") or {}
    print("customer_result.result type:", type(customer_payload))
    if isinstance(customer_payload, dict):
        print("customer_result.result keys:", list(customer_payload.keys()))
        print("available_monthly_saving in payload:", customer_payload.get("available_monthly_saving"))
        print("profile:", customer_payload.get("profile"))
        print("customer_profile:", customer_payload.get("customer_profile"))

eligibility_result = result.get("eligibility_result") or {}
print("\neligibility_result keys:", list(eligibility_result.keys()) if isinstance(eligibility_result, dict) else type(eligibility_result))
if isinstance(eligibility_result, dict):
    print("eligibility status:", eligibility_result.get("status"))
    print("eligibility summary:", str(eligibility_result.get("summary"))[:700])
    print("eligibility error:", eligibility_result.get("error"))
    payload = eligibility_result.get("result") or {}
    print("eligibility payload type:", type(payload))
    if isinstance(payload, dict):
        print("eligibility payload keys:", list(payload.keys()))
        print("eligibility_results count in payload:", len(payload.get("eligibility_results") or []))
        print("eligibility missing_fields:", payload.get("missing_fields"))
        print("eligibility issues:", payload.get("issues"))

financial_result = result.get("financial_result") or {}
print("\nfinancial_result keys:", list(financial_result.keys()) if isinstance(financial_result, dict) else type(financial_result))
if isinstance(financial_result, dict):
    print("financial status:", financial_result.get("status"))
    print("financial summary:", str(financial_result.get("summary"))[:700])
    print("financial error:", financial_result.get("error"))
    payload = financial_result.get("result") or {}
    print("financial payload type:", type(payload))
    if isinstance(payload, dict):
        print("financial payload keys:", list(payload.keys()))
        print("financial_results count in payload:", len(payload.get("financial_results") or []))
        print("financial missing_fields:", payload.get("missing_fields"))
        print("financial issues:", payload.get("issues"))

print("\n=== TOP-LEVEL COUNTS AFTER STATE FIX ===")
print("product_candidates:", len(result.get("product_candidates") or []))
print("eligibility_results:", len(result.get("eligibility_results") or []))
print("financial_results:", len(result.get("financial_results") or []))
print("recommendation_results:", len(result.get("recommendation_results") or []))


from pprint import pprint

print("\n=== ELIGIBILITY DETAIL ===")
for i, item in enumerate(result.get("eligibility_results") or [], 1):
    print(f"\n[ELIGIBILITY {i}]")
    pprint(item)

print("\n=== PRODUCT CANDIDATE DETAIL ===")
for i, item in enumerate(result.get("product_candidates") or [], 1):
    print(f"\n[PRODUCT {i}]")
    pprint(item)

print("\n=== FINANCIAL DETAIL ===")
for i, item in enumerate(result.get("financial_results") or [], 1):
    print(f"\n[FINANCIAL {i}]")
    print("product_name:", item.get("product_name"))
    print("status:", item.get("status"))
    print("monthly_amount:", item.get("monthly_amount"))
    print("term_months:", item.get("term_months"))
    print("applied_rate:", item.get("applied_rate"))
    print("estimated_interest:", item.get("estimated_interest"))
    print("maturity_amount:", item.get("maturity_amount"))
    print("reason:", item.get("reason"))

print("\n=== RECOMMEND / VALIDATION DEBUG ===")

recommend_result = result.get("recommend_result") or {}
print("recommend_result keys:", list(recommend_result.keys()) if isinstance(recommend_result, dict) else type(recommend_result))

if isinstance(recommend_result, dict):
    print("recommend status:", recommend_result.get("status"))
    print("recommend summary:", str(recommend_result.get("summary"))[:1000])
    print("recommend error:", recommend_result.get("error"))

    payload = recommend_result.get("result") or {}
    print("recommend payload type:", type(payload))
    if isinstance(payload, dict):
        print("recommend payload keys:", list(payload.keys()))
        print("recommendation_results count:", len(payload.get("recommendation_results") or []))
        print("recommend missing_fields:", payload.get("missing_fields"))
        print("recommend excluded_products:", payload.get("excluded_products"))

        recs = payload.get("recommendation_results") or []
        for i, rec in enumerate(recs[:3], 1):
            print(f"\n[RECOMMENDATION {i}]")
            print(rec)

validation_result = result.get("validation_result") or {}
print("\nvalidation_result keys:", list(validation_result.keys()) if isinstance(validation_result, dict) else type(validation_result))

if isinstance(validation_result, dict):
    print("validation status:", validation_result.get("status"))
    print("validation summary:", str(validation_result.get("summary"))[:1000])
    print("validation error:", validation_result.get("error"))

    payload = validation_result.get("result") or {}
    print("validation payload type:", type(payload))
    if isinstance(payload, dict):
        print("validation payload keys:", list(payload.keys()))
        print("validation is_valid:", payload.get("is_valid"))
        print("validation missing_fields:", payload.get("missing_fields"))
        print("validation issues:", payload.get("issues"))
        print("validation blocking_issues:", payload.get("blocking_issues"))
        print("validation summary:", payload.get("summary"))

print("\n=== TOP-LEVEL RECOMMENDATION RESULTS ===")
print("recommendation_results:", result.get("recommendation_results"))

print("\n=== FINAL ANSWER ===")
print(result.get("final_answer") or result["messages"][-1].content)