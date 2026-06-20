"""
Compare two saved run result JSON files from different branches.

Expected input:
- a full state/result JSON that includes customer_result, product_result,
  eligibility_result, recommend_result, final_answer
- or a reduced JSON snapshot with the same top-level keys
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _get_nested(data: dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _normalize_name(value: Any) -> str:
    return "".join(str(value or "").lower().split())


def _extract_metrics(data: dict[str, Any]) -> dict[str, Any]:
    product_perf = _get_nested(data, "product_result", "result", "performance") or data.get("product_performance") or {}
    customer_profile = _get_nested(data, "customer_result", "result", "customer_profile") or data.get("customer_profile") or {}
    eligibility_results = _get_nested(data, "eligibility_result", "result", "results") or data.get("eligibility_results") or []
    recommendations = _get_nested(data, "recommend_result", "result", "recommended_products") or []
    final_answer = data.get("final_answer") or _get_nested(data, "recommend_result", "result", "summary") or ""

    eligible_names = [
        item.get("product_name")
        for item in eligibility_results
        if isinstance(item, dict) and item.get("product_name")
    ]
    recommended_names = [
        item.get("product_name")
        for item in recommendations
        if isinstance(item, dict) and item.get("product_name")
    ]

    hallucinated_names = [
        name
        for name in recommended_names
        if _normalize_name(name) not in {_normalize_name(item) for item in eligible_names}
    ]

    customer_profile_accuracy = {
        "job": customer_profile.get("job"),
        "salary_transfer": customer_profile.get("salary_transfer"),
        "monthly_saving_amount": customer_profile.get("monthly_saving_amount"),
        "is_soldier": customer_profile.get("is_soldier"),
    }

    return {
        "total_execution_time_ms": data.get("total_execution_time_ms"),
        "product_agent_execution_time_ms": product_perf.get("total_duration_ms"),
        "tool_count": product_perf.get("tool_count"),
        "product_candidate_count": product_perf.get("product_candidate_count"),
        "final_products_count": product_perf.get("final_products_count"),
        "recommendation_count": len(recommended_names),
        "eligibility_product_names": eligible_names,
        "customer_profile_accuracy": customer_profile_accuracy,
        "final_answer_preview": str(final_answer)[:300],
        "hallucination_detected": bool(hallucinated_names),
        "hallucinated_product_names": hallucinated_names,
    }


def _print_block(title: str, metrics: dict[str, Any]) -> None:
    print(f"\n[{title}]")
    for key, value in metrics.items():
        print(f"{key}: {json.dumps(value, ensure_ascii=False)}")


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("Usage: python scripts/compare_product_agent_runs.py <main_run.json> <branch_run.json>")
        return 1

    left_path, right_path = argv[1], argv[2]
    left = _extract_metrics(_load_json(left_path))
    right = _extract_metrics(_load_json(right_path))

    _print_block(Path(left_path).name, left)
    _print_block(Path(right_path).name, right)

    print("\n[Comparison]")
    for key in left.keys():
        print(
            f"{key}: "
            f"left={json.dumps(left.get(key), ensure_ascii=False)} | "
            f"right={json.dumps(right.get(key), ensure_ascii=False)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
