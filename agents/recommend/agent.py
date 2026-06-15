"""
Recommend agent.
"""
from langchain_core.messages import AIMessage

from agents.base import make_agent_result
from agents.recommend.prompts import RECOMMEND_SYSTEM_PROMPT
from agents.recommend.tools import (
    build_recommendation_summary,
    build_recommendations,
    classify_eligibility_results,
    parse_financial_results,
)
from graph.state import AgentState
from observability.langfuse import langfuse_observation, update_observation


def recommend_agent_node(state: AgentState) -> dict:
    with langfuse_observation(
        name="recommend_agent",
        as_type="span",
        input={"task_type": state.get("task_type")},
        metadata={"agent": "recommend_agent"},
    ) as observation:
        agent_outputs = dict(state.get("agent_outputs") or {})

        eligibility_results = state.get("eligibility_results") or []
        if not eligibility_results:
            eligibility_result = state.get("eligibility_result") or {}
            if isinstance(eligibility_result, dict):
                payload = eligibility_result.get("result", {})
                if isinstance(payload, dict):
                    eligibility_results = payload.get("results") or []

        financial_results = state.get("financial_results")
        if financial_results is None:
            financial_result = state.get("financial_result")
            if financial_result:
                if isinstance(financial_result, dict):
                    payload = financial_result.get("result", {})
                    tool_results = payload.get("tool_results", []) if isinstance(payload, dict) else []
                    extracted_results = []

                    for item in tool_results:
                        if not isinstance(item, dict):
                            continue
                        tool_result = item.get("tool_result")
                        if isinstance(tool_result, str):
                            extracted_results.extend(parse_financial_results(tool_result))

                    financial_results = extracted_results
                else:
                    financial_results = parse_financial_results(financial_result)
            else:
                financial_agent_output = agent_outputs.get("financial_agent", "")
                if isinstance(financial_agent_output, dict):
                    payload = financial_agent_output.get("result", {})
                    tool_results = payload.get("tool_results", []) if isinstance(payload, dict) else []
                    extracted_results = []
                    for item in tool_results:
                        if not isinstance(item, dict):
                            continue
                        tool_result = item.get("tool_result")
                        if isinstance(tool_result, str):
                            extracted_results.extend(parse_financial_results(tool_result))
                    financial_results = extracted_results
                else:
                    financial_results = parse_financial_results(financial_agent_output)

            if not isinstance(financial_results, list):
                financial_results = parse_financial_results(financial_results)

        classified = classify_eligibility_results(eligibility_results)
        recommendations = build_recommendations(classified["recommendable"], financial_results)
        summary = build_recommendation_summary(
            recommendations,
            classified["needs_check"],
            classified["rejected"],
        )

        recommend_result = make_agent_result(
            status="success",
            result={
                "summary": summary,
                "recommendations": recommendations,
                "recommended_products": recommendations,
                "recommendation_count": len(recommendations),
                "needs_check_products": classified.get("needs_check", []),
                "rejected_products": classified.get("rejected", []),
                "financial_results": financial_results,
            },
            evidence=recommendations,
            error=None,
        )
        agent_outputs["recommend_agent"] = recommend_result

        update_observation(
            observation,
            output={
                "summary_preview": summary[:500],
                "recommendation_count": len(recommendations),
                "top_recommendations": [
                    {
                        "product_name": item.get("product_name"),
                        "score": item.get("score"),
                        "reason": (item.get("reason") or "")[:200],
                    }
                    for item in recommendations[:5]
                    if isinstance(item, dict)
                ],
                "needs_check_products": [
                    item.get("product_name")
                    for item in classified.get("needs_check", [])[:5]
                    if isinstance(item, dict)
                ],
                "rejected_products": [
                    item.get("product_name")
                    for item in classified.get("rejected", [])[:5]
                    if isinstance(item, dict)
                ],
            },
            metadata={
                "agent": "recommend_agent",
                "task_type": state.get("task_type"),
            },
        )

        completed_agents = list(state.get("completed_agents") or [])
        if "recommend_agent" not in completed_agents:
            completed_agents.append("recommend_agent")

        return {
            "messages": [AIMessage(content=summary)],
            "agent_outputs": agent_outputs,
            "current_step": (state.get("current_step") or 0) + 1,
            "current_agent": "recommend_agent",
            "completed_agents": completed_agents,
            "financial_results": financial_results,
            "recommendation_results": recommendations,
            "recommend_result": recommend_result,
            "context": {
                **(state.get("context") or {}),
                "recommend_prompt": RECOMMEND_SYSTEM_PROMPT,
            },
        }
