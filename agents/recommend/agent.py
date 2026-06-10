"""
Recommend agent - rank only eligible products with no pending checks.
"""
from langchain_core.messages import AIMessage

from agents.recommend.prompts import RECOMMEND_SYSTEM_PROMPT
from agents.recommend.tools import (
    build_recommendation_summary,
    build_recommendations,
    classify_eligibility_results,
    parse_financial_results,
)
from graph.state import AgentState


def recommend_agent_node(state: AgentState) -> dict:
    agent_outputs = dict(state.get("agent_outputs") or {})

    eligibility_results = state.get("eligibility_results") or []
    financial_results = state.get("financial_results")
    if financial_results is None:
        financial_results = parse_financial_results(agent_outputs.get("financial_agent", ""))

    classified = classify_eligibility_results(eligibility_results)
    recommendations = build_recommendations(classified["recommendable"], financial_results)
    summary = build_recommendation_summary(
        recommendations,
        classified["needs_check"],
        classified["rejected"],
    )
    agent_outputs["recommend_agent"] = summary

    return {
        "messages": [AIMessage(content=summary)],
        "agent_outputs": agent_outputs,
        "current_step": (state.get("current_step") or 0) + 1,
        "financial_results": financial_results,
        "recommendation_results": recommendations,
        "context": {
            **(state.get("context") or {}),
            "recommend_prompt": RECOMMEND_SYSTEM_PROMPT,
        },
    }
