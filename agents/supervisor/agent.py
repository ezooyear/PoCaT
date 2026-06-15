"""
Supervisor 에이전트
- 질문 의도 분석
- 실행 계획 생성
- Agent 순서 결정
- State 확인
- Validation 결과 반영
- 최종 답변 생성

# 구조:
START
→ supervisor(plan_mode)
→ plan 순서대로 agent 실행
→ 마지막 agent 실행 후 supervisor(synthesize_mode)
→ END

주의:
- Supervisor는 직접 고객 조회/상품 조회/계산/추천/검증을 수행하지 않는다.
- 각 전문 Agent는 state에 자신의 결과를 저장한다.
- Supervisor는 마지막에 state를 확인하고 validation_result까지 반영하여 final_answer를 생성한다.
"""

import json
from typing import Any

from langchain_core.messages import SystemMessage, AIMessage

from config.settings import get_llm
from graph.state import AgentState
from observability.langfuse import langfuse_observation, update_observation
from agents.supervisor.prompts import (
    SUPERVISOR_PLAN_PROMPT,
    SUPERVISOR_SYNTHESIZE_PROMPT,
    SUPERVISOR_DIRECT_RESPONSE_PROMPT,
)


AGENT_OPTIONS = [
    "customer_agent",
    "product_agent",
    "financial_agent",
    "eligibility_agent",
    "recommend_agent",
    "validation_agent",
    "FINISH",
]

TASK_TYPES = [
    "casual",
    "customer_lookup",
    "product_info",
    "financial_analysis",
    "eligibility_check",
    "recommendation",
    "early_termination",
    "switch_analysis",
]

MAX_PLAN_LENGTH = 6


def supervisor_node(state: AgentState) -> dict:
    """
    Supervisor 메인 노드.

    - 최초 호출: plan이 없으므로 _plan_mode 실행
    - 마지막 호출: plan이 있으므로 _synthesize_mode 실행

    이 함수는 방식 A를 전제로 한다.
    즉 builder에서 마지막 agent 실행 후 supervisor로 다시 연결되어야 한다.
    """

    plan = state.get("plan") or []

    if not plan:
        return _plan_mode(state)

    # 일반 대화가 이미 직접 응답으로 끝난 경우
    if plan == ["FINISH"]:
        return {
            "next": "FINISH",
            "final_answer": state.get("final_answer"),
        }

    return _synthesize_mode(state)


def _plan_mode(state: AgentState) -> dict:
    """
    사용자 질문을 분석해 task_type과 plan을 생성한다.
    규칙 기반 routing을 먼저 적용하고, 일반 대화/애매한 질문은 LLM routing으로 보완한다.
    """

    state_messages = list(state.get("messages", []))
    user_query = state.get("user_query") or _get_last_user_text(state_messages)
    normalized_query = (user_query or "").lower().replace(" ", "")

    # 1차: 규칙 기반 routing
    if (
        "추천" in normalized_query
        or "recommend" in normalized_query
        or "추천상품" in normalized_query
        or "가입불가상품" in normalized_query
        or "추가확인필요상품" in normalized_query
    ):
        routing = {
            "task_type": "recommendation",
            "plan": [
                "customer_agent",
                "product_agent",
                "eligibility_agent",
                "recommend_agent",
                "validation_agent",
            ],
        }
    else:
        routing = _fallback_routing(user_query)

    # 2차: 규칙으로 일반 대화로만 잡힌 경우 LLM에게 판단 요청
    # 단, 진짜 인사/감사 표현이면 direct response로 처리된다.
    if routing.get("task_type") == "casual":
        llm_routing = _llm_plan_routing(state_messages)
        normalized = _normalize_routing(llm_routing, user_query)

        # LLM이 유의미한 plan을 만들었으면 사용
        if normalized.get("task_type") != "casual":
            routing = normalized
        else:
            routing = _normalize_routing(routing, user_query)
    else:
        routing = _normalize_routing(routing, user_query)

    task_type = routing.get("task_type", "casual")
    plan = routing.get("plan", ["FINISH"])

    if task_type == "casual" or plan == ["FINISH"]:
        return _direct_response(state, user_query)

    next_agent = plan[0]

    return {
        "user_query": user_query,
        "task_type": task_type,
        "plan": plan,
        "next": next_agent,
        "current_step": 0,
        "current_agent": next_agent,
        "completed_agents": [],
        "agent_outputs": state.get("agent_outputs") or {},
        "agent_logs": _append_log(
            state,
            {
                "agent": "supervisor",
                "mode": "plan",
                "task_type": task_type,
                "plan": plan,
                "next": next_agent,
            },
        ),
    }


def _synthesize_mode(state: AgentState) -> dict:
    """
    모든 Agent 실행 후 State를 종합하여 최종 답변을 생성한다.
    agent_outputs뿐 아니라 구조화된 개별 result 필드를 함께 사용한다.
    """

    user_query = state.get("user_query") or _get_last_user_text(state.get("messages", []))
    final_context = _build_final_context(state, user_query)

    deterministic_answer = _build_deterministic_final_answer(final_context)
    if deterministic_answer:
        return {
            "messages": [AIMessage(content=deterministic_answer)],
            "draft_answer": deterministic_answer,
            "final_answer": deterministic_answer,
            "next": "FINISH",
            "current_agent": "supervisor",
            "current_step": len(state.get("plan") or []),
            "agent_logs": _append_log(
                state,
                {
                    "agent": "supervisor",
                    "mode": "synthesize",
                    "task_type": state.get("task_type"),
                    "used_validation": bool(state.get("validation_result")),
                    "deterministic": True,
                },
            ),
        }

    llm = get_llm()
    agent_results = _format_agent_results(final_context)

    prompt = SUPERVISOR_SYNTHESIZE_PROMPT.format(agent_results=agent_results)
    messages = [SystemMessage(content=prompt)] + list(state.get("messages", []))

    try:
        response = llm.invoke(messages)
        final_answer = _clean_text(response.content)

    except Exception as e:
        final_answer = _fallback_final_answer(final_context, error=e)

    return {
        "messages": [AIMessage(content=final_answer)],
        "draft_answer": final_answer,
        "final_answer": final_answer,
        "next": "FINISH",
        "current_agent": "supervisor",
        "current_step": len(state.get("plan") or []),
        "agent_logs": _append_log(
            state,
            {
                "agent": "supervisor",
                "mode": "synthesize",
                "task_type": state.get("task_type"),
                "used_validation": bool(state.get("validation_result")),
            },
        ),
    }


def _build_deterministic_final_answer_legacy(final_context: dict) -> str:
    task_type = final_context.get("task_type")
    if task_type != "recommendation":
        return ""

    recommend_result = final_context.get("recommend_result") or {}
    if not isinstance(recommend_result, dict):
        return ""

    payload = recommend_result.get("result", {})
    if not isinstance(payload, dict):
        return ""

    recommendations = payload.get("recommendations") or []
    needs_check = payload.get("needs_check_products") or []
    rejected = payload.get("rejected_products") or []

    if not recommendations and not needs_check and not rejected:
        return ""

    lines = ["추천 결과를 정리해 드립니다.", ""]

    if recommendations:
        lines.append("추천 상품")
        for index, item in enumerate(recommendations, start=1):
            if not isinstance(item, dict):
                continue
            product_name = str(item.get("product_name") or "미확인 상품").strip()
            reason = str(item.get("reason") or "가입 가능, 기본 조건 충돌 없음").strip()
            score = item.get("score")
            interest = item.get("estimated_interest")

            line = f"{index}. {product_name}"
            if score is not None:
                line += f" (점수 {score})"
            lines.append(line)
            lines.append(f"추천 이유: {reason}")
            if interest is not None:
                try:
                    lines.append(f"예상 이자 참고: {int(interest):,}원")
                except (TypeError, ValueError):
                    pass

    if needs_check:
        lines.append("")
        lines.append("추가 확인 필요 상품")
        for item in needs_check:
            if not isinstance(item, dict):
                continue
            product_name = str(item.get("product_name") or "미확인 상품").strip()
            reasons = item.get("check_required") or []
            reason_text = ", ".join(str(reason) for reason in reasons) if reasons else "세부 조건 확인 필요"
            lines.append(f"- {product_name}: {reason_text}")

    if rejected:
        lines.append("")
        lines.append("가입 불가 또는 제외 상품")
        for item in rejected:
            if not isinstance(item, dict):
                continue
            product_name = str(item.get("product_name") or "미확인 상품").strip()
            reasons = item.get("ineligibility_reasons") or []
            reason_text = ", ".join(str(reason) for reason in reasons) if reasons else "가입 불가"
            lines.append(f"- {product_name}: {reason_text}")

    lines.append("")
    lines.append("실제 적용 금리와 가입 가능 여부는 거래 시점, 고객 조건, 은행 정책에 따라 달라질 수 있습니다.")
    lines.append("정확한 내용은 상품설명서 또는 은행 상담을 통해 최종 확인하는 것이 좋습니다.")
    return "\n".join(lines)


def _build_deterministic_final_answer(final_context: dict) -> str:
    task_type = final_context.get("task_type")
    if task_type == "switch_analysis":
        return _build_switch_analysis_answer(final_context)
    if task_type != "recommendation":
        return ""
    user_query = str(final_context.get("user_query") or "")
    if all(keyword in user_query for keyword in ["추천 상품", "추가 확인", "가입 불가"]):
        bucketed_answer = _build_bucketed_recommendation_answer(final_context)
        if bucketed_answer:
            return bucketed_answer

    recommend_result = final_context.get("recommend_result") or {}
    if not isinstance(recommend_result, dict):
        return ""

    payload = recommend_result.get("result", {})
    if not isinstance(payload, dict):
        return ""

    recommendations = payload.get("recommendations") or []
    needs_check = payload.get("needs_check_products") or []
    rejected = payload.get("rejected_products") or []

    if not recommendations and not needs_check and not rejected:
        return ""

    filtered_recommendations = [item for item in recommendations if isinstance(item, dict)]
    filtered_needs_check = [item for item in needs_check if isinstance(item, dict)]
    filtered_rejected = [item for item in rejected if isinstance(item, dict)]

    lines = ["추천 결과를 정리해 드립니다."]

    if filtered_recommendations:
        top = filtered_recommendations[0]
        top_name = str(top.get("product_name") or "미확인 상품").strip()
        top_reason = str(top.get("reason") or "가입 가능하고 기본 조건 충돌이 없습니다.").strip()
        top_score = top.get("score")

        lines.append("")
        lines.append("가장 적합한 상품")
        if top_score is not None:
            lines.append(f"- {top_name} (점수 {top_score})")
        else:
            lines.append(f"- {top_name}")
        lines.append(f"- 추천 이유: {top_reason}")

        if top.get("estimated_interest") is not None:
            try:
                lines.append(f"- 예상 이자: {int(top['estimated_interest']):,}원")
            except (TypeError, ValueError):
                pass

        remaining = filtered_recommendations[1:4]
        if remaining:
            lines.append("")
            lines.append("함께 볼 상품")
            for item in remaining:
                product_name = str(item.get("product_name") or "미확인 상품").strip()
                reason = str(item.get("reason") or "가입 가능").strip()
                score = item.get("score")
                line = f"- {product_name}"
                if score is not None:
                    line += f" (점수 {score})"
                lines.append(line)
                lines.append(f"  이유: {reason}")

    if filtered_needs_check:
        lines.append("")
        lines.append("추가 확인 필요 상품")
        for item in filtered_needs_check[:5]:
            product_name = str(item.get("product_name") or "미확인 상품").strip()
            reasons = item.get("check_required") or []
            reason_text = ", ".join(str(reason) for reason in reasons) if reasons else "상세 조건 확인 필요"
            lines.append(f"- {product_name}: {reason_text}")

    if filtered_rejected:
        lines.append("")
        lines.append("가입 불가 또는 제외 상품")
        for item in filtered_rejected[:5]:
            product_name = str(item.get("product_name") or "미확인 상품").strip()
            reasons = item.get("ineligibility_reasons") or []
            reason_text = ", ".join(str(reason) for reason in reasons) if reasons else "가입 불가"
            lines.append(f"- {product_name}: {reason_text}")

    lines.append("")
    lines.append("실제 적용 금리와 가입 가능 여부는 거래 시점, 고객 조건, 은행 정책에 따라 달라질 수 있습니다.")
    lines.append("정확한 내용은 상품설명서 또는 은행 상담을 통해 최종 확인하는 것이 좋습니다.")
    return "\n".join(lines)


def _build_deterministic_final_answer(final_context: dict) -> str:
    task_type = final_context.get("task_type")
    if task_type == "switch_analysis":
        return _build_switch_analysis_answer(final_context)
    if task_type != "recommendation":
        return ""

    user_query = str(final_context.get("user_query") or "")
    if all(keyword in user_query for keyword in ["추천 상품", "추가 확인", "가입 불가"]):
        bucketed_answer = _build_bucketed_recommendation_answer(final_context)
        if bucketed_answer:
            return bucketed_answer

    recommend_result = final_context.get("recommend_result") or {}
    if not isinstance(recommend_result, dict):
        return ""

    payload = recommend_result.get("result", {})
    if not isinstance(payload, dict):
        return ""

    recommendations = [item for item in (payload.get("recommendations") or []) if isinstance(item, dict)]
    needs_check = [item for item in (payload.get("needs_check_products") or []) if isinstance(item, dict)]
    rejected = [item for item in (payload.get("rejected_products") or []) if isinstance(item, dict)]

    if not recommendations and not needs_check and not rejected:
        return ""

    lines = ["추천 결과를 정리해 드립니다."]

    if recommendations:
        lines.append("")
        lines.append("가장 적합한 상품")
        lines.extend(_format_recommendation_block(recommendations[0]))

        remaining = recommendations[1:4]
        if remaining:
            lines.append("")
            lines.append("함께 볼 상품")
            for item in remaining:
                lines.extend(_format_compact_recommendation_block(item))

    if needs_check:
        lines.append("")
        lines.append("추가 확인 필요 상품")
        for item in needs_check[:3]:
            name = str(item.get("product_name") or "미확인 상품").strip()
            reasons = [str(reason).strip() for reason in (item.get("check_required") or []) if str(reason).strip()]
            reason_text = ", ".join(reasons) if reasons else "상세 조건 확인 필요"
            lines.append(f"- {name}")
            lines.append(f"  확인할 항목: {reason_text}")

    if rejected:
        lines.append("")
        lines.append("가입 불가 또는 제외 상품")
        for item in rejected[:3]:
            name = str(item.get("product_name") or "미확인 상품").strip()
            reasons = [str(reason).strip() for reason in (item.get("ineligibility_reasons") or []) if str(reason).strip()]
            reason_text = ", ".join(reasons) if reasons else "가입 불가"
            lines.append(f"- {name}")
            lines.append(f"  제외 이유: {reason_text}")

    lines.append("")
    lines.append("실제 적용 금리와 가입 가능 여부는 거래 시점, 고객 조건, 은행 정책에 따라 달라질 수 있습니다.")
    lines.append("정확한 내용은 상품설명서 또는 은행 상담을 통해 최종 확인하는 것이 좋습니다.")
    return "\n".join(lines)


def _format_recommendation_block(item: dict) -> list[str]:
    name = str(item.get("product_name") or "미확인 상품").strip()
    score = item.get("score")
    reason = str(item.get("reason") or "가입 가능하고 기본 조건 충돌이 없습니다.").strip()

    lines = [f"- 상품명: {name}"]
    if score is not None:
        lines.append(f"  추천 점수: {score}")
    lines.append(f"  추천 이유: {reason}")

    estimated_interest = item.get("estimated_interest")
    if estimated_interest is not None:
        try:
            lines.append(f"  예상 이자: {int(estimated_interest):,}원")
        except (TypeError, ValueError):
            pass

    maturity_amount = item.get("maturity_amount")
    if maturity_amount is not None:
        try:
            lines.append(f"  만기 금액 참고: {int(maturity_amount):,}원")
        except (TypeError, ValueError):
            pass

    switch_comparison = str(item.get("switch_comparison") or "").strip()
    if switch_comparison:
        lines.append(f"  참고: {switch_comparison}")

    return lines


def _format_compact_recommendation_block(item: dict) -> list[str]:
    name = str(item.get("product_name") or "미확인 상품").strip()
    score = item.get("score")
    reason = str(item.get("reason") or "가입 가능").strip()

    label = f"- {name}"
    if score is not None:
        label += f" (점수 {score})"

    return [label, f"  이유: {reason}"]


def _build_switch_analysis_answer(final_context: dict) -> str:
    financial_result = final_context.get("financial_result") or {}
    recommend_result = final_context.get("recommend_result") or {}

    tool_results = []
    if isinstance(financial_result, dict):
        payload = financial_result.get("result", {})
        if isinstance(payload, dict):
            raw_tool_results = payload.get("tool_results", [])
            if isinstance(raw_tool_results, list):
                tool_results = raw_tool_results

    switch_result = ""
    for item in tool_results:
        if isinstance(item, dict) and item.get("tool_name") == "compare_switch_benefit":
            switch_result = str(item.get("tool_result") or "").strip()
            if switch_result:
                break

    if not switch_result:
        return ""

    lines = ["갈아타기 분석 결과를 정리해 드립니다.", "", switch_result]

    if isinstance(recommend_result, dict):
        payload = recommend_result.get("result", {})
        if isinstance(payload, dict):
            recommendations = payload.get("recommendations") or []
            filtered = [item for item in recommendations if isinstance(item, dict)]
            if filtered:
                top = filtered[0]
                lines.extend(
                    [
                        "",
                        "추천 방향",
                        f"- {str(top.get('product_name') or '미확인 상품').strip()}",
                        f"- 이유: {str(top.get('reason') or '비교 결과를 참고해 판단하는 것이 좋습니다.').strip()}",
                    ]
                )

    return "\n".join(lines)


def _build_bucketed_recommendation_answer(final_context: dict) -> str:
    recommend_result = final_context.get("recommend_result") or {}
    if not isinstance(recommend_result, dict):
        return ""

    payload = recommend_result.get("result", {})
    if not isinstance(payload, dict):
        return ""

    recommendations = [item for item in (payload.get("recommendations") or []) if isinstance(item, dict)]
    needs_check = [item for item in (payload.get("needs_check_products") or []) if isinstance(item, dict)]
    rejected = [item for item in (payload.get("rejected_products") or []) if isinstance(item, dict)]

    if not recommendations and not needs_check and not rejected:
        return ""

    lines = ["요청하신 기준으로 상품을 나눠서 정리해 드립니다."]

    lines.append("")
    lines.append("추천 상품")
    if recommendations:
        for item in recommendations[:5]:
            name = str(item.get("product_name") or "미확인 상품").strip()
            score = item.get("score")
            reason = str(item.get("reason") or "가입 가능").strip()
            line = f"- {name}"
            if score is not None:
                line += f" (점수 {score})"
            lines.append(line)
            lines.append(f"  이유: {reason}")
    else:
        lines.append("- 해당 없음")

    lines.append("")
    lines.append("추가 확인 필요 상품")
    if needs_check:
        for item in needs_check[:5]:
            name = str(item.get("product_name") or "미확인 상품").strip()
            reason_text = ", ".join(str(reason) for reason in (item.get("check_required") or [])) or "상세 조건 확인 필요"
            lines.append(f"- {name}: {reason_text}")
    else:
        lines.append("- 해당 없음")

    lines.append("")
    lines.append("가입 불가 또는 제외 상품")
    if rejected:
        for item in rejected[:5]:
            name = str(item.get("product_name") or "미확인 상품").strip()
            reason_text = ", ".join(str(reason) for reason in (item.get("ineligibility_reasons") or [])) or "가입 불가"
            lines.append(f"- {name}: {reason_text}")
    else:
        lines.append("- 해당 없음")

    lines.append("")
    lines.append("실제 적용 금리와 가입 가능 여부는 거래 시점, 고객 조건, 은행 정책에 따라 달라질 수 있습니다.")
    lines.append("정확한 내용은 상품설명서 또는 은행 상담을 통해 최종 확인하는 것이 좋습니다.")
    return "\n".join(lines)


def _llm_plan_routing(messages: list) -> dict:
    """
    LLM 기반 plan 생성.
    """

    llm = get_llm()

    try:
        request_messages = [SystemMessage(content=SUPERVISOR_PLAN_PROMPT)] + messages
        response = llm.invoke(request_messages)
        return _parse_plan_response(response.content.strip())

    except Exception:
        return {
            "task_type": "casual",
            "plan": ["FINISH"],
        }


def _parse_plan_response(text: str) -> dict:
    """
    LLM 응답에서 task_type과 plan을 파싱한다.
    """

    cleaned = _extract_json_text(text)

    try:
        data = json.loads(cleaned)

    except json.JSONDecodeError:
        return _parse_plan_from_text(text)

    if isinstance(data, str):
        nested = _extract_json_text(data)
        if nested != data:
            try:
                data = json.loads(nested)
            except json.JSONDecodeError:
                return _parse_plan_from_text(data)
        else:
            return _parse_plan_from_text(data)

    if not isinstance(data, dict):
        return _parse_plan_from_text(text)

    task_type = data.get("task_type", "casual")
    plan = data.get("plan", [])

    return {
        "task_type": task_type,
        "plan": plan,
    }


def _normalize_routing(routing: dict, user_query: str) -> dict:
    """
    LLM 또는 규칙 기반 routing 결과를 현재 Agent 구조에 맞게 보정한다.
    """

    if not isinstance(routing, dict):
        return _fallback_routing(user_query)

    task_type = routing.get("task_type", "casual")
    plan = routing.get("plan", [])

    if task_type not in TASK_TYPES:
        task_type = "casual"

    if not isinstance(plan, list):
        plan = []

    converted_plan = []

    for agent in plan:
        if agent == "analysis_agent":
            converted_plan.extend(["customer_agent", "financial_agent"])
        elif agent == "calculation_agent":
            converted_plan.append("financial_agent")
        elif agent == "supervisor_final":
            continue
        else:
            converted_plan.append(agent)

    valid_plan = []

    for agent in converted_plan:
        if agent in AGENT_OPTIONS and agent not in valid_plan:
            valid_plan.append(agent)

    if not valid_plan:
        return _fallback_routing(user_query)

    # FINISH는 단독 사용만 허용
    if "FINISH" in valid_plan and len(valid_plan) > 1:
        valid_plan = [agent for agent in valid_plan if agent != "FINISH"]

    if valid_plan == ["FINISH"]:
        return {
            "task_type": "casual",
            "plan": ["FINISH"],
        }

    # task_type별 표준 실행 계획으로 보정
    valid_plan = _ensure_required_agents(task_type, valid_plan)

    # validation_agent는 필요한 task_type에서 마지막 위치 보장
    valid_plan = _ensure_validation_position(task_type, valid_plan)

    valid_plan = valid_plan[:MAX_PLAN_LENGTH]

    return {
        "task_type": task_type,
        "plan": valid_plan,
    }


def _ensure_required_agents(task_type: str, plan: list[str]) -> list[str]:
    """
    task_type에 따라 표준 실행 계획을 보장한다.
    LLM이 일부 Agent를 누락하거나 순서를 잘못 반환해도 역할표 기준으로 보정한다.
    """

    standard_plans = {
        "customer_lookup": [
            "customer_agent",
        ],
        "product_info": [
            "product_agent",
        ],
        "financial_analysis": [
            "customer_agent",
            "financial_agent",
            "validation_agent",
        ],
        "eligibility_check": [
            "customer_agent",
            "product_agent",
            "eligibility_agent",
            "validation_agent",
        ],
        "recommendation": [
            "customer_agent",
            "product_agent",
            "eligibility_agent",
            "recommend_agent",
            "validation_agent",
        ],
        "early_termination": [
            "customer_agent",
            "product_agent",
            "financial_agent",
            "recommend_agent",
            "validation_agent",
        ],
        "switch_analysis": [
            "customer_agent",
            "product_agent",
            "eligibility_agent",
            "financial_agent",
            "recommend_agent",
            "validation_agent",
        ],
    }

    # 단순 가상 계산처럼 customer_agent가 필요 없는 경우 예외 허용
    if task_type == "financial_analysis" and plan == ["financial_agent", "validation_agent"]:
        return plan

    return standard_plans.get(task_type, plan)


def _ensure_validation_position(task_type: str, plan: list[str]) -> list[str]:
    """
    검증이 필요한 task_type에서는 validation_agent를 마지막에 둔다.
    """

    need_validation = {
        "financial_analysis",
        "eligibility_check",
        "recommendation",
        "early_termination",
        "switch_analysis",
    }

    if task_type not in need_validation:
        return plan

    plan = [agent for agent in plan if agent != "validation_agent"]
    plan.append("validation_agent")

    return plan


def _fallback_routing(user_query: str) -> dict:
    """
    규칙 기반 routing.
    고객 DB가 필요한 질문은 LLM보다 규칙 기반으로 먼저 잡는다.
    """

    query = (user_query or "").lower().replace(" ", "")

    def has_any(keywords: list[str]) -> bool:
        return any(keyword and keyword in query for keyword in keywords)

    early_termination_keywords = [
        "중도해지",
        "해지손실",
        "해지하면",
        "해지했을때",
        "해지할때",
        "손실",
        "급전",
        "깨면",
        "깨도",
        "해지해도",
    ]
    switch_keywords = [
        "갈아타기",
        "갈아타는게",
        "갈아탈까",
        "바꾸는게",
        "바꿀까",
        "유지하는게",
        "새상품",
        "금리높은상품",
        "더높은금리",
        "이득이야",
        "이득인지",
    ]
    recommendation_keywords = [
        "추천",
        "recommend",
        "추천상품",
        "추가확인가능상품",
        "추가확인필요상품",
        "가입불가상품",
        "추가가입",
        "가입할만한",
        "뭐가좋아",
        "뭐가나아",
        "어떤상품이좋아",
        "나한테맞는",
        "내조건에맞는",
        "목적에맞는",
        "가입가능한상품중",
    ]
    eligibility_keywords = [
        "가입가능",
        "가입할수",
        "가입돼",
        "가입되",
        "가입할수있",
        "조건충족",
        "우대조건충족",
        "우대금리받을수",
        "우대금리받",
        "가능한상품",
        "가입가능한상품",
        "필터링",
    ]
    financial_keywords = [
        "납입",
        "납입현황",
        "몇번냈",
        "몇회",
        "만기",
        "이자",
        "잔액",
        "남은기간",
        "남은납입",
        "얼마받",
        "만기때",
        "수령액",
        "예상금액",
    ]
    customer_lookup_keywords = [
        "내가입",
        "가입상품",
        "내계좌",
        "고객정보",
        "내정보",
        "보유상품",
        "가입목록",
        "내상품",
        "내적금",
        "내예금",
        "가입한상품",
    ]
    product_info_keywords = [
        "약관",
        "우대",
        "우대금리",
        "예금자보호",
        "금리조건",
        "가입조건",
        "가입제한",
        "유의사항",
        "중도해지유의사항",
        "상품설명서",
        "pdf",
        "rag",
        "금리알려",
        "금리조회",
        "상품정보",
        "조건알려",
    ]
    simple_calc_keywords = [
        "단리",
        "복리",
        "계산해줘",
        "계산하면",
    ]

    # 1. 중도해지
    if has_any(early_termination_keywords):
        return {
            "task_type": "early_termination",
            "plan": [
                "customer_agent",
                "product_agent",
                "financial_agent",
                "recommend_agent",
                "validation_agent",
            ],
        }

    # 2. 갈아타기
    if has_any(switch_keywords):
        return {
            "task_type": "switch_analysis",
            "plan": [
                "customer_agent",
                "financial_agent",
                "product_agent",
                "eligibility_agent",
                "recommend_agent",
                "validation_agent",
            ],
        }

    # 3. 추천
    if has_any(recommendation_keywords):
        return {
            "task_type": "recommendation",
            "plan": [
                "customer_agent",
                "product_agent",
                "eligibility_agent",
                "recommend_agent",
                "validation_agent",
            ],
        }

    # 4. 가입 가능 여부 / 우대금리 충족 여부
    if has_any(eligibility_keywords):
        return {
            "task_type": "eligibility_check",
            "plan": [
                "customer_agent",
                "product_agent",
                "eligibility_agent",
                "validation_agent",
            ],
        }

    # 5. 고객 기준 계산/납입/잔액/만기
    if has_any(financial_keywords):
        return {
            "task_type": "financial_analysis",
            "plan": [
                "customer_agent",
                "financial_agent",
                "validation_agent",
            ],
        }

    # 6. 고객 조회
    # '상품'이라는 단어가 있어도 '내 가입 상품'이면 product_agent가 아니라 customer_agent로 가야 함.
    if has_any(customer_lookup_keywords):
        return {
            "task_type": "customer_lookup",
            "plan": [
                "customer_agent",
            ],
        }

    # 7. 단순 상품/약관/RAG
    # '상품' 단독 키워드는 넣지 않음.
    if has_any(product_info_keywords):
        return {
            "task_type": "product_info",
            "plan": [
                "product_agent",
            ],
        }

    # 8. 단순 가상 계산
    if has_any(simple_calc_keywords):
        return {
            "task_type": "financial_analysis",
            "plan": [
                "financial_agent",
                "validation_agent",
            ],
        }

    # 9. 일반 대화
    return {
        "task_type": "casual",
        "plan": [
            "FINISH",
        ],
    }


def _direct_response(state: AgentState, user_query: str) -> dict:
    """
    일반 대화에 대해 Supervisor가 직접 응답한다.
    """

    llm = get_llm()

    try:
        messages = [
            SystemMessage(content=SUPERVISOR_DIRECT_RESPONSE_PROMPT),
        ] + list(state.get("messages", []))

        response = llm.invoke(messages)
        answer = _clean_text(response.content)

    except Exception:
        answer = (
            "안녕하세요. 예적금 상담을 도와드릴게요.\n"
            "가입 상품 조회, 금리 비교, 상품 추천, 이자 계산, 만기 확인 등을 물어보실 수 있습니다."
        )

    return {
        "messages": [AIMessage(content=answer)],
        "user_query": user_query,
        "task_type": "casual",
        "plan": ["FINISH"],
        "next": "FINISH",
        "current_step": 1,
        "current_agent": "supervisor",
        "completed_agents": [],
        "agent_outputs": state.get("agent_outputs") or {},
        "draft_answer": answer,
        "final_answer": answer,
        "agent_logs": _append_log(
            state,
            {
                "agent": "supervisor",
                "mode": "direct_response",
                "task_type": "casual",
            },
        ),
    }


def _build_final_context(state: AgentState, user_query: str) -> dict:
    """
    최종 답변 생성을 위한 내부 context를 구성한다.
    """

    return {
        "user_query": user_query,
        "task_type": state.get("task_type"),
        "member_id": state.get("member_id"),
        "customer_id": state.get("customer_id"),
        "plan": state.get("plan"),
        "current_step": state.get("current_step"),
        "current_agent": state.get("current_agent"),
        "completed_agents": state.get("completed_agents"),
        "agent_outputs": state.get("agent_outputs"),
        "customer_result": state.get("customer_result"),
        "product_result": state.get("product_result"),
        "financial_result": state.get("financial_result"),
        "eligibility_result": state.get("eligibility_result"),
        "recommend_result": state.get("recommend_result"),
        "validation_result": state.get("validation_result"),
        "errors": state.get("errors"),
    }


def _format_agent_results(final_context: dict) -> str:
    """
    구조화된 state 결과를 LLM이 읽기 쉬운 텍스트로 변환한다.
    """

    label_map = {
        "user_query": "사용자 질문",
        "task_type": "작업 유형",
        "member_id": "회원 ID",
        "customer_id": "고객 ID",
        "plan": "실행 계획",
        "completed_agents": "완료된 작업",
        "customer_result": "고객 기본 정보, 가입 계좌, 납입 이력 조회 결과",
        "product_result": "상품 기본 정보, 금리, 가입 조건, 약관 근거 조회 결과",
        "financial_result": "이자, 만기, 납입 현황, 중도해지 손실, 갈아타기 비교 계산 결과",
        "eligibility_result": "가입 가능 여부, 우대금리 충족 여부, 가입 가능 상품 필터링 결과",
        "recommend_result": "추천 상품, 추천 이유, 주의사항 결과",
        "validation_result": "조건 충돌, 계산 일치 여부, 근거, 부적절 추천 검증 결과",
        "agent_outputs": "Agent 공통 출력 기록",
        "errors": "오류 목록",
    }

    lines = []

    for key, label in label_map.items():
        value = final_context.get(key)

        if value is None or value == {} or value == []:
            continue

        lines.append(f"## {label}")
        lines.append(_safe_json_dumps(value))
        lines.append("")

    if not lines:
        return "사용 가능한 분석 결과가 없습니다."

    return "\n".join(lines)


def _fallback_final_answer(final_context: dict, error: Exception | None = None) -> str:
    """
    최종 답변 LLM 호출 실패 시 최소한의 답변을 생성한다.
    """

    lines = [
        "요청하신 내용을 기준으로 확인한 결과를 정리해드릴게요.",
        "",
    ]

    if final_context.get("customer_result"):
        lines.append("- 고객님의 기본 정보, 가입 계좌, 납입 이력을 확인했습니다.")

    if final_context.get("product_result"):
        lines.append("- 관련 상품의 금리, 가입 조건, 약관 정보를 확인했습니다.")

    if final_context.get("financial_result"):
        lines.append("- 이자, 만기, 납입 현황 또는 손실 비교 계산 결과를 확인했습니다.")

    if final_context.get("eligibility_result"):
        lines.append("- 고객 조건과 상품 조건을 비교해 가입 가능 여부를 확인했습니다.")

    if final_context.get("recommend_result"):
        lines.append("- 가입 가능 여부와 목적을 고려한 추천 결과를 확인했습니다.")

    if final_context.get("validation_result"):
        lines.append("- 조건 충돌이나 계산 오류가 있는지 검토했습니다.")

    if error:
        lines.append("")
        lines.append("다만 최종 문장을 생성하는 과정에서 일부 문제가 있어 요약 형태로 안내드립니다.")

    if len(lines) <= 2:
        lines.append("현재 확인 가능한 결과가 부족합니다. 고객 정보나 상품명을 함께 입력하면 더 정확히 안내드릴 수 있습니다.")

    lines.append("")
    lines.append("실제 적용 금리와 가입 가능 여부는 거래 시점, 고객 조건, 은행 정책에 따라 달라질 수 있으니 최종 내용은 상품설명서 또는 은행 상담을 통해 확인하는 것이 좋습니다.")

    return "\n".join(lines)


def _parse_plan_from_text(text: str) -> dict:
    """
    JSON 파싱 실패 시 텍스트에 포함된 Agent 이름을 순서대로 추출한다.
    """

    found = []

    for agent in AGENT_OPTIONS:
        if agent == "FINISH":
            continue

        pos = text.lower().find(agent.lower())
        if pos != -1:
            found.append((pos, agent))

    if found:
        found.sort()
        plan = [item[1] for item in found]
        return {
            "task_type": "casual",
            "plan": plan[:MAX_PLAN_LENGTH],
        }

    if "finish" in text.lower():
        return {
            "task_type": "casual",
            "plan": ["FINISH"],
        }

    return {
        "task_type": "casual",
        "plan": ["FINISH"],
    }


def _extract_json_text(text: str) -> str:
    """
    LLM 응답에서 JSON 문자열만 추출한다.
    """

    cleaned = text.strip()

    if "```" in cleaned:
        parts = cleaned.split("```")
        if len(parts) >= 2:
            cleaned = parts[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()

    if cleaned.startswith("{{") and cleaned.endswith("}}"):
        cleaned = cleaned[1:-1]
    elif "{{" in cleaned and "}}" in cleaned:
        cleaned = cleaned.replace("{{", "{").replace("}}", "}")

    return cleaned


def _get_last_user_text(messages: list) -> str:
    """
    messages에서 마지막 사용자 질문을 추출한다.
    tuple 형태와 LangChain Message 형태를 모두 처리한다.
    """

    for msg in reversed(messages):
        if isinstance(msg, tuple) and len(msg) >= 2:
            role, content = msg[0], msg[1]
            if role in ["user", "human"]:
                return str(content)

        if hasattr(msg, "type") and hasattr(msg, "content"):
            if msg.type in ["human", "user"]:
                return str(msg.content)

    return ""


def _append_log(state: AgentState, log: dict) -> list:
    """
    agent_logs에 로그를 추가한다.
    """

    logs = list(state.get("agent_logs") or [])
    logs.append(log)
    return logs


def _safe_json_dumps(value: Any) -> str:
    """
    객체를 안전하게 JSON 문자열로 변환한다.
    """

    try:
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return str(value)


def _clean_text(text: str) -> str:
    """
    최종 답변에서 HTML 태그 등 불필요한 표현을 정리한다.
    """

    if not text:
        return ""

    return (
        text.replace("<br>", "\n")
        .replace("<br/>", "\n")
        .replace("<br />", "\n")
        .strip()
    )


_original_supervisor_node = supervisor_node


def supervisor_node(state: AgentState) -> dict:
    plan = state.get("plan") or []
    mode = "plan" if not plan else "synthesize"

    with langfuse_observation(
        name="supervisor",
        as_type="span",
        input={
            "mode": mode,
            "task_type": state.get("task_type"),
            "message_count": len(state.get("messages") or []),
        },
        metadata={"agent": "supervisor", "mode": mode},
    ) as observation:
        result = _original_supervisor_node(state)
        update_observation(
            observation,
            output={
                "mode": mode,
                "task_type": result.get("task_type", state.get("task_type")),
                "next": result.get("next"),
                "plan": result.get("plan", state.get("plan")),
                "completed_agents": result.get("completed_agents", state.get("completed_agents") or []),
                "final_answer_preview": str(result.get("final_answer") or "")[:500],
            },
        )
        return result
