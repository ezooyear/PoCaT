"""
Validation Agent
- State에 저장된 Agent별 결과를 검증합니다.
"""

from typing import Any

from graph.state import AgentState
from agents.base import make_agent_result
from agents.validation.tools import run_validation_checks


def validation_agent_node(state: AgentState) -> dict[str, Any]:
    """
    Validation Agent 노드 함수

    검증 항목:
    1. Agent별 구조화 결과 공통 포맷 확인
    2. task_type별 필수 결과 존재 여부 확인
    3. plan에 포함된 Agent 실행 완료 여부 확인
    4. errors 기록 여부 확인
    5. 추천 결과와 가입 가능 여부 결과의 기본 일관성 확인
    """

    issues, checked_items = run_validation_checks(state)
    is_valid = len(issues) == 0

    validation_result = make_agent_result(
        status="success" if is_valid else "failed",
        result={
            "is_valid": is_valid,
            "issues": issues,
            "revision_required": not is_valid,
            "checked_items": checked_items,
        },
        evidence=[],
        error=None if is_valid else "검증 이슈가 발견되었습니다.",
    )

    agent_outputs = dict(state.get("agent_outputs") or {})
    agent_outputs["validation_agent"] = validation_result

    completed_agents = list(state.get("completed_agents") or [])
    if "validation_agent" not in completed_agents:
        completed_agents.append("validation_agent")

    return {
        "validation_result": validation_result,
        "agent_outputs": agent_outputs,
        "current_agent": "validation_agent",
        "completed_agents": completed_agents,
        "current_step": (state.get("current_step") or 0) + 1,
    }