"""
A2A Agent Card Loader

agent_cards/*.json 파일을 읽어
Supervisor와 Validation Agent가 사용할 수 있는 형태로 제공합니다.
"""

"""
loader의 역할 
1. JSON Agent Card를 읽는다.
2. 필수 필드가 있는지 검사한다.
3. Supervisor prompt에 넣을 요약문을 만든다.
4. Validation에서 Card 기준 output 누락 검증에 쓸 수 있게 한다.
"""


import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List


AGENT_CARD_DIR = Path(__file__).resolve().parent

REQUIRED_FIELDS = [
    "name",
    "description",
    "version",
    "input_state_keys",
    "output_state_keys",
    "tools",
    "skills",
    "handoff",
]


def _validate_agent_card(card: Dict[str, Any], path: Path) -> None:
    """Agent Card에 필수 필드가 있는지 확인합니다."""
    missing_fields = [field for field in REQUIRED_FIELDS if field not in card]

    if missing_fields:
        raise ValueError(
            f"[Agent Card Error] {path.name}에 필수 필드가 없습니다: {missing_fields}"
        )

    if not isinstance(card["skills"], list):
        raise ValueError(f"[Agent Card Error] {path.name}의 skills는 list여야 합니다.")

    if not isinstance(card["handoff"], dict):
        raise ValueError(f"[Agent Card Error] {path.name}의 handoff는 dict여야 합니다.")


@lru_cache(maxsize=1)
def load_agent_cards() -> Dict[str, Dict[str, Any]]:
    """
    agent_cards 폴더의 모든 JSON Agent Card를 읽어옵니다.

    Returns:
        {
            "customer_agent": {...},
            "product_agent": {...},
            "supervisor": {...}
        }
    """
    cards: Dict[str, Dict[str, Any]] = {}

    for path in sorted(AGENT_CARD_DIR.glob("*.json")):
        with path.open("r", encoding="utf-8") as file:
            card = json.load(file)

        _validate_agent_card(card, path)

        agent_name = card["name"]

        if agent_name in cards:
            raise ValueError(f"[Agent Card Error] 중복된 Agent name입니다: {agent_name}")

        cards[agent_name] = card

    return cards


def get_agent_card(agent_name: str) -> Dict[str, Any]:
    """Agent 이름으로 특정 Agent Card를 조회합니다."""
    cards = load_agent_cards()

    if agent_name not in cards:
        raise KeyError(f"[Agent Card Error] 등록되지 않은 Agent입니다: {agent_name}")

    return cards[agent_name]


def get_available_agent_names(include_supervisor: bool = False) -> List[str]:
    """
    사용 가능한 Agent 이름 목록을 반환합니다.

    Supervisor plan 생성에는 보통 supervisor를 제외한 전문 Agent만 사용합니다.
    """
    names = list(load_agent_cards().keys())

    if not include_supervisor:
        names = [name for name in names if name != "supervisor"]

    return names


def get_output_state_keys(agent_name: str) -> List[str]:
    """특정 Agent가 State에 저장해야 하는 output_state_keys를 반환합니다."""
    card = get_agent_card(agent_name)
    return card.get("output_state_keys", [])


def format_agent_card_summary(include_supervisor: bool = False) -> str:
    """
    Supervisor prompt에 넣기 좋은 Agent Card 요약 문자열을 생성합니다.

    너무 긴 JSON 전체를 넣지 않고,
    Agent 이름 / 역할 / skill / input / output / handoff만 짧게 요약합니다.
    """
    cards = load_agent_cards()

    lines: List[str] = []

    for agent_name, card in cards.items():
        if not include_supervisor and agent_name == "supervisor":
            continue

        skill_names = [
            skill.get("name", skill.get("id", "unknown_skill"))
            for skill in card.get("skills", [])
        ]

        next_agents = card.get("handoff", {}).get("next_agents", [])

        lines.append(
            "\n".join(
                [
                    f"- Agent: {agent_name}",
                    f"  Description: {card.get('description', '')}",
                    f"  Skills: {', '.join(skill_names)}",
                    f"  Input State Keys: {', '.join(card.get('input_state_keys', []))}",
                    f"  Output State Keys: {', '.join(card.get('output_state_keys', []))}",
                    f"  Tools: {', '.join(card.get('tools', []))}",
                    f"  Next Agents: {', '.join(next_agents)}",
                ]
            )
        )

    return "\n\n".join(lines)


def validate_agent_outputs_by_card(state: Dict[str, Any]) -> List[str]:
    """
    Validation Agent에서 사용할 수 있는 간단한 Card 기반 검증 함수입니다.

    plan에 포함된 Agent가 자신의 output_state_keys를 State에 남겼는지 확인합니다.
    단, messages/current_step/current_agent/completed_agents처럼 공통으로 항상 변하는 key는
    누락 검증에서 제외합니다.
    """
    cards = load_agent_cards()
    plan = state.get("plan") or []

    ignored_keys = {
        "messages",
        "current_step",
        "current_agent",
        "completed_agents",
        "agent_outputs",
        "errors",
        "context",
    }

    issues: List[str] = []

    for agent_name in plan:
        if agent_name in ["FINISH", "END", "validation_agent"]:
            continue

        if agent_name not in cards:
            issues.append(f"plan에 등록되지 않은 Agent가 포함되어 있습니다: {agent_name}")
            continue    

        card = cards[agent_name]
        output_keys = card.get("output_state_keys", [])

        for key in output_keys:
            if key in ignored_keys:
                continue

            if key.startswith("agent_outputs."):
                _, output_agent_name = key.split(".", 1)
                agent_outputs = state.get("agent_outputs") or {}

                if output_agent_name not in agent_outputs:
                    issues.append(
                        f"{agent_name} 실행 결과가 agent_outputs.{output_agent_name}에 없습니다."
                    )
                continue

            if state.get(key) in (None, "", [], {}):
                issues.append(
                    f"{agent_name}의 필수 output_state_key가 비어 있습니다: {key}"
                )

    return issues