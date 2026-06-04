"""
Validation 에이전트 — 다른 에이전트 결과 검증
"""
from langchain_core.messages import SystemMessage
from config.settings import get_llm
from graph.state import AgentState
from agents.base import run_agent_loop
from agents.validation.prompts import VALIDATION_SYSTEM_PROMPT
from agents.validation.tools import VALIDATION_TOOLS


def validation_agent_node(state: AgentState) -> dict:
    # 이전 에이전트 결과를 검증 대상으로 주입
    agent_outputs = state.get("agent_outputs") or {}
    prev_results = ""
    for name, output in agent_outputs.items():
        prev_results += f"\n### {name} 결과\n{output}\n"

    extra_prompt = ""
    if prev_results:
        extra_prompt = f"\n\n## 검증 대상 결과\n{prev_results}"

    return run_agent_loop(
        state=state,
        system_prompt=VALIDATION_SYSTEM_PROMPT + extra_prompt,
        tools=VALIDATION_TOOLS,
        output_key="validation_agent",
        max_iterations=2,
    )
