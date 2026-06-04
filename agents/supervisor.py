"""
Supervisor 에이전트
- 사용자 의도를 분석하여 task_type과 실행 계획 plan을 생성
- 단순 라우터가 아니라 실행 계획 수립자 역할
- 고객 DB가 필요한 질문은 LLM보다 규칙 기반 라우팅을 우선 적용
- 복합 질문은 여러 Agent가 순차 협업하도록 plan을 구성
"""

import json
from langchain_core.messages import SystemMessage

from config.settings import get_llm
from graph.state import AgentState
from prompts.supervisor_prompt import SUPERVISOR_SYSTEM_PROMPT


AGENT_OPTIONS = [
    "customer_agent",
    "calculation_agent",
    "product_agent",
    "recommend_agent",
    "validation_agent",
    "supervisor_final",
    "FINISH",
]

TASK_TYPES = [
    "casual",
    "product_info",
    "customer_lookup",
    "calculation",
    "recommendation",
    "early_termination",
    "switch_analysis",
]


def supervisor_node(state: AgentState) -> dict:
    """
    Supervisor 노드 함수

    역할:
    - 사용자 질문을 분석한다.
    - task_type을 결정한다.
    - 실행할 Agent 순서 plan을 만든다.
    - next에는 plan의 첫 번째 Agent를 넣는다.

    중요:
    - 고객 정보가 필요한 질문은 LLM 라우팅보다 규칙 기반 라우팅을 먼저 적용한다.
    - 이유: "내 가입 상품"처럼 상품이라는 단어가 포함되어도 product_agent가 아니라 customer_agent로 가야 하기 때문.
    """

    state_messages = list(state.get("messages", []))
    user_query = state.get("user_query") or _get_last_user_text(state_messages)

    # 1차: 규칙 기반 라우팅 우선
    # 고객 DB/계산/추천 계열은 LLM이 잘못 분류하면 안 되므로 먼저 확정한다.
    rule_routing = _fallback_routing(user_query)

    if rule_routing.get("task_type") != "casual":
        return {
            "next": rule_routing["next"],
            "task_type": rule_routing["task_type"],
            "plan": rule_routing["plan"],
            "user_query": user_query,
        }

    # 2차: 규칙으로 못 잡은 일반 대화/애매한 질문만 LLM 라우팅 사용
    llm = get_llm()

    messages = [
        SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
    ] + state_messages

    try:
        response = llm.invoke(messages)
        response_text = response.content.strip()
        routing = _parse_routing(response_text)
        routing = _normalize_routing(routing, user_query)

    except Exception:
        routing = rule_routing

    return {
        "next": routing.get("next", "FINISH"),
        "task_type": routing.get("task_type", "casual"),
        "plan": routing.get("plan", []),
        "user_query": user_query,
    }


def _parse_routing(response_text: str) -> dict:
    """
    LLM 응답에서 라우팅 결과를 파싱합니다.

    기대 JSON 형식:
    {
      "next": "customer_agent",
      "task_type": "recommendation",
      "plan": [
        "customer_agent",
        "calculation_agent",
        "product_agent",
        "recommend_agent",
        "validation_agent"
      ]
    }
    """

    text = response_text.strip()

    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    data = json.loads(text)

    return {
        "next": data.get("next", "FINISH"),
        "task_type": data.get("task_type", "casual"),
        "plan": data.get("plan", []),
    }


def _normalize_routing(routing: dict, user_query: str) -> dict:
    """
    LLM이 반환한 routing 결과를 현재 그래프 구조에 맞게 보정합니다.
    - analysis_agent 제거
    - supervisor_final은 plan에서 제거
    - next와 plan 불일치 보정
    - product_agent + casual 같은 모순 결과 보정
    """

    if not isinstance(routing, dict):
        return _fallback_routing(user_query)

    next_agent = routing.get("next", "FINISH")
    task_type = routing.get("task_type", "casual")
    plan = routing.get("plan", [])

    if task_type not in TASK_TYPES:
        task_type = "casual"

    if not isinstance(plan, list):
        plan = []

    # 과거 구조의 analysis_agent가 남아 있으면 새 구조로 변환
    converted_plan = []
    for agent in plan:
        if agent == "analysis_agent":
            converted_plan.extend(["customer_agent", "calculation_agent"])
        else:
            converted_plan.append(agent)

    plan = converted_plan

    # plan에서 유효한 Agent만 남김
    # supervisor_final은 builder가 마지막에 자동으로 보내므로 plan에서 제거
    valid_plan = []
    for agent in plan:
        if agent in AGENT_OPTIONS and agent not in ["FINISH", "supervisor_final"]:
            if agent not in valid_plan:
                valid_plan.append(agent)

    plan = valid_plan

    # next 보정
    if next_agent == "analysis_agent":
        next_agent = "customer_agent"

    if next_agent == "supervisor_final":
        next_agent = "FINISH"

    if next_agent not in AGENT_OPTIONS:
        next_agent = "FINISH"

    # LLM이 next는 agent인데 task_type을 casual로 주는 모순 방지
    # 예: {"next": "product_agent", "task_type": "casual", "plan": []}
    if next_agent != "FINISH" and task_type == "casual":
        fallback = _fallback_routing(user_query)
        if fallback.get("task_type") != "casual":
            return fallback

        # fallback도 casual이면 agent 실행 의미가 없으므로 FINISH로 보정
        next_agent = "FINISH"
        plan = []

    # next가 FINISH인데 plan이 있으면 plan의 첫 번째 Agent로 시작
    if next_agent == "FINISH" and plan:
        next_agent = plan[0]

    # next가 plan 안에 없고 plan이 있으면 plan 첫 번째 Agent로 보정
    if plan and next_agent not in plan:
        next_agent = plan[0]

    # LLM 결과가 애매하면 사용자 질문 기반 fallback
    if next_agent == "FINISH" and not plan and task_type != "casual":
        return _fallback_routing(user_query)

    return {
        "next": next_agent,
        "task_type": task_type,
        "plan": plan,
    }


def _fallback_routing(user_query: str) -> dict:
    """
    규칙 기반 라우팅.
    반드시 현재 새 Agent 구조만 사용합니다.

    주의:
    - "내 가입 상품"에는 '상품'이라는 단어가 들어가지만 product_info가 아니라 customer_lookup/calculation이다.
    - 따라서 고객 관련 키워드를 상품 약관 키워드보다 먼저 검사한다.
    - product_info 조건에는 '상품' 단독 키워드를 넣지 않는다.
    """

    query = (user_query or "").lower().replace(" ", "")

    # 1. 추천/추가 가입
    if any(keyword in query for keyword in [
        "추천",
        "recommend",
        "추가가입",
        "가입할만한",
        "뭐가좋아",
        "어떤상품이좋아",
        "나한테맞는",
        "내조건에맞는",
    ]):
        return {
            "next": "customer_agent",
            "task_type": "recommendation",
            "plan": [
                "customer_agent",
                "calculation_agent",
                "product_agent",
                "recommend_agent",
                "validation_agent",
            ],
        }

    # 2. 중도해지
    if any(keyword in query for keyword in [
        "중도해지",
        "해지손실",
        "해지하면",
        "손실",
        "급전",
    ]):
        return {
            "next": "customer_agent",
            "task_type": "early_termination",
            "plan": [
                "customer_agent",
                "calculation_agent",
                "product_agent",
                "recommend_agent",
                "validation_agent",
            ],
        }

    # 3. 갈아타기
    if any(keyword in query for keyword in [
        "갈아타기",
        "갈아타는게",
        "유지하는게",
        "새상품",
        "바꾸는게",
        "금리높은상품",
    ]):
        return {
            "next": "customer_agent",
            "task_type": "switch_analysis",
            "plan": [
                "customer_agent",
                "calculation_agent",
                "product_agent",
                "recommend_agent",
                "validation_agent",
            ],
        }

    # 4. 계산/납입/잔액/만기
    # "내 가입 상품과 납입 현황 보여줘"는 여기로 잡혀야 함.
    if any(keyword in query for keyword in [
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
    ]):
        return {
            "next": "customer_agent",
            "task_type": "calculation",
            "plan": [
                "customer_agent",
                "calculation_agent",
            ],
        }

    # 5. 고객 조회
    if any(keyword in query for keyword in [
        "내가입",
        "가입상품",
        "내계좌",
        "고객정보",
        "내정보",
        "보유상품",
        "가입목록",
        "내상품",
    ]):
        return {
            "next": "customer_agent",
            "task_type": "customer_lookup",
            "plan": [
                "customer_agent",
            ],
        }

    # 6. 상품/약관/RAG
    # '상품' 단독 키워드는 넣지 않는다.
    # "내 가입 상품"이 product_agent로 잘못 가는 것을 막기 위함.
    if any(keyword in query for keyword in [
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
    ]):
        return {
            "next": "product_agent",
            "task_type": "product_info",
            "plan": [
                "product_agent",
            ],
        }

    # 7. 일반 대화
    return {
        "next": "FINISH",
        "task_type": "casual",
        "plan": [],
    }


def _get_last_user_text(messages: list) -> str:
    """
    messages에서 마지막 사용자 질문을 추출합니다.
    app.py는 tuple("user", prompt) 형태를 쓰고,
    LangChain message 객체가 섞일 수도 있으므로 둘 다 처리합니다.
    """

    for msg in reversed(messages):
        # tuple 형태: ("user", "질문")
        if isinstance(msg, tuple) and len(msg) >= 2:
            role, content = msg[0], msg[1]
            if role in ["user", "human"]:
                return str(content)

        # LangChain Message 형태
        if hasattr(msg, "type") and hasattr(msg, "content"):
            if msg.type in ["human", "user"]:
                return str(msg.content)

    return ""