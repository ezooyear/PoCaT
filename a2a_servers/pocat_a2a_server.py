"""
PoCaT A2A Server Wrapper

- 기존 LangGraph 기반 PoCaT 멀티 에이전트 그래프를
  하나의 표준 A2A Agent로 외부에 공개합니다.
- 내부 graph/state/agent 구조는 변경하지 않습니다.
"""

import asyncio
import logging

import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from a2a.helpers import (
    get_message_text,
    new_task_from_user_message,
    new_text_message,
    new_text_part,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
from a2a.types.a2a_pb2 import TaskState

from graph.builder import build_graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PoCaTGraphAgent:
    """기존 PoCaT LangGraph를 실행하는 내부 비즈니스 로직."""

    def __init__(self) -> None:
        self.graph = build_graph()

    async def invoke(self, user_request: str) -> str:
        """A2A 요청 문자열을 받아 기존 LangGraph를 실행하고 최종 답변을 반환합니다."""

        initial_state = {
            "messages": [("user", user_request)],
            "next": "",
            "member_id": None,
            "context": None,
            "plan": [],
            "current_step": 0,
            "agent_outputs": {},
        }

        # graph.invoke는 동기 함수이므로 A2A async 실행 흐름에서 안전하게 분리
        result = await asyncio.to_thread(self.graph.invoke, initial_state)

        # 1순위: final_answer가 있으면 사용
        final_answer = result.get("final_answer")
        if final_answer:
            return str(final_answer)

        # 2순위: 마지막 메시지 사용
        messages = result.get("messages") or []
        if messages:
            last_message = messages[-1]
            content = getattr(last_message, "content", None)
            if content:
                return str(content)
            return str(last_message)

        return "PoCaT 그래프 실행은 완료되었지만 반환된 답변이 없습니다."


class PoCaTAgentExecutor(AgentExecutor):
    """A2A 요청을 받아 PoCaT LangGraph 실행 결과를 반환하는 Executor."""

    def __init__(self) -> None:
        self.agent = PoCaTGraphAgent()

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """A2A 요청 처리."""

        if context.current_task:
            task = context.current_task
        else:
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)

        task_updater = TaskUpdater(
            event_queue=event_queue,
            task_id=task.id,
            context_id=task.context_id,
        )

        await task_updater.update_status(
            state=TaskState.TASK_STATE_WORKING,
            message=new_text_message("PoCaT 상담 그래프를 실행 중입니다."),
        )

        user_query = get_message_text(context.message)

        if not user_query:
            result_text = "텍스트 입력이 없어 PoCaT 상담을 실행할 수 없습니다."
        else:
            try:
                result_text = await self.agent.invoke(user_query)
            except Exception as error:
                logger.exception("PoCaT graph execution failed")
                result_text = f"PoCaT 그래프 실행 중 오류가 발생했습니다: {error}"

        await task_updater.add_artifact(
            parts=[new_text_part(text=result_text, media_type="text/plain")]
        )

        await task_updater.update_status(
            state=TaskState.TASK_STATE_COMPLETED,
            message=new_text_message("PoCaT 상담 그래프 실행이 완료되었습니다."),
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("Cancel is not supported.")


def create_public_agent_card(host: str, port: int) -> AgentCard:
    """외부 공개용 표준 A2A Agent Card."""

    skill_recommend = AgentSkill(
        id="pocat_deposit_recommendation",
        name="KB 예적금 상품 추천",
        description=(
            "고객 정보, 보유 계좌, 상품 조건, 가입 가능 여부, 금융 계산 결과를 바탕으로 "
            "KB 예적금 상품 추천 상담을 수행합니다."
        ),
        input_modes=["text/plain"],
        output_modes=["text/plain"],
        tags=["finance", "deposit", "savings", "recommendation", "pocat"],
        examples=[
            "고객 123번에게 적합한 예적금 상품 추천해줘",
            "고객 101번이 가입 가능한 적금 알려줘",
        ],
    )

    skill_lookup = AgentSkill(
        id="pocat_customer_product_lookup",
        name="고객 가입 상품 조회",
        description="고객 ID를 기준으로 현재 가입한 예금·적금 상품과 계좌 정보를 조회합니다.",
        input_modes=["text/plain"],
        output_modes=["text/plain"],
        tags=["finance", "customer", "account", "lookup"],
        examples=[
            "고객 123번이 가입한 상품 알려줘",
            "고객 101번 계좌 현황 조회해줘",
        ],
    )

    skill_compare = AgentSkill(
        id="pocat_product_compare",
        name="예적금 상품 비교",
        description="예금·적금 상품의 금리, 가입 조건, 우대 조건, 기간, 납입 방식을 비교합니다.",
        input_modes=["text/plain"],
        output_modes=["text/plain"],
        tags=["finance", "product", "comparison", "rag"],
        examples=[
            "KB Star 정기예금이랑 KB 스타적금Ⅲ 비교해줘",
            "금리가 높은 적금 상품 비교해줘",
        ],
    )

    return AgentCard(
        name="PoCaT KB Deposit Consultation Agent",
        description=(
            "KB 예적금 상담을 위한 PoCaT 멀티 에이전트입니다. "
            "내부적으로 LangGraph 기반 Supervisor, Customer, Product, Eligibility, "
            "Financial, Recommend, Validation Agent가 협업합니다."
        ),
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(
            streaming=True,
            extended_agent_card=False,
        ),
        supported_interfaces=[
            AgentInterface(
                protocol_binding="JSONRPC",
                url=f"http://{host}:{port}",
            )
        ],
        skills=[skill_recommend, skill_lookup, skill_compare],
    )


def create_app(host: str = "127.0.0.1", port: int = 9999) -> Starlette:
    public_agent_card = create_public_agent_card(host, port)

    request_handler = DefaultRequestHandler(
        agent_executor=PoCaTAgentExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=public_agent_card,
    )

    public_agent_card_json = {
        "name": "PoCaT KB Deposit Consultation Agent",
        "description": (
            "KB 예적금 상담을 위한 PoCaT 멀티 에이전트입니다. "
            "내부적으로 LangGraph 기반 Supervisor, Customer, Product, Eligibility, "
            "Financial, Recommend, Validation Agent가 협업합니다."
        ),
        "url": f"http://{host}:{port}/",
        "version": "1.0.0",
        "capabilities": {
            "streaming": True
        },
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": [
            {
                "id": "pocat_deposit_recommendation",
                "name": "KB 예적금 상품 추천",
                "description": (
                    "고객 정보, 보유 계좌, 상품 조건, 가입 가능 여부, 금융 계산 결과를 바탕으로 "
                    "KB 예적금 상품 추천 상담을 수행합니다."
                ),
                "tags": ["finance", "deposit", "savings", "recommendation", "pocat"],
                "examples": [
                    "고객 123번에게 적합한 예적금 상품 추천해줘",
                    "고객 101번이 가입 가능한 적금 알려줘"
                ]
            },
            {
                "id": "pocat_customer_product_lookup",
                "name": "고객 가입 상품 조회",
                "description": "고객 ID를 기준으로 현재 가입한 예금·적금 상품과 계좌 정보를 조회합니다.",
                "tags": ["finance", "customer", "account", "lookup"],
                "examples": [
                    "고객 123번이 가입한 상품 알려줘",
                    "고객 101번 계좌 현황 조회해줘"
                ]
            },
            {
                "id": "pocat_product_compare",
                "name": "예적금 상품 비교",
                "description": "예금·적금 상품의 금리, 가입 조건, 우대 조건, 기간, 납입 방식을 비교합니다.",
                "tags": ["finance", "product", "comparison", "rag"],
                "examples": [
                    "KB Star 정기예금이랑 KB 스타적금Ⅲ 비교해줘",
                    "금리가 높은 적금 상품 비교해줘"
                ]
            }
        ],
        "supportsAuthenticatedExtendedCard": False
    }

    async def agent_json(request):
        return JSONResponse(public_agent_card_json)

    routes = [
        Route("/.well-known/agent.json", agent_json, methods=["GET"]),
        Route("/.well-known/agent-card.json", agent_json, methods=["GET"]),
    ]

    routes.extend(create_jsonrpc_routes(request_handler, "/"))

    return Starlette(routes=routes)


def main() -> None:
    host = "127.0.0.1"
    port = 9999

    logger.info("Starting PoCaT A2A server")
    logger.info("A2A server URL: http://%s:%s", host, port)
    logger.info("Agent Card SDK route: http://%s:%s/.well-known/agent-card.json", host, port)
    logger.info("Agent Card legacy route: http://%s:%s/.well-known/agent.json", host, port)

    app = create_app(host=host, port=port)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()