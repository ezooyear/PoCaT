"""
PostgreSQL MCP Client Wrapper

- Customer Agent에서 MCP PostgreSQL Tool을 호출하기 위한 얇은 wrapper입니다.
- 기존 Agent 코드는 동기 함수 기반이므로 call_postgres_mcp_tool() 동기 wrapper를 제공합니다.
"""

import asyncio
import os
from typing import Any

from dotenv import load_dotenv
from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client


load_dotenv()

MCP_POSTGRES_URL = os.getenv("MCP_POSTGRES_URL", "http://localhost:8000/mcp")


def _extract_text_result(result: Any) -> str:
    """
    MCP call_tool 결과에서 TextContent를 문자열로 추출합니다.
    """
    if getattr(result, "isError", False):
        errors = []
        for content in result.content:
            if isinstance(content, types.TextContent):
                errors.append(content.text)
        return "MCP Tool 실행 오류: " + "\n".join(errors)

    texts = []
    for content in result.content:
        if isinstance(content, types.TextContent):
            texts.append(content.text)

    if texts:
        return "\n".join(texts)

    if getattr(result, "structuredContent", None):
        return str(result.structuredContent)

    return str(result)


async def call_postgres_mcp_tool_async(
    tool_name: str,
    arguments: dict[str, Any] | None = None,
) -> str:
    """
    MCP PostgreSQL Server의 tool을 비동기로 호출합니다.
    """
    arguments = arguments or {}

    async with streamable_http_client(MCP_POSTGRES_URL) as (
        read_stream,
        write_stream,
        _,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments=arguments)
            return _extract_text_result(result)


def call_postgres_mcp_tool(
    tool_name: str,
    arguments: dict[str, Any] | None = None,
) -> str:
    """
    기존 LangChain tool 함수에서 사용하기 위한 동기 wrapper입니다.
    """
    return asyncio.run(call_postgres_mcp_tool_async(tool_name, arguments))