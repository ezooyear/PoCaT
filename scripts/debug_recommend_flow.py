# scripts/debug_recommend_flow.py
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import asyncio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from graph.builder import build_graph

prompt = "월 30만원씩 2년 동안 저축하려고 해. 내 가입 상품과 월 저축 가능액을 고려해서 가입 가능한 예적금 상품을 추천해줘."
graph = build_graph()

async def main():
    inputs = {
        "messages": [("user", prompt)],
        "user_query": prompt,
        "next": "",
        "member_id": "123",
        "customer_id": 123,
        "context": None,
        "plan": [],
        "current_step": 0,
        "agent_outputs": {},
    }
    
    print("--- START STREAM EVENTS ---", flush=True)
    async for event in graph.astream_events(inputs, version="v2"):
        kind = event["event"]
        node_name = event["metadata"].get("langgraph_node")
        run_name = event.get("name")
        print(f"EVENT: kind={kind}, node={node_name}, name={run_name}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())