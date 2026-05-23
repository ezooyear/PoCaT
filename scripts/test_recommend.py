"""Test script: simulate recommend_agent with dummy LLM and fake vector search results."""
from types import SimpleNamespace
import sys
import os

# 프로젝트 루트 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agents.recommend_agent as ra
import config.settings as cs
from langchain_core.messages import HumanMessage


class DummyLLM:
    def invoke(self, messages):
        return SimpleNamespace(content="[더미응답] 추천: KB 청년희망적금, KB 직장인우대적금")


def fake_search(query, k=4):
    # return fake simple objects with metadata and page_content
    return [SimpleNamespace(metadata={"source_file": "KB_prod.pdf", "page": 1}, page_content="KB 청년희망적금 설명..."),
            SimpleNamespace(metadata={"source_file": "KB_prod.pdf", "page": 2}, page_content="KB 직장인우대적금 설명...")]


def main():
    # patch get_llm and the internal search_products reference
    cs.get_llm = lambda *a, **k: DummyLLM()
    ra.search_products = fake_search

    state = {
        "messages": [HumanMessage(content="결혼자금 2년, 월 30만원 추천해줘")],
        "member_id": "C001",
    }

    resp = ra.recommend_node(state)
    print("== Recommend Agent Response ==")
    print(resp["messages"][0].content)


if __name__ == "__main__":
    main()
