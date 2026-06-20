"""추천 버튼 오류 재현 — app._run_assistant 경로를 헤드리스로 호출."""
import sys, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph.builder import build_graph

PROMPT = ("내 가입 상품, 월 저축 가능액, 가입 가능 조건을 종합해서 "
          "나에게 추천할 만한 예적금 상품을 순위와 이유, 주의사항까지 알려줘.")
graph_prompt = (
    "현재 로그인한 고객은 고객_123(테스트 고객번호 123)입니다.\n"
    "고객 본인이 이해하기 쉬운 말투로 답변해 주세요.\n\n"
    f"사용자 질문: {PROMPT}"
)

try:
    g = build_graph()
    result = g.invoke({
        "messages": [("user", graph_prompt)],
        "next": "",
        "member_id": "123",
        "customer_id": 123,
        "context": None,
        "plan": [],
        "current_step": 0,
        "agent_outputs": {},
    })
    msg = result["messages"][-1]
    print("=== OK ===")
    print(getattr(msg, "content", msg))
except Exception:
    print("=== EXCEPTION (this is the bug) ===")
    traceback.print_exc()
