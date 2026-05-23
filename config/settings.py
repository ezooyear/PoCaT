"""
전역 설정 모듈
- LLM 모델 초기화
- 환경변수 로드
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ─── 환경변수 ───
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
FSS_API_KEY = os.getenv("FSS_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b:free")


def get_llm(model: str = None, temperature: float = 0):
    """OpenRouter LLM 인스턴스를 생성합니다."""
    from langchain_openrouter import ChatOpenRouter

    return ChatOpenRouter(
        model=model or LLM_MODEL,
        temperature=temperature,
        api_key=OPENROUTER_API_KEY,
    )
