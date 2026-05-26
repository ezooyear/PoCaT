"""
전역 설정 모듈
- LLM 모델 초기화
- 환경변수 로드
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ─── 환경변수 ───
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
FSS_API_KEY = os.getenv("FSS_API_KEY", "0VZ3JA8KE01E3V6BCR1F")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b:free")


def get_llm(model: str = None, temperature: float = 0):
    """OpenRouter LLM 인스턴스를 생성합니다."""
    from langchain_openrouter import ChatOpenRouter

    return ChatOpenRouter(
        model=model or LLM_MODEL,
        temperature=temperature,
        api_key=OPENROUTER_API_KEY,
    )
