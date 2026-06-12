"""
Product 에이전트 전용 도구
약관 및 상품설명서 RAG 검색 전담 (postgres DB nl2sql 차단)
"""
import re
from typing import Any

from langchain_core.tools import tool
from db.vectorstore import search_products


@tool
def search_terms(query: str) -> str:
    """상품의 약관, 상세 조건(가입 조건, 우대금리 요건, 가입 제한 나이/금액 등), 유의사항 등을 PDF 문서에서 검색합니다.
    KB국민은행 상품의 가입 요건을 조사할 때 사용합니다.

    Args:
        query: 검색할 질문 (예: "KB Star 정기예금 가입 제한 나이 및 최소 가입금액 조건", "KB국민은행 모든 예적금 상품 목록과 가입 요건")
    """
    results = search_products(query, k=5)
    if not results:
        return "검색된 약관 정보가 없습니다. Vector DB가 구축되지 않았거나 관련 정보가 없습니다."

    output = []
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source_file", "알 수 없음")
        page = doc.metadata.get("page", "?")
        output.append(f"[{i}] 출처: {source} / p.{page}\n{doc.page_content}")

    return "\n\n---\n\n".join(output)


def extract_product_candidates_from_search_results(raw_results: Any) -> list[dict[str, Any]]:
    if isinstance(raw_results, list):
        chunks = [str(item).strip() for item in raw_results if str(item).strip()]
    else:
        text = str(raw_results or "").strip()
        if not text:
            return []
        chunks = [chunk.strip() for chunk in re.split(r"\n\s*---+\s*\n", text) if chunk.strip()]

    candidates = []
    seen = set()

    for chunk in chunks:
        product_name = _extract_product_name_from_chunk(chunk)
        if not product_name:
            continue

        normalized = re.sub(r"\s+", "", product_name).lower()
        if normalized in seen:
            continue

        seen.add(normalized)
        candidates.append({"product_name": product_name, "raw_text": chunk})

    return candidates


def _extract_product_name_from_chunk(chunk: str) -> str:
    keywords = ("KB", "적금", "예금", "통장", "청년", "군인")

    for line in chunk.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("[") and "출처:" in stripped:
            continue
        if "출처:" in stripped and "/ p." in stripped:
            continue
        if len(stripped) > 100:
            continue
        if any(keyword in stripped for keyword in keywords):
            return stripped

    for line in chunk.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("["):
            return stripped[:100]

    return ""


# 이 에이전트에 바인딩될 도구 목록 (get_product_info를 제거하여 nl2sql을 통한 postgres db 접근 완전 차단)
PRODUCT_TOOLS = [search_terms]
