"""
Product 에이전트 전용 도구
약관 및 상품설명서 RAG 검색 전담 (postgres DB nl2sql 차단)
"""
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


# 이 에이전트에 바인딩될 도구 목록 (get_product_info를 제거하여 nl2sql을 통한 postgres db 접근 완전 차단)
PRODUCT_TOOLS = [search_terms]
