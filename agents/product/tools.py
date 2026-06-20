"""
Product 에이전트 전용 도구
약관 및 상품설명서 RAG 검색 전담 (postgres DB nl2sql 차단)
"""
import re
from typing import Any

from langchain_core.tools import tool
from db.vectorstore import search_products


def reformulate_query(query: str) -> str:
    """사용자 질문을 RAG 검색에 최적화된 형태로 재정형합니다."""
    try:
        from config.settings import get_llm
        from langchain_core.messages import SystemMessage, HumanMessage
        
        llm = get_llm(temperature=0)
        system_prompt = (
            "당신은 금융 상품 약관 검색에 최적화된 검색 키워드 정형화 전문가입니다.\n"
            "사용자의 질문을 분석하여, Vector DB(Chroma) 및 BM25 검색기에서 관련 금융 상품 약관을 "
            "가장 잘 찾아낼 수 있도록 '상품명'과 '검색 키워드'를 직관적으로 조합한 검색 쿼리 단 하나만 생성하십시오.\n\n"
            "예시:\n"
            "- 질문: '직업군인 나라사랑적금 가입대상'\n"
            "- 출력: 'KB나라사랑적금(직업군인용) 가입대상 조건 서류'\n"
            "- 질문: 'KBStar 예금이랑 일반예금 다른점'\n"
            "- 출력: 'KBStar정기예금 일반정기예금 차이점 가입대상 금액'\n\n"
            "출력에는 절대로 부가적인 설명이나 서론/결론, 따옴표 없이 오직 최적화된 쿼리 텍스트 한 줄만 출력해야 합니다."
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"사용자 질문: '{query}'")
        ]
        response = llm.invoke(messages)
        reformed = response.content.strip()
        reformed = reformed.replace("'", "").replace('"', "")
        return reformed if reformed else query
    except Exception:
        return query


@tool
def search_terms(query: str) -> str:
    """상품의 약관, 상세 조건(가입 조건, 우대금리 요건, 가입 제한 나이/금액 등), 유의사항 등을 PDF 문서에서 검색합니다.
    KB국민은행 상품의 가입 요건을 조사할 때 사용합니다.

    Args:
        query: 검색할 질문 (예: "KB Star 정기예금 가입 제한 나이 및 최소 가입금액 조건", "KB국민은행 모든 예적금 상품 목록과 가입 요건")
    """
    # Dynamic K 적용: 질문에 비교 분석형 성격이 담겨 있으면 더 넉넉한 컨텍스트(k=6) 확보, 그렇지 않으면 k=4
    is_comparative = any(keyword in query for keyword in ["비교", "차이", "모두", "목록", "공통", "다른점"])
    target_k = 6 if is_comparative else 4

    # RAG 검색 최적화를 위한 쿼리 재정형 적용
    reformed_query = reformulate_query(query)
    results = search_products(reformed_query, k=target_k)
    if not results:
        return "검색된 약관 정보가 없습니다. Vector DB가 구축되지 않았거나 관련 정보가 없습니다."

    output = []
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source_file", "알 수 없음")
        page = doc.metadata.get("page", "?")
        output.append(f"[{i}] 출처: {source} / p.{page}\n{doc.page_content}")

    return "\n\n---\n\n".join(output)


def extract_product_candidates_from_search_results(raw_results: Any) -> list[dict[str, Any]]:
    # 리스트 또는 단일 문자열 여부와 상관없이 모두 '---' 구분자를 분석하여 최종 청크 단위로 분할
    if isinstance(raw_results, list):
        raw_list = raw_results
    else:
        raw_list = [raw_results]

    chunks = []
    for item in raw_list:
        text = str(item or "").strip()
        if not text:
            continue
        # 개별 텍스트 내에 존재하는 '---' 구분자로 세부 분할
        split_parts = [part.strip() for part in re.split(r"\n\s*---+\s*\n", text) if part.strip()]
        chunks.extend(split_parts)

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
    # 상품명 후보에서 제외할 무효 키워드 목록
    invalid_markers = ["적용조건", "우대이율", "신규가입일", "영업점", "가입방법", "유의사항", "가입대상", "가입자격", "가입채널", "충족사례", "가입:", "이율", "금리"]
    raw_name = ""

    # 키워드가 포함된 라인 우선 검색
    for line in chunk.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("[") and "출처:" in stripped:
            continue
        if "출처:" in stripped and "/ p." in stripped:
            continue
        if len(stripped) > 80:
            continue
        # 상품명이 될 수 없는 무효 문구 필터링
        if any(marker in stripped for marker in invalid_markers):
            continue
        if any(keyword in stripped for keyword in keywords):
            raw_name = stripped
            break

    # 키워드가 포함된 라인이 없으면 첫 줄 선택
    if not raw_name:
        for line in chunk.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("["):
                # 상품명이 될 수 없는 무효 문구 필터링
                if any(marker in stripped for marker in invalid_markers):
                    continue
                raw_name = stripped[:80]
                break

    # 추출된 상품명 정제 (괄호, 따옴표, '상품설명서' 텍스트 제거)
    if raw_name:
        # 1. '상품설명서' 또는 '상품 설명서' 단어 먼저 제거
        name = re.sub(r"\s*상품설명서|\s*상품\s*설명서", "", raw_name).strip()
        # 2. 양 끝의 대괄호, 따옴표, 괄호 등 기호 제거 (제거 순서 교정)
        name = re.sub(r"^[「『\"'\[\(]+|[」』\"'\]\)]+$", "", name).strip()
        return name

    return ""


# 이 에이전트에 바인딩될 도구 목록 (get_product_info를 제거하여 nl2sql을 통한 postgres db 접근 완전 차단)
PRODUCT_TOOLS = [search_terms]
