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


_PRODUCT_NAME_MAP: dict[str, str] | None = None


def _get_product_name_map() -> dict[str, str]:
    """products.rag_document_key → product_name 매핑을 1회 로드한다.

    PDF 파일명 stem == rag_document_key 이므로, RAG 청크의 출처 파일명으로
    DB의 정규 상품명을 복원할 수 있다. DB 접근 실패 시 빈 맵으로 degrade한다.
    """
    global _PRODUCT_NAME_MAP
    if _PRODUCT_NAME_MAP is None:
        _PRODUCT_NAME_MAP = {}
        try:
            from db.postgres_db import get_connection

            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT rag_document_key, product_name FROM products "
                "WHERE rag_document_key IS NOT NULL"
            )
            for key, name in cursor.fetchall():
                if key and name:
                    _PRODUCT_NAME_MAP[str(key).strip()] = str(name).strip()
            cursor.close()
            conn.close()
        except Exception:
            _PRODUCT_NAME_MAP = {}
    return _PRODUCT_NAME_MAP


_PRODUCT_DETAIL_MAP: dict[str, dict] | None = None


def _normalize_product_key(name: str) -> str:
    return "".join(str(name or "").lower().split())


def get_product_detail_map() -> dict[str, dict]:
    """정규화된 상품명 → products 테이블 정형 필드 매핑을 1회 로드한다.

    RAG 자유텍스트에서 금액·금리·기간을 정규식으로 뽑던 취약점(예: '9만원'을 9원으로 오인)을
    근본 차단하기 위해, 다운스트림(eligibility/financial/recommend)이 신뢰할 수 있는
    정형 값을 상품 후보에 주입하는 용도다. DB 접근 실패 시 빈 맵으로 degrade한다.
    """
    global _PRODUCT_DETAIL_MAP
    if _PRODUCT_DETAIL_MAP is None:
        _PRODUCT_DETAIL_MAP = {}
        try:
            from db.postgres_db import get_connection

            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT product_id, product_name, product_type, min_amount, max_amount,
                       min_period_months, max_period_months, base_rate, max_rate,
                       age_min, age_max, is_active, rag_document_key
                FROM products
                """
            )
            cols = [d[0] for d in cursor.description]
            for row in cursor.fetchall():
                rec = dict(zip(cols, row))
                name = rec.get("product_name")
                if name:
                    _PRODUCT_DETAIL_MAP[_normalize_product_key(name)] = rec
            cursor.close()
            conn.close()
        except Exception:
            _PRODUCT_DETAIL_MAP = {}
    return _PRODUCT_DETAIL_MAP


def _extract_source_stem(chunk: str) -> str:
    """청크 헤더의 '출처: {파일명}.pdf'에서 확장자를 뗀 stem을 추출한다."""
    match = re.search(r"출처:\s*(.+?)\.pdf", chunk)
    if not match:
        return ""
    return match.group(1).strip()


def _product_name_from_source(stem: str) -> str:
    """출처 파일명 stem을 정규 상품명으로 변환한다.

    1순위: products 테이블(rag_document_key) 정규명.
    2순위: 파일명 stem에서 날짜 접미사/언더스코어/문서종류 표기를 정리한 값.
    """
    if not stem:
        return ""

    name_map = _get_product_name_map()
    if stem in name_map:
        return name_map[stem]

    cleaned = re.sub(r"_?\d{4,6}$", "", stem)  # 끝의 날짜 접미사 제거 (_250901, 260402 등)
    cleaned = re.sub(r"[ _]*(약관|상품설명서|상품설명|설명서)[ _]*", " ", cleaned)
    cleaned = cleaned.replace("_", " ").strip()
    return cleaned


def _extract_product_name_from_chunk(chunk: str) -> str:
    # 1순위: 출처 PDF 파일명 → 정규 상품명. 문서 본문을 상품명으로 잘못 잡는 문제를 근본 차단한다.
    source_name = _product_name_from_source(_extract_source_stem(chunk))
    if source_name:
        return source_name

    # 2순위(폴백): 본문에서 상품명처럼 보이는 줄을 휴리스틱으로 추출
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
