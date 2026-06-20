"""
Product 에이전트 전용 도구

- 상품 약관 및 상품설명서 RAG 검색 전담
- PostgreSQL NL2SQL 접근은 차단
- RAG 검색 결과를 downstream agent가 읽을 수 있는 product_candidates 형태로 구조화
"""

import re
from typing import Any

from langchain_core.tools import tool

from db.vectorstore import search_products


# ---------------------------------------------------------------------------
# source_file 기반 상품명 보정
# ---------------------------------------------------------------------------

SOURCE_PRODUCT_META: dict[str, dict[str, str]] = {
    "KBStar정기예금_250901.pdf": {
        "product_name": "KB Star 정기예금",
        "product_type": "예금",
    },
    "KB상호부금_자유_250901.pdf": {
        "product_name": "KB상호부금(자유적립식)",
        "product_type": "적금",
    },
    "KB상호부금_정액_250901.pdf": {
        "product_name": "KB상호부금(정액적립식)",
        "product_type": "적금",
    },
    "KB스타적금3상품_2604.pdf": {
        "product_name": "KB 스타 적금Ⅲ",
        "product_type": "적금",
    },
    "직업군인_약관 및 상품설명서.pdf": {
        "product_name": "KB나라사랑적금(직업군인용)",
        "product_type": "적금",
    },
    "장병내일적금상품설명서260402.pdf": {
        "product_name": "KB장병내일준비적금",
        "product_type": "적금",
    },
    "장기간부_약관 및 상품설명서.pdf": {
        "product_name": "장기간부 적금",
        "product_type": "적금",
    },
}


INVALID_PRODUCT_NAME_MARKERS = [
    "적용조건",
    "우대이율",
    "우대금리",
    "신규가입일",
    "영업점",
    "가입방법",
    "유의사항",
    "판매기간",
    "가입대상",
    "가입조건",
    "계약기간",
    "저축금액",
    "저축방법",
    "페이지",
]


# ---------------------------------------------------------------------------
# RAG query
# ---------------------------------------------------------------------------

def reformulate_query(query: str) -> str:
    """사용자 질문을 RAG 검색에 최적화된 형태로 재정형합니다."""
    try:
        from config.settings import get_llm
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = get_llm(temperature=0)

        system_prompt = (
            "당신은 금융 상품 약관 검색에 최적화된 검색 키워드 정형화 전문가입니다.\n"
            "사용자의 질문을 분석하여, Vector DB(Chroma) 검색기에서 관련 금융 상품 약관을 "
            "가장 잘 찾아낼 수 있도록 상품명과 검색 키워드를 조합한 검색 쿼리 단 하나만 생성하십시오.\n\n"
            "예시:\n"
            "- 질문: '직업군인 나라사랑적금 가입대상'\n"
            "- 출력: 'KB나라사랑적금 직업군인 가입대상 조건 서류 우대금리'\n"
            "- 질문: 'KBStar 예금이랑 일반예금 다른점'\n"
            "- 출력: 'KB Star 정기예금 가입대상 금리 기간 가입금액 유의사항'\n\n"
            "출력에는 설명 없이 오직 최적화된 쿼리 텍스트 한 줄만 출력해야 합니다."
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"사용자 질문: '{query}'"),
        ]

        response = llm.invoke(messages)
        reformed = str(getattr(response, "content", response) or "").strip()
        reformed = reformed.replace("'", "").replace('"', "")

        return reformed if reformed else query

    except Exception:
        return query


@tool
def search_terms(query: str) -> str:
    """
    상품의 약관, 상세 조건, 가입 조건, 우대금리 요건,
    가입 제한 나이/금액, 유의사항 등을 PDF 문서에서 검색합니다.

    KB국민은행 상품의 가입 요건을 조사할 때 사용합니다.
    """
    is_comparative = any(
        keyword in query
        for keyword in ["비교", "차이", "모두", "목록", "공통", "다른점", "추천", "순위", "예적금"]
    )

    target_k = 10 if is_comparative else 6

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


# ---------------------------------------------------------------------------
# RAG result parser
# ---------------------------------------------------------------------------

def extract_product_candidates_from_search_results(raw_results: Any) -> list[dict[str, Any]]:
    """
    search_terms 결과 문자열을 product_candidates로 변환합니다.

    핵심 수정:
    - raw_results가 list여도 내부 "---" 기준으로 다시 chunk 분리
    - source_file 메타데이터를 우선 사용해 상품명 보정
    - 상품명 대신 우대이율/적용조건/본문 일부가 들어가는 문제 방지
    - product_result.evidence로 넘길 수 있도록 source_file/page/raw_text 보존
    """
    text = _normalize_raw_results(raw_results)

    if not text:
        return []

    chunks = [
        chunk.strip()
        for chunk in re.split(r"\n\s*---+\s*\n", text)
        if chunk.strip()
    ]

    if not chunks:
        return []

    grouped: dict[tuple[str, str], dict[str, Any]] = {}

    for chunk in chunks:
        source_file, page = _extract_source_info(chunk)
        product_name = _extract_product_name_from_chunk(chunk, source_file)
        product_type = _infer_product_type(product_name, chunk, source_file)

        if not _is_valid_product_name(product_name):
            continue

        key = (product_name, source_file or "unknown")

        if key not in grouped:
            grouped[key] = {
                "product_name": product_name,
                "product_type": product_type,
                "source_file": source_file,
                "pages": [],
                "raw_parts": [],
            }

        if page and page not in grouped[key]["pages"]:
            grouped[key]["pages"].append(page)

        grouped[key]["raw_parts"].append(chunk)

    candidates: list[dict[str, Any]] = []

    for item in grouped.values():
        raw_text = "\n\n".join(item["raw_parts"])
        pages = item["pages"]

        candidate = {
            "product_name": item["product_name"],
            "product_type": item["product_type"],
            "raw_text": raw_text,
            "source_file": item["source_file"],
            "page": pages[0] if pages else None,
            "source_pages": pages,
            "base_rate": _extract_base_rate(raw_text),
            "max_rate": _extract_max_rate(raw_text),
            "min_amount": _extract_min_amount(raw_text),
            "max_amount": _extract_max_amount(raw_text),
            "min_period_months": _extract_min_months(raw_text),
            "max_period_months": _extract_max_months(raw_text),
            "evidence": [
                {
                    "source": "rag_search",
                    "source_file": item["source_file"],
                    "page": pages[0] if pages else None,
                    "pages": pages,
                    "text": raw_text[:500],
                    "confidence": "medium",
                }
            ],
        }

        candidates.append(candidate)

    return candidates


def _normalize_raw_results(raw_results: Any) -> str:
    if isinstance(raw_results, list):
        return "\n\n---\n\n".join(
            str(item).strip()
            for item in raw_results
            if str(item).strip()
        )

    return str(raw_results or "").strip()


def _extract_source_info(chunk: str) -> tuple[str | None, str | None]:
    match = re.search(r"출처:\s*([^/\n]+)\s*/\s*p\.?([0-9A-Za-z_.-]+|\?)", chunk)

    if not match:
        return None, None

    return match.group(1).strip(), match.group(2).strip()


def _extract_product_name_from_chunk(chunk: str, source_file: str | None = None) -> str:
    # 1순위: source_file 기반 보정
    if source_file in SOURCE_PRODUCT_META:
        return SOURCE_PRODUCT_META[source_file]["product_name"]

    # 2순위: 문서 내 명시적 상품명 패턴
    patterns = [
        r"상\s*품\s*명\s*[:：]\s*([^\n]+)",
        r"상품명\s*[:：]\s*([^\n]+)",
        r"상품명\s+([^\n]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, chunk)

        if match:
            candidate = _clean_product_name(match.group(1))

            if _is_valid_product_name(candidate):
                return candidate

    # 3순위: 짧은 줄 중 실제 상품명처럼 보이는 줄
    for line in chunk.splitlines():
        candidate = _clean_product_name(line)

        if not candidate:
            continue

        if len(candidate) > 40:
            continue

        if any(marker in candidate for marker in INVALID_PRODUCT_NAME_MARKERS):
            continue

        if _is_valid_product_name(candidate):
            return candidate

    # 4순위: source_file에서 안전한 이름 추론
    if source_file:
        inferred = _infer_product_name_from_source_file(source_file)

        if _is_valid_product_name(inferred):
            return inferred

    return ""


def _infer_product_name_from_source_file(source_file: str) -> str:
    name = str(source_file or "")
    name = re.sub(r"\.pdf$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"[_\-]", " ", name)
    name = re.sub(r"[0-9]{4,}", "", name)
    name = name.replace("약관 및 상품설명서", "")
    name = name.replace("상품설명서", "")
    name = name.replace("상품", "")
    name = re.sub(r"\s+", " ", name).strip()

    return name


def _clean_product_name(text: Any) -> str:
    cleaned = str(text or "").strip()
    cleaned = cleaned.replace("•", "").replace("▪", "").replace("ㅇ", "")
    cleaned = cleaned.strip(":-：*·ㆍ ")
    cleaned = re.sub(r"\s+", " ", cleaned)

    # 설명 문장이 붙은 경우 앞부분만 사용
    cleaned = re.split(r"\s{2,}|[。]|[.]|[,]", cleaned)[0].strip()

    return cleaned


def _is_valid_product_name(product_name: Any) -> bool:
    text = str(product_name or "").strip()

    if not text:
        return False

    if text == "미확인 상품":
        return False

    if len(text) > 40:
        return False

    if any(marker in text for marker in INVALID_PRODUCT_NAME_MARKERS):
        return False

    # 너무 긴 설명 문장 방지
    if len(text.split()) > 6:
        return False

    return any(
        keyword in text
        for keyword in ["KB", "적금", "예금", "부금", "장병", "군인", "간부"]
    )


def _infer_product_type(product_name: str, chunk: str, source_file: str | None = None) -> str | None:
    if source_file in SOURCE_PRODUCT_META:
        return SOURCE_PRODUCT_META[source_file]["product_type"]

    merged = f"{product_name}\n{chunk}"

    if "예금" in merged:
        return "예금"

    if "적금" in merged:
        return "적금"

    if "부금" in merged:
        return "부금"

    return None


# ---------------------------------------------------------------------------
# field extraction helpers
# ---------------------------------------------------------------------------

def _extract_base_rate(text: str) -> float | None:
    patterns = [
        r"기본\s*이율[^0-9]*([0-9]+(?:\.[0-9]+)?)\s*%",
        r"기본\s*금리[^0-9]*([0-9]+(?:\.[0-9]+)?)\s*%",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)

        if match:
            return float(match.group(1))

    return None


def _extract_max_rate(text: str) -> float | None:
    rates = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*%", text)

    if not rates:
        return None

    return max(float(rate) for rate in rates)


def _extract_min_months(text: str) -> int | None:
    months = [int(value) for value in re.findall(r"([0-9]{1,2})\s*개월", text)]

    return min(months) if months else None


def _extract_max_months(text: str) -> int | None:
    months = [int(value) for value in re.findall(r"([0-9]{1,2})\s*개월", text)]

    return max(months) if months else None


def _extract_min_amount(text: str) -> int | None:
    candidates = []

    patterns = [
        r"(?:최소|최저|가입금액).{0,20}?([0-9,]+)\s*(만원|원)\s*이상",
        r"([0-9,]+)\s*(만원|원)\s*이상",
    ]

    for pattern in patterns:
        for amount, unit in re.findall(pattern, text):
            parsed = _parse_money(amount, unit)

            if parsed is not None:
                candidates.append(parsed)

    return min(candidates) if candidates else None


def _extract_max_amount(text: str) -> int | None:
    candidates = []

    patterns = [
        r"(?:최대|최고|납입한도|한도).{0,30}?([0-9,]+)\s*(만원|원)\s*이하",
        r"([0-9,]+)\s*(만원|원)\s*이하",
    ]

    for pattern in patterns:
        for amount, unit in re.findall(pattern, text):
            parsed = _parse_money(amount, unit)

            if parsed is not None:
                candidates.append(parsed)

    return max(candidates) if candidates else None


def _parse_money(amount_text: str, unit: str) -> int | None:
    digits = re.sub(r"[^0-9]", "", str(amount_text or ""))

    if not digits:
        return None

    amount = int(digits)

    if unit == "만원":
        return amount * 10000

    return amount


# 이 에이전트에 바인딩될 도구 목록
# Product Agent의 PostgreSQL NL2SQL 접근 차단 유지
PRODUCT_TOOLS = [search_terms]