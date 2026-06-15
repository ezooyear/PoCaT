"""
Product agent tools.
"""
from __future__ import annotations

import re
from typing import Any

from langchain_core.tools import tool

from db.vectorstore import search_products

TITLE_LIKE_MARKERS = [
    "가입 및",
    "저축 방법",
    "상품 안내",
    "상품 정보",
    "가입 조건",
    "유의사항",
    "예상 이자",
    "다음 단계",
]

ADVICE_LIKE_MARKERS = [
    "언제든 말씀",
    "말씀해 주세요",
    "도와드릴",
    "확인해 보세요",
    "선택하시면",
    "원하시면",
    "알려 주시면",
    "안내해 드릴",
    "가입 신청",
]


PRODUCT_SUFFIXES = ("적금", "예금", "부금", "통장", "저축")

GENERIC_EXACT_NAMES = {
    "미확인 상품",
    "직업군인 전용 적금",
    "정기예금",
    "적금",
    "예금",
    "거치식 적금",
    "적금(거치식)",
    "정기예금(장기예금)",
    "주택청약통장 보유",
    "KB국민은행 입출금통장",
    "입출금통장",
    "주택청약종합저축",
    "KB 예적금 상품군",
    "최적의 적금",
    "고액예금",
    "KB 고액예금",
}

GENERIC_CONTAINS = [
    "유형",
    "상품군",
    "최적",
    "적합한",
    "고액예금",
    "추천",
    "추가 확인",
    "가입 불가",
    "함께 볼 상품",
    "가장 적합한",
    "입출금통장",
    "보유",
    "장기예금",
    "거치식",
    "청약통장 포함",
]

SUSPICIOUS_CONTAINS = [
    "본인 명의",
    "가입일",
    "급여",
    "개월",
    "우대 금리",
    "기본 금리",
    "계약 기간",
    "계약기간",
    "납입",
    "추천 이유",
    "자동이체",
    "카드사용",
    "출처",
    "예상 이자",
]


@tool
def search_terms(query: str) -> str:
    """PDF 기반 상품/약관 검색 결과를 반환합니다."""
    results = search_products(query, k=_pick_search_k(query))
    if not results:
        return "검색된 상품 정보가 없습니다. Vector DB가 비어 있거나 관련 정보가 없습니다."

    output = []
    for index, doc in enumerate(results, start=1):
        source = doc.metadata.get("source_file", "source_unknown")
        page = doc.metadata.get("page", "?")
        output.append(f"[{index}] 출처: {source} / p.{page}\n{doc.page_content}")

    return "\n\n---\n\n".join(output)


def extract_product_candidates_from_search_results(raw_results: Any) -> list[dict[str, Any]]:
    if isinstance(raw_results, list):
        chunks: list[str] = []
        for item in raw_results:
            text = str(item or "").strip()
            if not text:
                continue
            chunks.extend(_split_product_chunks(text))
    else:
        text = str(raw_results or "").strip()
        if not text:
            return []
        chunks = _split_product_chunks(text)

    candidates: list[dict[str, Any]] = []
    seen = set()

    for chunk in chunks:
        found_from_table = False
        for candidate in _extract_product_candidates_from_table_rows(chunk):
            normalized = _canonicalize_product_name(candidate["product_name"])
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            candidates.append(candidate)
            found_from_table = True

        # If a chunk is already a structured product table, do not try to
        # re-parse trailing guidance text from the same block as another product.
        if found_from_table:
            continue

        product_name = _extract_product_name_from_chunk(chunk)
        if not product_name:
            continue

        product_name = _normalize_product_name(product_name)
        if not _looks_like_strict_product_name(product_name):
            continue

        normalized = _canonicalize_product_name(product_name)
        if not normalized or normalized in seen:
            continue

        seen.add(normalized)
        candidates.append({"product_name": product_name, "raw_text": chunk})

    return candidates


def _extract_product_name_from_chunk(chunk: str) -> str:
    table_name = _extract_product_name_from_table(chunk)
    if table_name:
        return table_name

    markdown_name = _extract_markdown_product_name(chunk)
    if markdown_name:
        return markdown_name

    for line in chunk.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("[") and "출처:" in stripped:
            continue
        if "출처:" in stripped and "/ p." in stripped:
            continue

        normalized = _normalize_product_name(stripped)
        if _looks_like_strict_product_name(normalized):
            return normalized

    return ""


def _split_product_chunks(text: str) -> list[str]:
    dashed_chunks = [
        chunk.strip()
        for chunk in re.split(r"\n\s*---+\s*\n", str(text or ""))
        if chunk.strip()
    ]

    chunks: list[str] = []
    for dashed_chunk in dashed_chunks or [str(text or "").strip()]:
        lines = dashed_chunk.splitlines()
        current: list[str] = []

        for line in lines:
            if _is_product_start_line(line):
                if current:
                    chunks.append("\n".join(current).strip())
                    current = []
            if line.strip() or current:
                current.append(line)

        if current:
            chunks.append("\n".join(current).strip())

    unique_chunks = []
    seen = set()
    for chunk in chunks:
        normalized = re.sub(r"\s+", " ", chunk).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique_chunks.append(chunk)

    return unique_chunks or ([str(text).strip()] if str(text or "").strip() else [])


def _is_product_start_line(line: str) -> bool:
    stripped = _normalize_product_name(line)
    return _looks_like_strict_product_name(stripped)


def _split_pipe_row(line: str) -> list[str]:
    parts = [part.strip() for part in str(line).split("|")]
    if parts and parts[0] == "":
        parts = parts[1:]
    if parts and parts[-1] == "":
        parts = parts[:-1]
    return parts


def _extract_product_name_from_table(text: str) -> str:
    for line in text.splitlines():
        if "|" not in line:
            continue
        parts = _split_pipe_row(line)
        if len(parts) < 2:
            continue

        left = re.sub(r"[*` ]", "", parts[0]).lower()
        right = re.sub(r"[*`]", "", parts[1]).strip()

        if left in {"상품명", "product_name"}:
            candidate = _normalize_product_name(right[:100])
            if _looks_like_strict_product_name(candidate):
                return candidate

        first_col = _normalize_product_name(parts[0])
        second_col = _normalize_product_name(parts[1])
        if _looks_like_strict_product_name(first_col) and any(
            keyword in second_col for keyword in PRODUCT_SUFFIXES
        ):
            return first_col
    return ""


def _extract_product_candidates_from_table_rows(text: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for line in text.splitlines():
        if "|" not in line:
            continue
        parts = _split_pipe_row(line)
        if len(parts) < 2:
            continue

        first_col = _normalize_product_name(parts[0])
        second_col = _normalize_product_name(parts[1])
        if _looks_like_strict_product_name(first_col) and any(
            keyword in second_col for keyword in PRODUCT_SUFFIXES
        ):
            candidates.append({"product_name": first_col, "raw_text": line.strip()})
    return candidates


def _extract_markdown_product_name(text: str) -> str:
    patterns = [
        r"\*\*(KB[^*\n]+(?:적금|예금|부금|통장|저축)[^*\n]*)\*\*",
        r"#+\s*([^\n#]*KB[^\n#]*(?:적금|예금|부금|통장|저축)[^\n#]*)",
        r"^\s*[-*]?\s*(KB[^\n]*(?:적금|예금|부금|통장|저축))\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.MULTILINE)
        if not match:
            continue
        candidate = _normalize_product_name(match.group(1).strip()[:100])
        if _looks_like_strict_product_name(candidate):
            return candidate
    return ""


def _normalize_product_name(product_name: str) -> str:
    text = str(product_name or "").strip()
    text = re.sub(r"^#+\s*", "", text)
    text = re.sub(r"^\s*[-*]\s*", "", text)
    text = re.sub(r"^\s*\d+[.)]\s*", "", text)
    text = text.replace("1️⃣", "").replace("2️⃣", "").replace("3️⃣", "")
    text = text.replace("4️⃣", "").replace("5️⃣", "")
    text = re.sub(r"^\s*\*\*(상품명|상품)\*\*\s*[:：]?\s*", "", text)
    text = re.sub(r"^\s*(상품명|상품)\s*[:：]?\s*", "", text)
    text = text.replace("**", "").strip()
    text = re.sub(r"\((?:[^()]*(?:개월|우대|연\s*[0-9]+(?:\.[0-9]+)?\s*%)\s*[^()]*)\)$", "", text).strip()
    text = re.sub(r"\s{2,}", " ", text)
    return text[:100]


def _looks_like_good_product_name(product_name: str) -> bool:
    if not product_name:
        return False
    if any(marker in product_name for marker in TITLE_LIKE_MARKERS):
        return False
    if any(marker in product_name for marker in ADVICE_LIKE_MARKERS):
        return False
    if _looks_like_document_name(product_name):
        return False
    if any(token in product_name for token in GENERIC_CONTAINS):
        return False
    if any(token in product_name for token in SUSPICIOUS_CONTAINS):
        return False
    return any(keyword in product_name for keyword in PRODUCT_SUFFIXES)


def _looks_like_strict_product_name(product_name: str) -> bool:
    text = _normalize_product_name(product_name)
    if any(marker in text for marker in TITLE_LIKE_MARKERS):
        return False
    if any(marker in text for marker in ADVICE_LIKE_MARKERS):
        return False
    if not _looks_like_good_product_name(text):
        return False
    if _looks_like_generic_product_label(text):
        return False
    if len(text) > 35:
        return False

    disqualifiers = [
        "경우",
        "예시",
        "포함",
        "제출",
        "확인",
        "신규",
        "보유",
        "고객",
        "가용",
        "일정",
        "안내",
        "가능한",
        "자유로운",
        "않거나",
        "입니다",
        "세후",
        "하려면",
        "하시면",
    ]
    if any(token in text for token in disqualifiers):
        return False
    if any(mark in text for mark in ["|", "*", ":", "->", "→", "▪", "•", "✅"]):
        return False
    return bool(re.search(r"(적금|예금|부금|통장|저축)$", text))


def _looks_like_document_name(product_name: Any) -> bool:
    text = str(product_name or "").strip()
    if not text:
        return False
    return any(marker in text for marker in ["상품설명서", "약관", "설명서", "상품 안내문", "가입안내", "확인서"])


def _canonicalize_product_name(product_name: str) -> str:
    text = _normalize_product_name(product_name)
    text = re.sub(r"\s+", "", text).lower()
    if text.startswith("kb"):
        text = text[2:]
    return text


def _looks_like_generic_product_label(product_name: str) -> bool:
    text = str(product_name or "").strip()
    if text in GENERIC_EXACT_NAMES:
        return True
    return any(token in text for token in GENERIC_CONTAINS)


def _pick_search_k(query: str) -> int:
    text = str(query or "")
    if any(keyword in text for keyword in ["추천", "갈아타", "유지", "비교"]):
        return 3
    return 4


PRODUCT_TOOLS = [search_terms]
