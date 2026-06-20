"""
Product agent tools.
Handles product RAG search and structured extraction.
"""

from __future__ import annotations

import contextvars
import re
from time import perf_counter
from typing import Any

from langchain_core.tools import tool

from db.vectorstore import search_products
from observability.langfuse import langfuse_observation, update_observation


_PRODUCT_PERF_CONTEXT: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "product_perf_context",
    default=None,
)

INVALID_PRODUCT_NAME_VALUES = {
    "상품명",
    "상품 유형",
    "가입 대상·나이",
    "가입·납입 금액",
    "가입 금액",
    "가입 가능 기간",
    "가입 기간",
    "기본 금리",
    "우대 금리",
    "최고 적용 금리",
    "우대조건 충족 여부",
    "추가 확인 필요",
    "항목",
    "내용",
    "순위",
}

NON_PRODUCT_HEADINGS = {
    "현재 상황 요약",
    "고객 상황 요약",
    "상황 요약",
    "내 계정 요약",
    "추천 상품",
    "추천 가능한 예·적금 상품",
    "우대조건 현황",
    "주의사항",
    "추가 확인 필요",
    "요약",
}


def reset_product_perf_stats() -> None:
    _PRODUCT_PERF_CONTEXT.set(
        {
            "query_generation_count": 0,
            "query_generation_duration_ms": 0.0,
            "search_terms_call_count": 0,
            "search_terms_duration_ms": 0.0,
            "rag_search_count": 0,
            "rag_search_duration_ms": 0.0,
            "llm_call_count": 0,
            "llm_duration_ms": 0.0,
            "last_query": None,
            "last_reformed_query": None,
            "last_target_k": None,
            "last_result_count": 0,
        }
    )


def get_product_perf_stats() -> dict[str, Any]:
    return dict(_PRODUCT_PERF_CONTEXT.get() or {})


def _update_product_perf_stats(**updates: Any) -> None:
    stats = _PRODUCT_PERF_CONTEXT.get()
    if stats is None:
        return

    merged = dict(stats)
    for key, value in updates.items():
        if isinstance(value, (int, float)) and isinstance(merged.get(key), (int, float)):
            merged[key] = merged.get(key, 0) + value
        else:
            merged[key] = value
    _PRODUCT_PERF_CONTEXT.set(merged)


def reformulate_query(query: str) -> str:
    started_at = perf_counter()
    try:
        from config.settings import get_llm
        from langchain_core.messages import HumanMessage, SystemMessage

        with langfuse_observation(
            name="product_agent.query_generation",
            as_type="span",
            input={"query": query},
            metadata={"agent": "product_agent", "step": "query_generation"},
        ) as observation:
            llm = get_llm(temperature=0)
            messages = [
                SystemMessage(
                    content=(
                        "You rewrite a Korean banking product question into a concise RAG search query.\n"
                        "Return only the optimized query text without explanation.\n"
                    )
                ),
                HumanMessage(content=f"질문: {query}"),
            ]
            llm_started_at = perf_counter()
            response = llm.invoke(messages)
            llm_duration_ms = (perf_counter() - llm_started_at) * 1000
            final_query = str(response.content or "").strip().replace("'", "").replace('"', "") or query
            update_observation(
                observation,
                output={
                    "reformed_query": final_query,
                    "duration_ms": llm_duration_ms,
                    "usage_metadata": getattr(response, "usage_metadata", None),
                },
                metadata={"agent": "product_agent"},
            )
            _update_product_perf_stats(
                query_generation_count=1,
                query_generation_duration_ms=llm_duration_ms,
                llm_call_count=1,
                llm_duration_ms=llm_duration_ms,
                last_query=query,
                last_reformed_query=final_query,
            )
            return final_query
    except Exception:
        _update_product_perf_stats(
            query_generation_count=1,
            query_generation_duration_ms=(perf_counter() - started_at) * 1000,
            last_query=query,
            last_reformed_query=query,
        )
        return query


@tool
def search_terms(query: str) -> str:
    """Search product descriptions and conditions from the vector store."""
    started_at = perf_counter()
    with langfuse_observation(
        name="product_agent.search_terms",
        as_type="span",
        input={"query": query},
        metadata={"agent": "product_agent", "tool": "search_terms"},
    ) as observation:
        is_comparative = any(keyword in query for keyword in ["비교", "차이", "모두", "목록", "공통", "다른"])
        target_k = 6 if is_comparative else 3
        reformed_query = reformulate_query(query)

        rag_started_at = perf_counter()
        with langfuse_observation(
            name="product_agent.rag_search",
            as_type="span",
            input={"query": reformed_query, "k": target_k},
            metadata={"agent": "product_agent", "step": "rag_search"},
        ) as rag_observation:
            results = search_products(reformed_query, k=target_k)
            rag_duration_ms = (perf_counter() - rag_started_at) * 1000
            update_observation(
                rag_observation,
                output={"result_count": len(results or []), "duration_ms": rag_duration_ms},
                metadata={"agent": "product_agent"},
            )

        _update_product_perf_stats(
            search_terms_call_count=1,
            rag_search_count=1,
            rag_search_duration_ms=rag_duration_ms,
            last_query=query,
            last_reformed_query=reformed_query,
            last_target_k=target_k,
            last_result_count=len(results or []),
        )

        if not results:
            total_duration_ms = (perf_counter() - started_at) * 1000
            _update_product_perf_stats(search_terms_duration_ms=total_duration_ms)
            update_observation(
                observation,
                output={
                    "target_k": target_k,
                    "reformed_query": reformed_query,
                    "result_count": 0,
                    "duration_ms": total_duration_ms,
                },
                metadata={"agent": "product_agent"},
            )
            return "검색된 상품 정보가 없습니다."

        output = []
        for index, doc in enumerate(results, 1):
            source = doc.metadata.get("source_file", "unknown")
            page = doc.metadata.get("page", "?")
            output.append(f"[{index}] 출처: {source} / p.{page}\n{doc.page_content}")

        total_duration_ms = (perf_counter() - started_at) * 1000
        _update_product_perf_stats(search_terms_duration_ms=total_duration_ms)
        update_observation(
            observation,
            output={
                "target_k": target_k,
                "reformed_query": reformed_query,
                "result_count": len(results),
                "duration_ms": total_duration_ms,
            },
            metadata={"agent": "product_agent"},
        )
        return "\n\n---\n\n".join(output)


def extract_product_candidates_from_search_results(raw_results: Any) -> list[dict[str, Any]]:
    return extract_structured_products_from_search_results(raw_results)


def extract_structured_products_from_search_results(raw_results: Any) -> list[dict[str, Any]]:
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
        if not product_name or product_name in INVALID_PRODUCT_NAME_VALUES:
            continue

        normalized = re.sub(r"\s+", "", product_name).lower()
        if normalized in seen:
            continue

        seen.add(normalized)
        candidates.append(_build_structured_product_candidate(product_name, chunk))
    return candidates


def extract_products_from_product_summary(summary: Any) -> list[dict[str, Any]]:
    text = str(summary or "").strip()
    if not text:
        return []

    # 1순위: | 순위 | 상품명 | ... | 형태의 추천 상품 표
    if _has_recommendation_table(text):
        rec_products = _extract_products_from_markdown_list_table(text)
        if rec_products:
            return rec_products

    # 2순위: 상품별 | 항목 | 내용 | 상세 블록
    detail_products = _extract_products_from_detail_blocks(text)
    if detail_products:
        return detail_products

    # 3순위: 기타 | 상품명 | ... | 목록 표
    list_table_products = _extract_products_from_markdown_list_table(text)
    if list_table_products:
        return list_table_products

    # 4순위: 헤딩에서 상품명 추출 (keyword 필터 및 NON_PRODUCT_HEADINGS 필터 적용됨)
    return extract_structured_products_from_search_results(text)


def parse_korean_money(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None

    text = text.replace(",", "")
    text = text.replace(" ", "")

    eok = "\uC5B5"
    cheonman = "\uCC9C\uB9CC"
    baekman = "\uBC31\uB9CC"
    simman = "\uC2ED\uB9CC"
    man = "\uB9CC"
    won = "\uC6D0"
    match = re.search(
        rf"([0-9]+(?:\.[0-9]+)?)({eok}|{cheonman}|{baekman}|{simman}|{man})?{won}?",
        text,
    )
    if not match:
        return None

    number = float(match.group(1))
    unit = match.group(2) or ""
    multiplier = {
        eok: 100000000,
        cheonman: 10000000,
        baekman: 1000000,
        simman: 100000,
        man: 10000,
        "": 1,
    }.get(unit, 1)
    return int(number * multiplier)


def _has_recommendation_table(text: str) -> bool:
    """순위·상품명 컬럼이 모두 있는 추천 상품 표인지 확인."""
    for line in str(text or "").splitlines():
        if "|" not in line:
            continue
        cells = [_clean_table_cell(c) for c in line.split("|") if c.strip()]
        if "상품명" in cells and "순위" in cells:
            return True
    return False


def _extract_products_from_detail_blocks(text: str) -> list[dict[str, Any]]:
    blocks = _split_summary_blocks(text)
    products: list[dict[str, Any]] = []
    seen = set()

    for block in blocks:
        rows = _extract_detail_rows(block)
        if not rows:
            continue

        if rows[0][0] != "항목" or rows[0][1] != "내용":
            continue

        row_map = {key: value for key, value in rows[1:]}
        product_name = _clean_product_name(row_map.get("상품명") or _extract_heading_product_name(block) or "")
        if not product_name or product_name in INVALID_PRODUCT_NAME_VALUES:
            continue
        if product_name in NON_PRODUCT_HEADINGS:
            continue

        normalized = re.sub(r"\s+", "", product_name).lower()
        if normalized in seen:
            continue
        seen.add(normalized)

        amount_text = row_map.get("가입·납입 금액") or row_map.get("가입 금액") or _extract_amount_text(block)
        min_amount, max_amount, normalized_amount = _parse_amount_range_text(amount_text)

        products.append(
            {
                "product_name": product_name,
                "bank": "KB국민은행" if "KB" in block or "국민은행" in block else None,
                "product_type": row_map.get("상품 유형"),
                "term": row_map.get("가입 가능 기간") or row_map.get("가입 기간") or _extract_heading_term_text(block),
                "term_months": _parse_months(row_map.get("가입 가능 기간") or row_map.get("가입 기간") or _extract_heading_term_text(block)),
                "amount_text": amount_text,
                "min_amount": min_amount,
                "min_monthly_amount": min_amount,
                "max_monthly_amount": max_amount,
                "amount_unit_normalized": normalized_amount,
                "base_rate": _parse_rate(row_map.get("기본 금리")),
                "max_rate": _parse_rate(row_map.get("최고 적용 금리") or row_map.get("우대 금리")),
                "join_channel": _extract_join_channels(block),
                "age_condition": row_map.get("가입 대상·나이"),
                "eligibility_text": row_map.get("가입 대상·나이"),
                "preferential_conditions": _extract_preferential_conditions_from_text(block),
                "source": "summary_detail_table",
                "raw_text": block,
            }
        )

    return products


def _extract_products_from_markdown_list_table(text: str) -> list[dict[str, Any]]:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    header_index = -1
    header_cells: list[str] = []

    for index, line in enumerate(lines):
        if "|" not in line:
            continue
        cells = [_clean_table_cell(cell) for cell in line.split("|") if cell.strip()]
        if "상품명" in cells and len(cells) >= 3:
            if cells[:2] == ["항목", "내용"]:
                continue
            header_index = index
            header_cells = cells
            break

    if header_index < 0:
        return []

    column_index = {name: idx for idx, name in enumerate(header_cells)}
    products: list[dict[str, Any]] = []
    seen = set()

    for line in lines[header_index + 1 :]:
        if "|" not in line or re.fullmatch(r"[\|\-\s:*]+", line):
            continue
        cells = [_clean_table_cell(cell) for cell in line.split("|") if cell.strip()]
        if len(cells) < len(header_cells):
            continue

        product_name = _clean_product_name(cells[column_index["상품명"]])
        if not product_name or product_name in INVALID_PRODUCT_NAME_VALUES:
            continue

        normalized = re.sub(r"\s+", "", product_name).lower()
        if normalized in seen:
            continue
        seen.add(normalized)

        amount_text = (
            _get_cell(cells, column_index, "가입·납입 금액")
            or _get_cell(cells, column_index, "가입 금액")
            or _get_cell(cells, column_index, "최소·최대 가입 금액")
        )
        min_amount, max_amount, normalized_amount = _parse_amount_range_text(amount_text)
        is_recommendation = "순위" in header_cells
        preferential_text = _get_cell(cells, column_index, "우대조건 적용 여부") or ""

        products.append(
            {
                "product_name": product_name,
                "bank": "KB국민은행" if "KB" in product_name else None,
                "product_type": _get_cell(cells, column_index, "상품 유형"),
                "term": _get_cell(cells, column_index, "가입 가능 기간") or _get_cell(cells, column_index, "가입 기간"),
                "term_months": _parse_months(_get_cell(cells, column_index, "가입 가능 기간") or _get_cell(cells, column_index, "가입 기간")),
                "amount_text": amount_text,
                "min_amount": min_amount,
                "min_monthly_amount": min_amount,
                "max_monthly_amount": max_amount,
                "amount_unit_normalized": normalized_amount,
                "base_rate": _parse_rate(_get_cell(cells, column_index, "기본 금리*") or _get_cell(cells, column_index, "기본 금리")),
                "max_rate": _parse_rate(
                    _get_cell(cells, column_index, "총 예상 금리")
                    or _get_cell(cells, column_index, "우대 금리(가능 시)")
                    or _get_cell(cells, column_index, "우대 금리")
                ),
                "join_channel": [],
                "age_condition": _get_cell(cells, column_index, "가입 대상·나이"),
                "eligibility_text": _get_cell(cells, column_index, "가입 대상·나이"),
                "preferential_conditions": [],
                "preferential_conditions_text": preferential_text,
                "source": "recommendation_table" if is_recommendation else "markdown_table",
                "raw_text": line,
            }
        )

    return products


def _build_structured_product_candidate(product_name: str, chunk: str) -> dict[str, Any]:
    rates = [float(rate) for rate in re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*%", chunk)]
    months = [int(month) for month in re.findall(r"([0-9]{1,2})\s*개월", chunk)]
    amount_text = _extract_amount_text(chunk)
    min_amount, max_amount, normalized_amount = _parse_amount_range_text(amount_text)

    return {
        "product_name": product_name,
        "bank": "KB국민은행" if "KB" in chunk or "국민은행" in chunk else None,
        "product_type": _infer_product_type(product_name, chunk),
        "term": f"{max(months)}개월" if months else None,
        "term_months": max(months) if months else None,
        "amount_text": amount_text,
        "min_amount": min_amount,
        "min_monthly_amount": min_amount,
        "max_monthly_amount": max_amount,
        "amount_unit_normalized": normalized_amount,
        "base_rate": min(rates) if rates else None,
        "max_rate": max(rates) if rates else None,
        "join_channel": _extract_join_channels(chunk),
        "age_condition": _extract_first_match(chunk, [r"(만\s*[0-9]{1,2}세\s*(?:이상|이하|~\s*[0-9]{1,2}세)?)"]),
        "eligibility_text": _extract_first_match(chunk, [r"(만\s*[0-9]{1,2}세[^\n]*)", r"(가입 대상[^\n]*)"]),
        "preferential_conditions": _extract_preferential_conditions_from_text(chunk),
        "source": "rag",
        "raw_text": chunk,
    }


def _parse_amount_range_text(amount_text: Any) -> tuple[int | None, int | None, bool]:
    text = str(amount_text or "").strip()
    if not text:
        return None, None, False

    values = [parse_korean_money(match.group(0)) for match in re.finditer(r"[0-9][0-9,]*(?:억|천만|백만|십만|만)?원?", text)]
    values = [value for value in values if value is not None]
    if not values:
        return None, None, False

    if any(keyword in text for keyword in ["최소", "이상"]):
        min_amount = values[0]
    else:
        min_amount = min(values)

    if any(keyword in text for keyword in ["최대", "이하", "한도"]):
        max_amount = values[-1]
    else:
        max_amount = max(values)

    if len(values) == 1 and ("최대" in text or "이하" in text):
        min_amount = None
    if len(values) == 1 and ("최소" in text or "이상" in text):
        max_amount = None

    return min_amount, max_amount, True


def _extract_amount_text(text: str) -> str:
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if "만원" in stripped or "원" in stripped or "억" in stripped:
            if any(keyword in stripped for keyword in ["최소", "최대", "가입", "납입", "금액", "한도"]):
                return _clean_table_cell(stripped)
    return ""


def _split_summary_blocks(text: str) -> list[str]:
    lines = str(text or "").splitlines()
    blocks: list[list[str]] = []
    current: list[str] = []

    for line in lines:
        if line.strip().startswith("## "):
            if current:
                blocks.append(current)
            current = [line]
        elif current:
            current.append(line)
    if current:
        blocks.append(current)
    return ["\n".join(block).strip() for block in blocks if any(item.strip() for item in block)]


def _extract_detail_rows(block: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        if re.fullmatch(r"[\|\-\s:*]+", stripped):
            continue
        cells = [_clean_table_cell(cell) for cell in stripped.split("|") if cell.strip()]
        if len(cells) != 2:
            continue
        rows.append((cells[0], cells[1]))
    return rows


def _extract_heading_product_name(block: str) -> str:
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped.startswith("##"):
            continue
        cleaned = re.sub(r"^\s*##+\s*", "", stripped)
        cleaned = re.sub(r"^\d+[.)]?\s*", "", cleaned)
        cleaned = re.sub(r"[0-9]\ufe0f\u20e3", " ", cleaned)
        cleaned = cleaned.replace("\ufe0f", " ").replace("\u20e3", " ")
        cleaned = re.sub(r"[①②③④⑤⑥⑦⑧⑨⑩]", " ", cleaned)
        cleaned = re.sub(r"\b추천\s*상품\b", " ", cleaned)
        cleaned = cleaned.replace("**", "")
        cleaned = cleaned.replace("‘", "").replace("’", "").replace('"', "").replace("'", "")
        cleaned = re.sub(r"\([^)]*개월[^)]*\)", " ", cleaned)
        cleaned = re.sub(r"\((정기예금|적금|예금)\)", " ", cleaned)
        cleaned = re.sub(r"\s*[–-]\s*추천\s*[0-9]+위\s*$", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return _clean_product_name(cleaned)
    return ""


def _extract_heading_term_text(block: str) -> str:
    for line in block.splitlines():
        match = re.search(r"\(([0-9]{1,2}\s*개월)\)", line)
        if match:
            return match.group(1)
    return ""


def _infer_product_type(product_name: str, text: str) -> str | None:
    if "적금" in product_name or "적금" in text:
        return "적금"
    if "예금" in product_name or "예금" in text:
        return "예금"
    return None


def _extract_join_channels(text: str) -> list[str]:
    channels = []
    for keyword in ["지점", "KB스타뱅킹", "인터넷", "모바일", "비대면"]:
        if keyword in text and keyword not in channels:
            channels.append(keyword)
    return channels


def _extract_preferential_conditions_from_text(text: str) -> list[dict[str, Any]]:
    conditions = []
    for label, keyword in [
        ("급여이체", "급여이체"),
        ("자동이체", "자동이체"),
        ("카드사용", "카드"),
        ("주거래", "주거래"),
        ("마케팅동의", "마케팅"),
    ]:
        if keyword in text:
            conditions.append({"name": label, "condition": f"{label} 관련 우대조건", "matched": None})
    return conditions


def _extract_first_match(text: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return None


def _extract_product_name_from_chunk(chunk: str) -> str:
    blocked_prefixes = (
        "▪ 적용조건",
        "적용조건",
        "아래 항목별 적용조건",
        "신규가입일 당시",
        "영업점 및 KB국민은행",
        "- 가입:",
        "- 가입 대상:",
        "- 가입금액:",
        "- 월 납입:",
        "- 우대 금리:",
        "- 기본 금리:",
        "| 항목 | 내용 |",
    )

    heading_name = _extract_heading_product_name(chunk)
    if heading_name and heading_name not in INVALID_PRODUCT_NAME_VALUES and heading_name not in NON_PRODUCT_HEADINGS:
        return heading_name

    for line in chunk.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("[") and "출처:" in stripped:
            continue
        if "출처:" in stripped and "/ p." in stripped:
            continue
        if any(stripped.startswith(prefix) for prefix in blocked_prefixes):
            continue
        if len(stripped) > 100:
            continue
        cleaned = _clean_product_name(stripped)
        # 표 셀 행(| ... |)은 product_name으로 사용하지 않음
        if "|" in cleaned:
            continue
        if cleaned and cleaned not in INVALID_PRODUCT_NAME_VALUES and cleaned not in NON_PRODUCT_HEADINGS and any(keyword in cleaned for keyword in ["KB", "적금", "예금", "군인", "청년"]):
            return cleaned

    return ""


def _get_cell(cells: list[str], column_index: dict[str, int], column_name: str) -> str | None:
    index = column_index.get(column_name)
    if index is None or index >= len(cells):
        return None
    return cells[index]


def _clean_table_cell(value: str) -> str:
    cleaned = str(value or "").strip().strip("|")
    cleaned = cleaned.replace("**", "")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _clean_product_name(value: str) -> str:
    cleaned = _clean_table_cell(value)
    cleaned = cleaned.replace("‘", "").replace("’", "").replace('"', "").replace("'", "")
    cleaned = re.sub(r"[0-9]\ufe0f\u20e3", " ", cleaned)
    cleaned = cleaned.replace("\ufe0f", " ").replace("\u20e3", " ")
    cleaned = re.sub(r"[①②③④⑤⑥⑦⑧⑨⑩]", " ", cleaned)
    cleaned = re.sub(r"\b추천\s*상품\b", " ", cleaned)
    cleaned = re.sub(r"^\d+[.)]?\s*", "", cleaned)
    cleaned = re.sub(r"^\s*[–-]\s*", "", cleaned)
    cleaned = re.sub(r"\([^)]*개월[^)]*\)", " ", cleaned)
    cleaned = re.sub(r"\s*\((정기예금|적금|예금)\)\s*$", "", cleaned)
    cleaned = re.sub(r"\s*[–-]\s*추천\s*[0-9]+위\s*$", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _parse_rate(value: Any) -> float | None:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(value or ""))
    return float(match.group(1)) if match else None


def _parse_months(value: Any) -> int | None:
    match = re.search(r"([0-9]{1,2})\s*개월", str(value or ""))
    return int(match.group(1)) if match else None


def _parse_min_amount(value: Any) -> int | None:
    min_amount, _, _ = _parse_amount_range_text(value)
    return min_amount


def _parse_max_amount(value: Any) -> int | None:
    _, max_amount, _ = _parse_amount_range_text(value)
    return max_amount


PRODUCT_TOOLS = [search_terms]
