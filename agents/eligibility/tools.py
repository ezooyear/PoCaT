"""
Eligibility agent tools.
"""
from __future__ import annotations

import json
import re
from datetime import date
from typing import Any

from langchain_core.tools import tool

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

GENERIC_PRODUCT_NAMES = {
    "미확인 상품",
    "다음 단계",
    "최종 추천",
    "추천 상품",
    "추천 결과",
    "상품 유형",
    "유형",
    "정액적립식 예금",
    "자유적립식 예금",
    "전용 우대 적금",
    "군인 전용 우대 적금",
    "직업군인 우대 적금",
    "도약 적금",
    "입출금통장",
    "KB국민은행 입출금통장",
    "KB 예적금 상품군",
    "최적의 적금",
    "고액예금",
    "KB 고액예금",
}


def parse_customer_profile(raw_profile: Any) -> dict:
    if isinstance(raw_profile, dict):
        profile = dict(raw_profile)
        job = _normalize_job_text(profile.get("job") or profile.get("customer_job"))
        raw_text = json.dumps(raw_profile, ensure_ascii=False)
        return {
            "age": profile.get("age") or _extract_age_from_birth_date(profile.get("birth_date")),
            "job": job,
            "monthly_saving_amount": _to_int(
                profile.get("monthly_saving_amount") or profile.get("available_monthly_saving")
            ),
            "salary_transfer": _to_bool(profile.get("salary_transfer") or profile.get("salary_transfer_yn")),
            "auto_transfer": _to_bool(profile.get("auto_transfer") or profile.get("auto_transfer_yn")),
            "card_usage": _to_bool(profile.get("card_usage") or profile.get("card_usage_yn")),
            "main_bank": _to_bool(profile.get("main_bank") or profile.get("main_bank_yn")),
            "marketing_agree": _to_bool(profile.get("marketing_agree") or profile.get("marketing_agree_yn")),
            "is_soldier": _infer_is_soldier(job, raw_text),
            "is_miso_target": _infer_is_miso_target(raw_text),
            "raw_text": raw_text,
        }

    text = str(raw_profile or "").strip()
    if not text:
        return {}

    table_row = _parse_first_table_row(text)
    if table_row:
        return parse_customer_profile(table_row)

    job = _normalize_job_text(
        _extract_text(
            text,
            [
                r"직업[:\s]*([^\n,|]+)",
                r"customer_job\s*\|\s*([^\n|]+)",
                r"\*\*직업:\*\*\s*([^\n]+)",
            ],
        )
    )

    age = _extract_int(
        text,
        [
            r"현재\s*([0-9]{1,2})세",
            r"만\s*([0-9]{1,2})세",
            r"age\s*\|\s*([0-9]{1,2})",
        ],
    )
    if age is None:
        birth_date = _extract_text(text, [r"생년월일[:\s]*([0-9]{4}-[0-9]{2}-[0-9]{2})"])
        age = _extract_age_from_birth_date(birth_date)

    return {
        "age": age,
        "job": job,
        "monthly_saving_amount": _extract_int(
            text,
            [
                r"월\s*가용\s*저축액[:\s\*]*([0-9,]+)",
                r"가용\s*저축액[:\s\*]*([0-9,]+)",
                r"available_monthly_saving\s*\|\s*([0-9,]+)",
            ],
        ),
        "salary_transfer": _extract_labeled_bool(text, ["급여이체 여부", "급여이체", "salary_transfer_yn"]),
        "auto_transfer": _extract_labeled_bool(text, ["자동이체 여부", "자동이체", "auto_transfer_yn"]),
        "card_usage": _extract_labeled_bool(text, ["카드 사용 여부", "카드사용 여부", "card_usage_yn"]),
        "main_bank": _extract_labeled_bool(text, ["주거래은행", "주거래은행 여부", "main_bank_yn"]),
        "marketing_agree": _extract_labeled_bool(text, ["마케팅 동의 여부", "marketing_agree_yn"]),
        "is_soldier": _infer_is_soldier(job, text),
        "is_miso_target": _infer_is_miso_target(text),
        "raw_text": text,
    }


def parse_customer_accounts(raw_accounts: Any) -> list[dict]:
    if isinstance(raw_accounts, list):
        return raw_accounts

    text = str(raw_accounts or "").strip()
    if not text:
        return []

    parsed_rows = _parse_table_rows(text)
    if parsed_rows:
        return parsed_rows

    return [{"raw_text": line.strip()} for line in text.splitlines() if line.strip()]


def extract_product_candidates(raw_products: Any) -> list[dict]:
    if isinstance(raw_products, list):
        chunks = []
        for item in raw_products:
            if isinstance(item, dict):
                product_name = _normalize_product_name(item.get("product_name") or "")
                raw_text = str(item.get("raw_text") or product_name).strip()
                if raw_text:
                    chunks.append((product_name, raw_text))
            else:
                text = str(item or "").strip()
                if text:
                    chunks.extend((None, chunk) for chunk in _split_product_chunks(text))
    else:
        text = str(raw_products or "").strip()
        if not text:
            return []
        chunks = [(None, chunk) for chunk in _split_product_chunks(text)]

    candidates = []
    seen = set()

    for explicit_name, chunk in chunks:
        candidate = _build_product_candidate(chunk)
        if explicit_name:
            candidate["product_name"] = _normalize_product_name(explicit_name)

        product_name = candidate.get("product_name") or ""
        canonical = _canonicalize_product_name(product_name)
        if not canonical:
            continue
        if _looks_like_document_name(product_name):
            continue
        if _looks_like_advice_text(product_name):
            continue
        if _looks_like_generic_product_label(product_name):
            continue
        if canonical in seen:
            continue

        seen.add(canonical)
        candidates.append(candidate)

    return candidates


def evaluate_product_eligibility(customer_profile: dict, customer_accounts: list[dict], product: dict) -> dict:
    reasons: list[str] = []
    check_required: list[str] = []

    age = customer_profile.get("age")
    monthly_saving_amount = customer_profile.get("monthly_saving_amount")

    if product.get("sale_closed"):
        reasons.append("판매종료 또는 신규가입 불가로 해석됩니다.")

    min_age = product.get("min_age")
    max_age = product.get("max_age")
    if min_age is not None or max_age is not None:
        if age is None:
            check_required.append("고객 나이 정보")
        else:
            if min_age is not None and age < min_age:
                reasons.append(f"연령 조건 미충족: 최소 {min_age}세 이상 필요")
            if max_age is not None and age > max_age:
                reasons.append(f"연령 조건 미충족: 최대 {max_age}세 이하 필요")

    min_amount = product.get("min_amount")
    max_amount = product.get("max_amount")
    if min_amount is not None or max_amount is not None:
        if monthly_saving_amount is None:
            check_required.append("고객 월 가용 저축액")
        else:
            if min_amount is not None and monthly_saving_amount < min_amount:
                reasons.append(f"가입금액 조건 미충족: 최소 {min_amount:,}원 필요")
            if max_amount is not None and monthly_saving_amount > max_amount:
                check_required.append(f"최대 납입한도 {max_amount:,}원 초과 여부 상세 확인 필요")

    if product.get("military_only"):
        if customer_profile.get("is_soldier") is not True:
            if customer_profile.get("job"):
                reasons.append("군인 전용 상품으로 고객 직업 조건이 맞지 않습니다.")
            else:
                check_required.append("군복무/직업 정보")

    if product.get("miso_dream_only") and customer_profile.get("is_miso_target") is not True:
        check_required.append("청년/정책지원 대상 여부")

    bonus_conditions_met, bonus_conditions_missing = evaluate_bonus_conditions(
        customer_profile,
        customer_accounts,
        product,
    )

    return {
        "product_name": product.get("product_name") or "미확인 상품",
        "eligible": not reasons,
        "ineligibility_reasons": reasons,
        "bonus_conditions_met": bonus_conditions_met,
        "bonus_conditions_missing": bonus_conditions_missing,
        "check_required": _dedupe(check_required),
        "military_only": bool(product.get("military_only")),
        "miso_dream_only": bool(product.get("miso_dream_only")),
    }


def evaluate_bonus_conditions(customer_profile: dict, customer_accounts: list[dict], product: dict) -> tuple[list[str], list[str]]:
    condition_map = {
        "급여이체": bool(customer_profile.get("salary_transfer")),
        "자동이체": bool(customer_profile.get("auto_transfer")),
        "카드사용": bool(customer_profile.get("card_usage")),
        "주거래": bool(customer_profile.get("main_bank")),
        "마케팅동의": bool(customer_profile.get("marketing_agree")),
    }

    product_bonus = product.get("bonus_keywords") or []
    met = [name for name in product_bonus if condition_map.get(name)]
    missing = [name for name in product_bonus if not condition_map.get(name)]
    return met, missing


def build_eligibility_summary(results: list[dict]) -> str:
    if not results:
        return "가입 가능 여부를 판단할 상품 정보가 없어 자격 결과를 만들지 못했습니다."

    lines = ["가입 가능 여부를 상품별로 정리했습니다."]
    for item in results:
        marker = "✅" if item.get("eligible") else "❌"
        lines.append(f"- {marker} {item.get('product_name', '미확인 상품')}")
        if item.get("ineligibility_reasons"):
            lines.append(f"  사유: {', '.join(item['ineligibility_reasons'])}")
        if item.get("bonus_conditions_met"):
            lines.append(f"  우대 충족: {', '.join(item['bonus_conditions_met'])}")
        if item.get("bonus_conditions_missing"):
            lines.append(f"  우대 미충족: {', '.join(item['bonus_conditions_missing'])}")
        if item.get("check_required"):
            lines.append(f"  추가 확인: {', '.join(item['check_required'])}")
    return "\n".join(lines)


def _build_product_candidate(text: str) -> dict:
    lowered = text.lower()
    product_name = _normalize_product_name(_extract_product_name(text))
    combined_text = f"{product_name}\n{text}".lower()

    military_markers = [
        "군인 전용",
        "나라사랑",
        "직업군인",
        "군 복무",
        "현역",
        "국방",
        "장병",
        "부사관",
        "사관",
    ]

    bonus_keywords = []
    if "급여이체" in text:
        bonus_keywords.append("급여이체")
    if "자동이체" in text:
        bonus_keywords.append("자동이체")
    if "카드" in text:
        bonus_keywords.append("카드사용")
    if "주거래" in text:
        bonus_keywords.append("주거래")
    if "마케팅" in text:
        bonus_keywords.append("마케팅동의")

    return {
        "product_name": product_name,
        "raw_text": text,
        "min_age": _extract_int(text, [r"만\s*([0-9]{1,2})세\s*이상", r"최소\s*연령[:\s]*([0-9]{1,2})"]),
        "max_age": _extract_int(text, [r"만\s*([0-9]{1,2})세\s*이하", r"최대\s*연령[:\s]*([0-9]{1,2})"]),
        "min_amount": _extract_int(
            text,
            [
                r"최소\s*가입\s*금액[:\s]*([0-9,]+)",
                r"최소\s*금액[:\s]*([0-9,]+)",
                r"최소\s*납입[:\s]*([0-9,]+)",
            ],
        ),
        "max_amount": _extract_int(
            text,
            [
                r"최대\s*가입\s*금액[:\s]*([0-9,]+)",
                r"최대\s*금액[:\s]*([0-9,]+)",
                r"최대\s*납입[:\s]*([0-9,]+)",
                r"월\s*최대\s*([0-9,]+)",
            ],
        ),
        "sale_closed": any(keyword in lowered for keyword in ["판매 종료", "판매종료", "판매 중단", "신규 가입 불가"]),
        "military_only": any(keyword in combined_text for keyword in military_markers),
        "miso_dream_only": "미소" in combined_text or "청년드림" in combined_text,
        "bonus_keywords": _dedupe(bonus_keywords),
    }


def _split_product_chunks(text: str) -> list[str]:
    dashed_chunks = [chunk.strip() for chunk in re.split(r"\n\s*---+\s*\n", str(text or "")) if chunk.strip()]

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
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_chunks.append(chunk)
    return unique_chunks


def _extract_product_name(text: str) -> str:
    table_name = _extract_product_name_from_table(text)
    if table_name:
        return table_name[:80]

    markdown_name = _extract_markdown_product_name(text)
    if markdown_name:
        return markdown_name[:80]

    for line in text.splitlines():
        stripped = line.strip().lstrip("#").strip()
        stripped = re.sub(r"^[-*]\s+", "", stripped)
        if not stripped:
            continue
        if stripped.startswith("[") and "]" in stripped:
            stripped = stripped.split("]", 1)[1].strip()
        if _looks_like_product_name(stripped):
            return _normalize_product_name(stripped[:80])

    for line in text.splitlines():
        stripped = line.strip().lstrip("#").strip()
        stripped = re.sub(r"^[-*]\s+", "", stripped)
        if stripped and not _looks_like_heading(stripped):
            return _normalize_product_name(stripped[:80])

    return "미확인 상품"


def _is_product_start_line(line: str) -> bool:
    stripped = line.strip().lstrip("#").strip()
    stripped = re.sub(r"^[-*]\s+", "", stripped)
    if not stripped or _looks_like_heading(stripped):
        return False
    return _looks_like_product_name(stripped)


def _looks_like_heading(text: str) -> bool:
    if any(marker in text for marker in TITLE_LIKE_MARKERS):
        return True
    heading_markers = [
        "추천 상품",
        "상품 정보",
        "고객_",
        "가입 대상",
        "가입 조건",
        "금리",
        "월 납입",
        "주의사항",
        "예상 이자",
    ]
    return any(marker in text for marker in heading_markers)


def _looks_like_product_name(text: str) -> bool:
    if _looks_like_heading(text):
        return False
    if len(text) > 80:
        return False
    if any(marker in text for marker in TITLE_LIKE_MARKERS):
        return False
    if any(marker in text for marker in ADVICE_LIKE_MARKERS):
        return False
    if _looks_like_document_name(text):
        return False
    if _looks_like_advice_text(text):
        return False
    if any(
        phrase in text
        for phrase in [
            "가입 대상",
            "가입 금액",
            "우대 금리",
            "기본 금리",
            "계약 기간",
            "가입일",
            "만기",
            "본인 명의",
        ]
    ):
        return False
    if _looks_like_generic_product_label(text):
        return False
    return any(keyword in text for keyword in PRODUCT_SUFFIXES)


def _extract_product_name_from_table(text: str) -> str:
    for line in text.splitlines():
        if "|" not in line:
            continue
        parts = _split_pipe_row(line)
        if len(parts) < 2:
            continue
        left = re.sub(r"[*` ]", "", parts[0]).lower()
        right = re.sub(r"[*`]", "", parts[1]).strip()
        if left in {"상품명", "product_name"} and _looks_like_product_name(right):
            return _normalize_product_name(right)
    return ""


def _extract_markdown_product_name(text: str) -> str:
    patterns = [
        r"\*\*(KB[^*\n]+(?:적금|예금|부금|통장|저축)[^*\n]*)\*\*",
        r"#+\s*([^\n#]*KB[^\n#]*(?:적금|예금|부금|통장|저축)[^\n#]*)",
        r"([A-Za-z0-9가-힣\s()]+(?:적금|예금|부금|통장|저축)[A-Za-z0-9가-힣\s()]*)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            candidate = match.group(1).strip()
            if _looks_like_product_name(candidate):
                return _normalize_product_name(candidate)
    return ""


def _normalize_product_name(product_name: str) -> str:
    text = str(product_name or "").strip()
    text = re.sub(r"^#+\s*", "", text)
    text = re.sub(r"^\s*[-*]\s*", "", text)
    text = re.sub(r"^\s*\d+[.)]\s*", "", text)
    text = re.sub(r"^[✅☑✔•▪①②③④⑤⑥⑦⑧⑨⑩]+\s*", "", text)
    text = re.sub(r"^\s*\*\*(상품명|상품)\*\*\s*[:：]?\s*", "", text)
    text = re.sub(r"^\s*(상품명|상품)\s*[:：]?\s*", "", text)
    text = text.replace("**", "").strip()
    text = re.sub(r"\((?:[^()]*(?:개월|우대|연\s*[0-9]+(?:\.[0-9]+)?\s*%)\s*[^()]*)\)$", "", text).strip()
    text = re.sub(r"\((?:[^()]*(?:군인\s*전용|자유적립식|정액적립식|정기예금|적금\s*상품|예금\s*상품)\s*[^()]*)\)$", "", text).strip()
    text = re.sub(r"\s{2,}", " ", text)
    return text[:80]


def _normalize_job_text(job: Any) -> str:
    text = str(job or "").strip()
    text = text.replace("**", "").strip()
    text = re.sub(r"^\s*[:：]+\s*", "", text)
    return text


def _looks_like_document_name(product_name: Any) -> bool:
    text = str(product_name or "").strip()
    if not text:
        return False
    return any(marker in text for marker in ["상품설명서", "약관", "설명서", "상품 안내문", "가입안내"])


def _looks_like_advice_text(text: Any) -> bool:
    candidate = str(text or "").strip()
    if not candidate:
        return False
    if any(marker in candidate for marker in ADVICE_LIKE_MARKERS):
        return True
    return any(
        marker in candidate
        for marker in [
            "원하시면",
            "선택",
            "사용하고 싶다면",
            "정보",
            "유리",
            "추천 이유",
            "고객님",
        ]
    )


def _canonicalize_product_name(product_name: Any) -> str:
    text = _normalize_product_name(str(product_name or ""))
    text = re.sub(r"[\s]+", "", text).lower()
    if text.startswith("kb"):
        text = text[2:]
    return text


def _looks_like_generic_product_label(product_name: str) -> bool:
    text = str(product_name or "").strip()
    if text in GENERIC_PRODUCT_NAMES:
        return True
    if any(marker in text for marker in TITLE_LIKE_MARKERS):
        return True
    generic_contains = [
        "전용 적금",
        "입출금통장",
        "보유",
        "장기예금",
        "거치식",
        "청약통장 포함",
        "상품군",
        "최적",
        "적합한",
        "고액예금",
    ]
    return any(token in text for token in generic_contains)


def _parse_first_table_row(text: str) -> dict[str, str]:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    header = None

    for line in lines:
        if "|" not in line:
            continue
        parts = _split_pipe_row(line)
        if header is None and "customer_id" in line.lower():
            header = parts
            continue
        if header is not None:
            if _is_table_separator(parts):
                continue
            if len(parts) == len(header):
                return dict(zip(header, parts))

    return {}


def _parse_table_rows(text: str) -> list[dict[str, str]]:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    header = None
    rows = []

    for line in lines:
        if "|" not in line:
            continue
        parts = _split_pipe_row(line)
        if header is None and "customer_id" in line.lower():
            header = parts
            continue
        if header is not None:
            if _is_table_separator(parts):
                continue
            if len(parts) == len(header):
                row = dict(zip(header, parts))
                row["raw_text"] = line
                rows.append(row)

    return rows


def _split_pipe_row(line: str) -> list[str]:
    parts = [part.strip() for part in str(line).split("|")]
    if parts and parts[0] == "":
        parts = parts[1:]
    if parts and parts[-1] == "":
        parts = parts[:-1]
    return parts


def _is_table_separator(parts: list[str]) -> bool:
    if not parts:
        return False
    return all(part and set(part) <= {"-", ":"} for part in parts)


def _extract_text(text: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return None


def _extract_int(text: str, patterns: list[str]) -> int | None:
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            digits = re.sub(r"[^0-9]", "", match.group(1))
            if digits:
                return int(digits)
    return None


def _extract_labeled_bool(text: str, labels: list[str]) -> bool:
    for label in labels:
        patterns = [
            rf"{re.escape(label)}[:\s\*]*([^\n|]+)",
            rf"{re.escape(label)}\s*\|\s*([^\n|]+)",
        ]
        value = _extract_text(text, patterns)
        if value is None:
            continue
        lowered = value.strip().lower()
        if any(token in lowered for token in ["예", "yes", "true", "1", "y"]):
            return True
        if any(token in lowered for token in ["아니오", "아님", "없음", "no", "false", "0", "n"]):
            return False
    return False


def _to_bool(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return text in {"예", "yes", "true", "1", "y"}


def _to_int(value: Any) -> int | None:
    digits = re.sub(r"[^0-9]", "", str(value or ""))
    return int(digits) if digits else None


def _extract_age_from_birth_date(value: Any) -> int | None:
    text = str(value or "").strip()
    match = re.match(r"(\d{4})-(\d{2})-(\d{2})", text)
    if not match:
        return None

    birth_year, birth_month, birth_day = map(int, match.groups())
    today = date.today()
    age = today.year - birth_year
    if (today.month, today.day) < (birth_month, birth_day):
        age -= 1
    return age


def _infer_is_soldier(job: Any, raw_text: str) -> bool:
    markers = ["군인", "직업군인", "부사관", "사관", "국방", "장병", "현역", "복무"]
    job_text = str(job or "").strip().lower()
    raw = str(raw_text or "").lower()
    return any(marker in job_text for marker in markers) or any(marker in raw for marker in markers)


def _infer_is_miso_target(raw_text: str) -> bool:
    lowered = str(raw_text or "").lower()
    return any(keyword in lowered for keyword in ["미소", "청년드림", "정책", "취약계층"])


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    output = []
    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output


@tool
def evaluate_eligibility(customer_profile: str, product_info: str) -> str:
    """고객 정보와 상품 정보를 비교해 구조화된 가입 가능 여부를 반환합니다."""
    profile = parse_customer_profile(customer_profile)
    product = _build_product_candidate(product_info)
    result = evaluate_product_eligibility(profile, [], product)
    return json.dumps(result, ensure_ascii=False)


@tool
def evaluate_bonus_rate(customer_profile: str, customer_accounts: str = "") -> str:
    """고객 정보와 계좌 정보를 바탕으로 우대조건 충족 여부를 반환합니다."""
    profile = parse_customer_profile(customer_profile)
    accounts = parse_customer_accounts(customer_accounts)
    product = {"bonus_keywords": ["급여이체", "자동이체", "카드사용", "주거래", "마케팅동의"]}
    met, missing = evaluate_bonus_conditions(profile, accounts, product)
    return json.dumps(
        {"bonus_conditions_met": met, "bonus_conditions_missing": missing},
        ensure_ascii=False,
    )


@tool
def filter_eligible_products(customer_profile: str, products_info: str) -> str:
    """전체 상품 중 가입 가능한 상품만 골라 반환합니다."""
    profile = parse_customer_profile(customer_profile)
    products = extract_product_candidates(products_info)
    results = [evaluate_product_eligibility(profile, [], product) for product in products]
    filtered = [item for item in results if item["eligible"]]
    return json.dumps(filtered, ensure_ascii=False)


ELIGIBILITY_TOOLS = [evaluate_eligibility, evaluate_bonus_rate, filter_eligible_products]
