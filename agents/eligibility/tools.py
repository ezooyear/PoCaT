"""
Eligibility parsing and rule helpers.
"""

from __future__ import annotations

import json
import re
from datetime import date
from typing import Any

from langchain_core.tools import tool


MISO_KEYWORDS = ["미소저축", "서민", "저소득", "취약계층"]
SOLDIER_MARKERS = ["군인", "직업군인", "부사관", "장교", "병사", "군간부", "사관", "중사", "대위"]
PRODUCT_INVALID_PREFIXES = [
    "가입",
    "가입채널",
    "가입 조건",
    "가입 가능",
    "기본 금리",
    "우대 금리",
    "우대조건",
    "우대 조건",
]
PRODUCT_HEADING_MARKERS = [
    "추천 상품",
    "상품 정보",
    "고객_",
    "가입 조건",
    "기본 금리",
    "우대 금리",
    "필요 서류",
]


def parse_customer_profile(raw_profile: Any) -> dict:
    if isinstance(raw_profile, dict):
        return _normalize_customer_profile_dict(dict(raw_profile))

    text = str(raw_profile or "")
    lowered = text.lower()
    table_row = _parse_first_table_row(text)

    if table_row:
        return _normalize_customer_profile_dict(table_row)

    annual_income = _extract_int(
        text,
        [
            r"annual_income\s*\|\s*([0-9,]+)",
            r"연소득[:\s]*([0-9,]+)",
        ],
    )
    job = _extract_text(
        text,
        [
            r"직업[:\s]*([^\n,|]+)",
            r"customer_job\s*\|\s*([^\n|]+)",
            r"\|\s*customer_job\s*\|\s*([^\n|]+)",
        ],
    )

    return {
        "age": _extract_int(text, [r"나이[:\s]*([0-9]{1,2})", r"만\s*([0-9]{1,2})세"]),
        "job": job,
        "annual_income": annual_income,
        "income": annual_income,
        "income_level": _extract_text(
            text,
            [r"income_level\s*\|\s*([^\n|]+)", r"소득수준[:\s]*([^\n,|]+)"],
        ),
        "monthly_saving_amount": _extract_int(
            text,
            [
                r"월\s*가용\s*저축액[:\s]*([0-9,]+)",
                r"가용저축액[:\s]*([0-9,]+)",
                r"월\s*납입\s*가능\s*금액[:\s]*([0-9,]+)",
                r"available_monthly_saving\s*\|\s*([0-9,]+)",
            ],
        ),
        "salary_transfer": _extract_yes_no_value(text, ["급여이체", "salary_transfer"]) is True,
        "auto_transfer": _extract_yes_no_value(text, ["자동이체", "auto_transfer"]) is True,
        "card_usage": _extract_yes_no_value(text, ["카드 사용", "카드사용", "card_usage"]) is True,
        "main_bank": _extract_yes_no_value(text, ["주거래", "main_bank"]) is True,
        "marketing_agree": _extract_yes_no_value(text, ["마케팅동의", "마케팅 동의", "marketing_agree"]) is True,
        "is_soldier": _infer_is_soldier(job, text),
        "is_miso_target": any(keyword in lowered for keyword in MISO_KEYWORDS),
        "transaction_months": _extract_int(
            text,
            [
                r"transaction_months\s*\|\s*([0-9]{1,2})",
                r"거래\s*개월[:\s]*([0-9]{1,2})",
                r"거래개월[:\s]*([0-9]{1,2})",
                r"거래개월수[:\s]*([0-9]{1,2})",
            ],
        ),
        "raw_text": text,
    }


def _normalize_customer_profile_dict(raw_profile: dict[str, Any]) -> dict[str, Any]:
    profile = dict(raw_profile)
    job = profile.get("job") or profile.get("customer_job")
    annual_income = _to_int(profile.get("annual_income") or profile.get("income"))
    raw_text = json.dumps(raw_profile, ensure_ascii=False)

    normalized = {
        "age": profile.get("age") or _extract_age_from_birth_date(profile.get("birth_date")),
        "job": job,
        "annual_income": annual_income,
        "income": annual_income,
        "income_level": profile.get("income_level"),
        "monthly_saving_amount": _to_int(
            profile.get("monthly_saving_amount") or profile.get("available_monthly_saving")
        ),
        "salary_transfer": _to_bool(
            profile.get("salary_transfer") if "salary_transfer" in profile else profile.get("salary_transfer_yn")
        ),
        "auto_transfer": _to_bool(
            profile.get("auto_transfer") if "auto_transfer" in profile else profile.get("auto_transfer_yn")
        ),
        "card_usage": _to_bool(
            profile.get("card_usage") if "card_usage" in profile else profile.get("card_usage_yn")
        ),
        "main_bank": _to_bool(
            profile.get("main_bank") if "main_bank" in profile else profile.get("main_bank_yn")
        ),
        "marketing_agree": _to_bool(
            profile.get("marketing_agree") if "marketing_agree" in profile else profile.get("marketing_agree_yn")
        ),
        "transaction_months": _to_int(profile.get("transaction_months")),
        "raw_text": raw_text,
    }
    normalized["is_soldier"] = _infer_is_soldier(job, raw_text)
    normalized["is_miso_target"] = any(keyword in raw_text.lower() for keyword in MISO_KEYWORDS)
    return normalized


def parse_customer_accounts(raw_accounts: Any) -> list[dict]:
    if isinstance(raw_accounts, list):
        return raw_accounts
    text = str(raw_accounts or "")
    if not text.strip():
        return []
    return [{"raw_text": line.strip()} for line in text.splitlines() if line.strip()]


def extract_product_candidates(raw_products: Any) -> list[dict]:
    if isinstance(raw_products, list):
        return [dict(item) for item in raw_products if isinstance(item, dict)]

    text = str(raw_products or "").strip()
    if not text:
        return []

    chunks = _split_product_chunks(text) or [text]
    return [_build_product_candidate(chunk) for chunk in chunks if chunk.strip()]


def extract_product_candidates_from_markdown_table(text: Any) -> list[dict]:
    raw_text = str(text or "")
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    header_index = -1
    product_name_index = -1

    for index, line in enumerate(lines):
        if "|" not in line:
            continue
        cells = [_clean_table_cell(cell) for cell in line.split("|")]
        if "상품명" in cells:
            header_index = index
            product_name_index = cells.index("상품명")
            break

    if header_index < 0 or product_name_index < 0:
        return []

    candidates: list[dict] = []
    for line in lines[header_index + 1 :]:
        if "|" not in line:
            continue
        if re.fullmatch(r"[\|\-\s:]+", line):
            continue
        cells = [_clean_table_cell(cell) for cell in line.split("|")]
        if product_name_index >= len(cells):
            continue
        product_name = _clean_product_name(cells[product_name_index])
        if not product_name or _is_excluded_product_name_line(product_name):
            continue
        if _looks_like_product_name(product_name):
            candidates.append({"product_name": product_name, "raw_text": line})
    return candidates


def _build_product_candidate(text: str) -> dict:
    lowered = text.lower()
    product_name = _extract_product_name(text)
    combined_text = f"{product_name}\n{text}".lower()

    return {
        "product_name": product_name,
        "raw_text": text,
        "min_age": _extract_int(
            text,
            [
                r"만\s*([0-9]{1,2})세\s*이상",
                r"최소\s*연령[:\s]*([0-9]{1,2})",
                r"가입\s*가능\s*연령[:\s]*([0-9]{1,2})세\s*이상",
            ],
        ),
        "max_age": _extract_int(
            text,
            [
                r"만\s*([0-9]{1,2})세\s*이하",
                r"최대\s*연령[:\s]*([0-9]{1,2})",
                r"가입\s*가능\s*연령[:\s]*([0-9]{1,2})세\s*이하",
            ],
        ),
        "min_amount": _extract_int(
            text,
            [
                r"최소\s*가입\s*금액[:\s]*([0-9,]+)",
                r"최소\s*금액[:\s]*([0-9,]+)",
                r"월\s*최소\s*납입[:\s]*([0-9,]+)",
            ],
        ),
        "max_amount": _extract_int(
            text,
            [
                r"최대\s*가입\s*금액[:\s]*([0-9,]+)",
                r"최고\s*금액[:\s]*([0-9,]+)",
                r"월\s*최대\s*납입[:\s]*([0-9,]+)",
            ],
        ),
        "sale_closed": any(keyword in lowered for keyword in ["판매 종료", "판매중단", "신규 불가"]),
        "military_only": any(keyword in combined_text for keyword in SOLDIER_MARKERS + ["장병", "나라", "군 복무"]),
        "miso_dream_only": "미소저축" in text,
        "bonus_keywords": [
            label
            for label, keyword in [
                ("급여이체", "급여이체"),
                ("자동이체", "자동이체"),
                ("카드사용", "카드"),
                ("주거래", "주거래"),
                ("마케팅동의", "마케팅"),
            ]
            if keyword in text
        ],
    }


def evaluate_product_eligibility(customer_profile: dict, customer_accounts: list[dict], product: dict) -> dict:
    reasons: list[str] = []
    check_required: list[str] = []

    age = customer_profile.get("age")
    monthly_saving_amount = customer_profile.get("monthly_saving_amount")

    if product.get("sale_closed"):
        reasons.append("판매 종료 또는 신규 가입 불가 상품으로 해석됩니다.")

    min_age = product.get("min_age")
    max_age = product.get("max_age")
    if min_age is not None or max_age is not None:
        if age is None:
            check_required.append("고객 나이 정보")
        else:
            if min_age is not None and age < min_age:
                reasons.append(f"연령 제한 미충족: 최소 {min_age}세 이상 필요")
            if max_age is not None and age > max_age:
                reasons.append(f"연령 제한 미충족: 최대 {max_age}세 이하 필요")

    min_amount = product.get("min_amount")
    max_amount = product.get("max_amount")
    if min_amount is not None or max_amount is not None:
        if monthly_saving_amount is None:
            check_required.append("고객 월 가용 저축액")
        else:
            if min_amount is not None and monthly_saving_amount < min_amount:
                reasons.append(f"가입금액 제한 미충족: 최소 {min_amount:,}원 필요")
            if max_amount is not None and monthly_saving_amount > max_amount:
                check_required.append(f"최대 가입금액 {max_amount:,}원 초과 여부 상세 확인 필요")

    if product.get("military_only"):
        if customer_profile.get("is_soldier") is True:
            pass
        elif customer_profile.get("job"):
            reasons.append("군인 전용 상품으로 고객 직업 조건과 맞지 않습니다.")
        else:
            check_required.append("군 복무/직업 정보")

    if product.get("miso_dream_only"):
        if customer_profile.get("is_miso_target") is True:
            pass
        elif customer_profile.get("raw_text"):
            check_required.append("미소저축 적금 대상 조건 충족 여부")
        else:
            check_required.append("고객 지원 대상 정보")

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
    }


def evaluate_bonus_conditions(customer_profile: dict, customer_accounts: list[dict], product: dict) -> tuple[list[str], list[str]]:
    account_text = " ".join(item.get("raw_text", "") for item in customer_accounts)
    merged = f"{customer_profile.get('raw_text', '')}\n{account_text}".lower()

    condition_map = {
        "급여이체": customer_profile.get("salary_transfer") or "급여이체" in merged,
        "자동이체": customer_profile.get("auto_transfer") or "자동이체" in merged,
        "카드사용": customer_profile.get("card_usage") or "카드" in merged,
        "주거래": customer_profile.get("main_bank") or "주거래" in merged,
        "마케팅동의": customer_profile.get("marketing_agree") or ("마케팅" in merged and "동의" in merged),
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
        marker = "가능" if item["eligible"] else "확인필요"
        lines.append(f"- {marker} {item['product_name']}")
        if item["ineligibility_reasons"]:
            lines.append(f"  사유: {', '.join(item['ineligibility_reasons'])}")
        if item["bonus_conditions_met"]:
            lines.append(f"  우대 충족: {', '.join(item['bonus_conditions_met'])}")
        if item["bonus_conditions_missing"]:
            lines.append(f"  우대 미충족: {', '.join(item['bonus_conditions_missing'])}")
        if item["check_required"]:
            lines.append(f"  추가 확인: {', '.join(item['check_required'])}")
    return "\n".join(lines)


def _split_product_chunks(text: str) -> list[str]:
    dashed_chunks = [chunk.strip() for chunk in re.split(r"\n\s*---+\s*\n", text) if chunk.strip()]
    if len(dashed_chunks) > 1:
        return dashed_chunks

    lines = text.splitlines()
    chunks: list[str] = []
    current: list[str] = []

    for line in lines:
        if _is_product_start_line(line) and current:
            chunks.append("\n".join(current).strip())
            current = []
        if line.strip() or current:
            current.append(line)

    if current:
        chunks.append("\n".join(current).strip())

    return [chunk for chunk in chunks if chunk]


def _extract_product_name(text: str) -> str:
    table_candidates = extract_product_candidates_from_markdown_table(text)
    if table_candidates:
        return str(table_candidates[0].get("product_name") or "미확인 상품")

    for extractor in (_extract_heading_product_name, _extract_labeled_product_name):
        product_name = extractor(text)
        if product_name:
            return product_name

    for line in text.splitlines():
        stripped = _normalize_product_line(line)
        if not stripped or _is_excluded_product_name_line(stripped):
            continue
        if stripped.startswith("[") and "]" in stripped:
            stripped = stripped.split("]", 1)[1].strip()
        if _looks_like_product_name(stripped):
            return stripped[:80]

    return "미확인 상품"


def _extract_heading_product_name(text: str) -> str:
    patterns = [
        r"^\s*#+\s*\d+[.)]?\s*(.+?)\s*(?:\([^)]*\))?\s*$",
        r"^\s*#+\s*(.+?)\s*(?:\([^)]*\))?\s*$",
    ]
    for line in text.splitlines():
        for pattern in patterns:
            match = re.match(pattern, line.strip())
            if not match:
                continue
            candidate = _clean_product_name(match.group(1))
            if candidate and _looks_like_product_name(candidate):
                return candidate
    return ""


def _extract_labeled_product_name(text: str) -> str:
    for line in text.splitlines():
        match = re.search(r"(?:상품명|product_name)[:\s]+(.+)$", line.strip(), flags=re.IGNORECASE)
        if not match:
            continue
        candidate = _clean_product_name(match.group(1))
        if candidate:
            return candidate
    return ""


def _normalize_product_line(line: str) -> str:
    stripped = line.strip().lstrip("#").strip()
    stripped = re.sub(r"^[-*]\s+", "", stripped)
    return stripped


def _clean_product_name(value: str) -> str:
    text = re.sub(r"^\d+[.)\]]*\s*", "", str(value or "").strip())
    text = text.replace("**", "").replace("“", "").replace("”", "").replace("‘", "").replace("’", "")
    trailing_match = re.search(r"\s*\(([^)]*)\)\s*$", text)
    if trailing_match:
        trailing_text = trailing_match.group(1).strip()
        if any(keyword in trailing_text for keyword in ["적금", "예금", "정기", "상품 유형"]):
            text = re.sub(r"\s*\([^)]*\)\s*$", "", text)
    return text.strip()[:80]


def _is_product_start_line(line: str) -> bool:
    stripped = _normalize_product_line(line)
    if not stripped or _looks_like_heading(stripped) or _is_excluded_product_name_line(stripped):
        return False
    return _looks_like_product_name(stripped)


def _is_excluded_product_name_line(text: str) -> bool:
    stripped = str(text or "").strip()
    if not stripped:
        return True
    if stripped.startswith("|") and stripped.endswith("|"):
        return True
    if stripped.startswith("▪ 적용조건") or stripped.startswith("적용조건"):
        return True
    if "아래 항목별 적용조건" in stripped:
        return True
    if "신규가입일 당시" in stripped:
        return True
    if stripped == "영업점 및 KB국민은행":
        return True
    if stripped.lower().startswith("product_name |"):
        return True
    return any(stripped.startswith(prefix) for prefix in PRODUCT_INVALID_PREFIXES)


def _clean_table_cell(value: str) -> str:
    cleaned = str(value or "").strip()
    cleaned = cleaned.strip("|")
    cleaned = cleaned.replace("**", "")
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _looks_like_heading(text: str) -> bool:
    return any(marker in text for marker in PRODUCT_HEADING_MARKERS)


def _looks_like_product_name(text: str) -> bool:
    if _looks_like_heading(text) or _is_excluded_product_name_line(text):
        return False
    if len(text) > 80 or len(text.split()) > 8:
        return False
    return any(keyword in text for keyword in ["적금", "예금", "저축", "통장"])


def _extract_int(text: str, patterns: list[str]) -> int | None:
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            digits = re.sub(r"[^0-9]", "", match.group(1))
            if digits:
                return int(digits)
    return None


def _extract_text(text: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return None


def _extract_yes_no_value(text: str, labels: list[str]) -> bool | None:
    yes_tokens = {"예", "네", "yes", "true", "1", "y"}
    no_tokens = {"아니오", "no", "false", "0", "n"}
    patterns = []
    for label in labels:
        patterns.extend(
            [
                rf"{re.escape(label)}(?:_yn)?\s*\|\s*([^\n|]+)",
                rf"{re.escape(label)}[:\s|]*([^\n|]+)",
            ]
        )

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        value = match.group(1).strip().lower()
        if value in yes_tokens:
            return True
        if value in no_tokens:
            return False
    return None


def _infer_is_soldier(job: Any, raw_text: str) -> bool:
    job_text = str(job or "").strip().lower()
    if job_text and any(marker in job_text for marker in SOLDIER_MARKERS):
        return True

    lowered = str(raw_text or "").lower()
    if any(marker in lowered for marker in SOLDIER_MARKERS):
        return True

    explicit_patterns = [
        r"(?:직업|customer_job)[:\s|]*(군인|직업군인|부사관|장교|병사|군간부)",
        r"군\s*복무[:\s|]*(중|yes|true)",
        r"군인\s*여부[:\s|]*(예|yes|true)",
    ]
    return any(re.search(pattern, lowered) for pattern in explicit_patterns)


def _parse_first_table_row(text: str) -> dict[str, str]:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    header = None

    for line in lines:
        if "|" not in line:
            continue

        parts = [part.strip() for part in line.split("|")]
        lower_parts = [part.lower() for part in parts]

        if header is None and (
            "customer_id" in lower_parts
            or "customer_name" in lower_parts
            or "birth_date" in lower_parts
            or "customer_job" in lower_parts
        ):
            header = parts
            continue

        if header is not None:
            if all(not part or set(part) <= {"-"} for part in parts):
                continue
            if len(parts) == len(header):
                return dict(zip(header, parts))

    return {}


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"예", "네", "yes", "true", "1", "y"}


def _to_int(value: Any) -> int | None:
    digits = re.sub(r"[^0-9]", "", str(value or ""))
    if not digits:
        return None
    return int(digits)


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
    """Compare customer and product info and return structured eligibility."""
    profile = parse_customer_profile(customer_profile)
    product = _build_product_candidate(product_info)
    result = evaluate_product_eligibility(profile, [], product)
    return json.dumps(result, ensure_ascii=False)


@tool
def evaluate_bonus_rate(customer_profile: str, customer_accounts: str = "") -> str:
    """Return bonus condition matches from customer info and accounts."""
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
    """Filter a product list down to eligible products only."""
    profile = parse_customer_profile(customer_profile)
    products = extract_product_candidates(products_info)
    results = [evaluate_product_eligibility(profile, [], product) for product in products]
    filtered = [item for item in results if item["eligible"]]
    return json.dumps(filtered, ensure_ascii=False)


ELIGIBILITY_TOOLS = [evaluate_eligibility, evaluate_bonus_rate, filter_eligible_products]
