"""
Eligibility Agent 도구
- 다른 Agent가 모아둔 고객/상품 정보를 비교해 자격 판단에 필요한 값을 만듭니다.
"""
import json
import re
from typing import Any

from langchain_core.tools import tool


def parse_customer_profile(raw_profile: Any) -> dict:
    # 고객 관련 자유형 텍스트를 읽어 판단에 필요한 핵심 정보만 뽑습니다.
    if isinstance(raw_profile, dict):
        profile = dict(raw_profile)
        profile["raw_text"] = json.dumps(raw_profile, ensure_ascii=False)
        profile["is_soldier"] = _infer_is_soldier(profile.get("job"), profile["raw_text"])
        return profile

    text = str(raw_profile or "")
    lowered = text.lower()
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
        "monthly_saving_amount": _extract_int(
            text,
            [
                r"월\s*가용\s*저축액[:\s]*([0-9,]+)",
                r"가용저축액[:\s]*([0-9,]+)",
                r"월\s*납입\s*가능\s*금액[:\s]*([0-9,]+)",
                r"available_monthly_saving\s*\|\s*([0-9,]+)",
            ],
        ),
        "salary_transfer": "급여이체" in lowered and "없" not in lowered,
        "auto_transfer": "자동이체" in lowered and "없" not in lowered,
        "card_usage": "카드" in lowered and "없" not in lowered,
        "main_bank": "주거래" in lowered,
        "marketing_agree": "마케팅" in lowered and "동의" in lowered,
        "is_soldier": _infer_is_soldier(job, text),
        "is_miso_target": any(keyword in lowered for keyword in ["미소드림", "서민", "저소득", "취약계층"]),
        "raw_text": text,
    }


def parse_customer_accounts(raw_accounts: Any) -> list[dict]:
    # 고객 계좌/거래 텍스트를 단순 목록 형태로 정리합니다.
    if isinstance(raw_accounts, list):
        return raw_accounts
    text = str(raw_accounts or "")
    if not text.strip():
        return []
    return [{"raw_text": line.strip()} for line in text.splitlines() if line.strip()]


def extract_product_candidates(raw_products: Any) -> list[dict]:
    # Product Agent가 넘긴 원문에서 상품 후보를 상품 단위로 나눕니다.
    if isinstance(raw_products, list):
        return [dict(item) for item in raw_products]

    text = str(raw_products or "").strip()
    if not text:
        return []

    chunks = _split_product_chunks(text)
    if not chunks:
        chunks = [text]

    return [_build_product_candidate(chunk) for chunk in chunks]


def _build_product_candidate(text: str) -> dict:
    # 상품 설명 한 덩어리에서 가입 판단에 필요한 규칙성 정보만 뽑습니다.
    lowered = text.lower()
    product_name = _extract_product_name(text)
    combined_text = f"{product_name}\n{text}".lower()
    military_markers = [
        "군인 전용",
        "장병",
        "직업군인",
        "군 복무",
        "현역",
        "장기간부",
        "군간부",
        "병사",
        "부사관",
        "장교",
    ]

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
                r"최저\s*금액[:\s]*([0-9,]+)",
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
        "sale_closed": any(keyword in lowered for keyword in ["판매 종료", "판매종료", "판매 중단", "신규 불가"]),
        "military_only": any(keyword in combined_text for keyword in military_markers),
        "miso_dream_only": "미소드림" in text,
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
    # 고객 정보와 상품 조건을 비교해 상품별 eligibility 결과를 만듭니다.
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
            reasons.append("군인 전용 상품으로 고객 직업 조건이 맞지 않습니다.")
        else:
            check_required.append("군 복무/직업 정보")

    if product.get("miso_dream_only"):
        if customer_profile.get("is_miso_target") is True:
            pass
        elif customer_profile.get("raw_text"):
            check_required.append("미소드림적금 대상 조건 충족 여부")
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
    # 우대조건 충족 여부를 충족/미충족 두 목록으로 나눕니다.
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
    # 구조화 결과를 사람이 읽기 쉬운 요약 문자열로 바꿉니다.
    if not results:
        return "가입 가능 여부를 판단할 상품 정보가 없어 자격 결과를 만들지 못했습니다."

    lines = ["가입 가능 여부를 상품별로 정리했습니다."]
    for item in results:
        marker = "✅" if item["eligible"] else "❌"
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
    # 긴 상품 원문을 상품별 블록으로 최대한 안정적으로 나눕니다.
    dashed_chunks = [chunk.strip() for chunk in re.split(r"\n\s*---+\s*\n", text) if chunk.strip()]
    if len(dashed_chunks) > 1:
        return dashed_chunks

    lines = text.splitlines()
    chunks = []
    current = []
    for line in lines:
        if _is_product_start_line(line):
            if current:
                chunks.append("\n".join(current).strip())
                current = []
        if line.strip() or current:
            current.append(line)
    if current:
        chunks.append("\n".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def _extract_product_name(text: str) -> str:
    # 상품 블록 안에서 실제 상품명처럼 보이는 줄을 우선 선택합니다.
    for line in text.splitlines():
        stripped = line.strip().lstrip("#").strip()
        stripped = re.sub(r"^[-*]\s+", "", stripped)
        if not stripped:
            continue
        if stripped.startswith("[") and "]" in stripped:
            stripped = stripped.split("]", 1)[1].strip()
        if _looks_like_product_name(stripped):
            return stripped[:80]
    for line in text.splitlines():
        stripped = line.strip().lstrip("#").strip()
        stripped = re.sub(r"^[-*]\s+", "", stripped)
        if stripped and not _looks_like_heading(stripped):
            return stripped[:80]
    return "미확인 상품"


def _is_product_start_line(line: str) -> bool:
    stripped = line.strip().lstrip("#").strip()
    stripped = re.sub(r"^[-*]\s+", "", stripped)
    if not stripped:
        return False
    if _looks_like_heading(stripped):
        return False
    return _looks_like_product_name(stripped)


def _looks_like_heading(text: str) -> bool:
    heading_markers = [
        "추천 상품",
        "상품 후보",
        "군인 전용 포함",
        "아래는",
        "고객_",
        "가입 대상",
        "가입 조건",
        "금리",
        "월 납입",
        "필요 서류",
    ]
    return any(marker in text for marker in heading_markers)


def _looks_like_product_name(text: str) -> bool:
    if _looks_like_heading(text):
        return False
    if len(text) > 80:
        return False
    return any(keyword in text for keyword in ["적금", "예금", "부금"])


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


def _infer_is_soldier(job: Any, raw_text: str) -> bool:
    markers = ["군인", "장병", "직업군인", "부사관", "장교", "군간부", "병사", "하사", "중사", "대위"]

    job_text = str(job or "").strip().lower()
    if job_text and any(marker in job_text for marker in markers):
        return True

    explicit_patterns = [
        r"(?:직업|customer_job)[:\s|]*(군인|직업군인|부사관|장교|병사|군간부)",
        r"군\s*복무[:\s|]*(예|참|중)",
        r"군인\s*여부[:\s|]*(예|참|yes|true)",
    ]
    lowered = str(raw_text or "").lower()

    return any(re.search(pattern, lowered) for pattern in explicit_patterns)


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
