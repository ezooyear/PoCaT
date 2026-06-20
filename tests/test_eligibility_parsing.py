import json
import sys
import types
import unittest
from unittest.mock import patch

mcp_module = types.ModuleType("mcp")
mcp_module.ClientSession = object
mcp_module.types = types.SimpleNamespace()
client_module = types.ModuleType("mcp.client")
streamable_http_module = types.ModuleType("mcp.client.streamable_http")
streamable_http_module.streamable_http_client = object()

sys.modules.setdefault("mcp", mcp_module)
sys.modules.setdefault("mcp.client", client_module)
sys.modules.setdefault("mcp.client.streamable_http", streamable_http_module)

fake_vectorstore_module = types.ModuleType("db.vectorstore")
fake_vectorstore_module.search_products = lambda query, k=3: []
sys.modules.setdefault("db.vectorstore", fake_vectorstore_module)

from agents.customer import agent as customer_agent
from agents.eligibility import agent as eligibility_agent
from agents.eligibility.tools import (
    extract_product_candidates,
    extract_product_candidates_from_markdown_table,
    parse_customer_profile,
)
from agents.product import agent as product_agent
from agents.product.tools import extract_products_from_product_summary, parse_korean_money
from agents.validation import agent as validation_agent


RAW_CUSTOMER_PROFILE = """customer_name | birth_date | customer_job | annual_income | income_level | main_bank_yn | salary_transfer_yn | auto_transfer_yn | card_usage_yn | marketing_agree_yn | transaction_months | available_monthly_saving
고객_223 | 2004-08-20 | 군인 | 2900 | 낮음 | 아니오 | 예 | 아니오 | 아니오 | 아니오 | 7 | 100000
"""


class FakeTool:
    def __init__(self, name: str, result: str):
        self.name = name
        self.result = result

    def invoke(self, args: dict):
        return self.result


class EligibilityParsingTest(unittest.TestCase):
    def test_parse_customer_profile_maps_columns_by_name(self):
        profile = parse_customer_profile(RAW_CUSTOMER_PROFILE)

        self.assertEqual(profile["job"], "군인")
        self.assertEqual(profile["annual_income"], 2900)
        self.assertEqual(profile["income"], 2900)
        self.assertEqual(profile["monthly_saving_amount"], 100000)
        self.assertTrue(profile["salary_transfer"])
        self.assertTrue(profile["is_soldier"])

    def test_customer_agent_fast_path_stores_structured_profile(self):
        fake_tools = [
            FakeTool("get_customer_profile", RAW_CUSTOMER_PROFILE),
            FakeTool("get_customer_accounts", "account_number | product_name\n123 | 테스트적금"),
            FakeTool("get_payment_history", "없음"),
        ]

        original_tools = customer_agent.CUSTOMER_TOOLS
        customer_agent.CUSTOMER_TOOLS = fake_tools
        try:
            result = customer_agent.customer_agent_node(
                {
                    "customer_id": 223,
                    "messages": [],
                    "agent_outputs": {},
                    "completed_agents": [],
                    "current_step": 0,
                }
            )
        finally:
            customer_agent.CUSTOMER_TOOLS = original_tools

        profile = result["customer_result"]["result"]["customer_profile"]
        self.assertEqual(profile["job"], "군인")
        self.assertTrue(profile["salary_transfer"])
        self.assertEqual(profile["monthly_saving_amount"], 100000)
        self.assertTrue(profile["is_soldier"])

    def test_extract_product_name_prefers_heading_over_join_channel_line(self):
        summary = """## 1. KB 장기간부 도약적금(정기예금)
- 가입: 지점, KB스타뱅킹
"""

        products = extract_product_candidates(summary)

        self.assertEqual(products[0]["product_name"], "KB 장기간부 도약적금")
        self.assertNotEqual(products[0]["product_name"], "- 가입: 지점, KB스타뱅킹")

    def test_load_product_candidates_uses_top_level_products_first(self):
        products, source = eligibility_agent._load_product_candidates_with_source(
            state={},
            product_result={"products": [{"product_name": "KB 직장인·군인 적금(특별우대)", "raw_text": "raw"}]},
            agent_outputs={},
        )

        self.assertEqual([item["product_name"] for item in products], ["KB 직장인·군인 적금(특별우대)"])
        self.assertEqual(source, "product_result.products")

    def test_load_product_candidates_reads_nested_result_products(self):
        products, source = eligibility_agent._load_product_candidates_with_source(
            state={},
            product_result={
                "result": {
                    "products": [{"product_name": "KB 직장인·군인 적금(특별우대)", "raw_text": "raw"}]
                }
            },
            agent_outputs={},
        )

        self.assertEqual([item["product_name"] for item in products], ["KB 직장인·군인 적금(특별우대)"])
        self.assertEqual(source, "product_result.result.products")

    def test_extract_product_name_from_markdown_table(self):
        summary = """| 순위 | 상품명 | 상품 유형 | 기본 금리* |
|------|--------|-----------|------------|
| 1 | **KB 직장인·군인 적금(특별우대)** | 적금 | **1.80 %** |
"""

        products = extract_product_candidates_from_markdown_table(summary)

        self.assertEqual([item["product_name"] for item in products], ["KB 직장인·군인 적금(특별우대)"])

    def test_extract_structured_products_from_product_summary_table(self):
        summary = """| 순위 | 상품명 | 상품 유형 | 기본 금리* |
|------|--------|-----------|------------|
| 1 | **KB 직장인·군인 적금(특별우대)** | 적금 | **1.80 %** |
"""

        products = extract_products_from_product_summary(summary)

        self.assertEqual(products[0]["product_name"], "KB 직장인·군인 적금(특별우대)")
        self.assertEqual(products[0]["product_type"], "적금")
        self.assertEqual(products[0]["base_rate"], 1.8)

    def test_extract_structured_products_from_detail_table(self):
        summary = """## 1️⃣ ① **‘장기간부 도약적금’** (36개월)
| 항목 | 내용 |
|------|------|
| **상품명** | 장기간부 도약적금 |
| **상품 유형** | 적금 |
| **가입 대상·나이** | 만 19세 ~ 64세 |
| **기본 금리** | 연 5.5% |
"""

        products = extract_products_from_product_summary(summary)

        self.assertEqual(len(products), 1)
        self.assertEqual(products[0]["product_name"], "장기간부 도약적금")
        self.assertEqual(products[0]["product_type"], "적금")
        self.assertEqual(products[0]["base_rate"], 5.5)
        self.assertEqual(products[0]["source"], "summary_detail_table")

    def test_excluded_condition_line_is_not_product_name(self):
        line = "▪ 적용조건 : 아래 항목별 적용조건을 충족한 경우 신규가입일 당시 영업점 및 KB국민은행"

        products = extract_product_candidates(line)

        self.assertEqual(products[0]["product_name"], "미확인 상품")

    def test_mock_state_prefers_structured_product_result_schema(self):
        mock_state = {
            "customer_result": {
                "result": {
                    "profile": {
                        "age": 21,
                        "job": "군인",
                        "income": 2900,
                        "monthly_saving_amount": 100000,
                        "salary_transfer": True,
                        "auto_transfer": False,
                        "card_usage": False,
                        "main_bank": False,
                        "is_soldier": True,
                    }
                }
            },
            "product_result": {
                "status": "success",
                "products": [
                    {
                        "product_name": "KB 직장인·군인 적금(특별우대)",
                        "bank": "KB국민은행",
                        "product_type": "적금",
                        "base_rate": 1.8,
                        "max_rate": 4.0,
                        "eligibility_text": "군인 가입 가능, 만 19세 이상 가입 가능, 월 10만원 납입 가능",
                        "raw_text": "가입 대상: 군인, 만 19세 이상 가입 가능",
                        "preferential_conditions": [
                            {
                                "name": "급여이체",
                                "condition": "급여이체 시 우대금리 제공",
                                "rate": 2.2,
                            }
                        ],
                    }
                ],
                "summary": "",
            },
        }

        customer_profile = eligibility_agent._load_customer_profile(
            mock_state,
            mock_state["customer_result"],
            mock_state.get("agent_outputs", {}),
        )
        product_candidates, source = eligibility_agent._load_product_candidates_with_source(
            mock_state,
            mock_state["product_result"],
            mock_state.get("agent_outputs", {}),
        )
        results, fallback_notes = eligibility_agent._build_guarded_eligibility_results(
            customer_profile=customer_profile,
            customer_accounts=[],
            product_candidates=product_candidates,
            user_constraints={},
        )

        self.assertEqual(source, "product_result.products")
        self.assertEqual([item["product_name"] for item in results], ["KB 직장인·군인 적금(특별우대)"])
        self.assertEqual(fallback_notes, [])

    def test_invalid_product_name_values_are_rejected(self):
        results, fallback_notes = eligibility_agent._build_guarded_eligibility_results(
            customer_profile={
                "age": 21,
                "job": "군인",
                "income": 2900,
                "monthly_saving_amount": 100000,
                "salary_transfer": True,
                "auto_transfer": False,
            },
            customer_accounts=[],
            product_candidates=[
                {"product_name": "상품 유형", "raw_text": "raw"},
                {"product_name": "기본 금리", "raw_text": "raw"},
            ],
            user_constraints={},
        )

        self.assertIn("invalid_product_name", fallback_notes)
        self.assertTrue(all(item["status"] == "invalid_product" for item in results))
        self.assertTrue(all("product_name" in item["invalid_fields"] for item in results))

    def test_product_agent_result_is_json_serializable_without_circular_reference(self):
        mock_loop_result = {
            "messages": [types.SimpleNamespace(content="summary", usage_metadata={"input_tokens": 10, "output_tokens": 5})],
            "agent_outputs": {
                "product_agent": {
                    "status": "success",
                    "result": {
                        "summary": """## 1️⃣ ① **‘장기간부 도약적금’** (36개월)
| 항목 | 내용 |
|------|------|
| **상품명** | 장기간부 도약적금 |
| **상품 유형** | 적금 |
| **가입 대상·나이** | 만 19세 ~ 64세 |
| **기본 금리** | 연 5.5% |
""",
                        "tool_results": [],
                    },
                }
            },
            "product_result": {
                "status": "success",
                "result": {
                    "summary": """## 1️⃣ ① **‘장기간부 도약적금’** (36개월)
| 항목 | 내용 |
|------|------|
| **상품명** | 장기간부 도약적금 |
| **상품 유형** | 적금 |
| **가입 대상·나이** | 만 19세 ~ 64세 |
| **기본 금리** | 연 5.5% |
""",
                    "tool_results": [],
                },
            },
            "current_step": 1,
            "completed_agents": ["product_agent"],
        }

        with patch.object(product_agent, "run_agent_loop", return_value=mock_loop_result):
            result = product_agent.product_agent_node({"messages": [], "agent_outputs": {}, "completed_agents": []})

        json.dumps(result, ensure_ascii=False)
        self.assertEqual(result["product_result"]["structured_product_names"], ["장기간부 도약적금"])
        self.assertNotIn("product_result", result["product_result"])

    def test_validation_state_is_json_serializable(self):
        product_result = {
            "status": "success",
            "summary": "",
            "products": [
                {
                    "product_name": "장기간부 도약적금",
                    "product_type": "적금",
                    "base_rate": 5.5,
                    "source": "summary_detail_table",
                }
            ],
            "structured_product_count": 1,
            "structured_product_names": ["장기간부 도약적금"],
            "result": {
                "summary": "",
                "products": [
                    {
                        "product_name": "장기간부 도약적금",
                        "product_type": "적금",
                        "base_rate": 5.5,
                        "source": "summary_detail_table",
                    }
                ],
            },
        }
        state = {
            "task_type": "recommendation",
            "user_query": "군인에게 맞는 적금 추천해줘",
            "customer_id": 223,
            "customer_result": {
                "result": {
                    "customer_profile": {
                        "age": 21,
                        "job": "군인",
                        "income": 2900,
                        "monthly_saving_amount": 100000,
                        "salary_transfer": True,
                        "auto_transfer": False,
                    }
                }
            },
            "product_result": product_result,
            "recommend_result": {
                "result": {
                    "summary": "추천 결과",
                    "recommendations": [{"product_name": "장기간부 도약적금"}],
                }
            },
            "agent_outputs": {
                "product_agent": {"product_result": product_result},
                "recommend_agent": {"result": {"summary": "추천 결과"}},
            },
            "completed_agents": ["customer_agent", "product_agent", "eligibility_agent", "recommend_agent"],
            "current_step": 4,
        }

        with patch.object(validation_agent, "_should_run_llm_validation", return_value=False):
            validation_result = validation_agent.validation_agent_node(state)

        json.dumps(state, ensure_ascii=False)
        json.dumps(validation_result, ensure_ascii=False, default=str)

    def test_transaction_months_is_not_treated_as_period_constraint(self):
        state = {
            "user_query": "거래 개월: 7개월\n월 저축 가능액: 100,000원\n추천할 만한 예적금 상품 알려줘"
        }

        constraints = eligibility_agent._extract_user_constraints(state)
        customer_profile = eligibility_agent._enrich_customer_profile(
            {"raw_text": state["user_query"]},
            source="test",
        )
        results, fallback_notes = eligibility_agent._build_guarded_eligibility_results(
            customer_profile={
                **customer_profile,
                "age": 21,
                "job": "군인",
                "income": 2900,
                "monthly_saving_amount": 100000,
                "salary_transfer": True,
                "auto_transfer": False,
            },
            customer_accounts=[],
            product_candidates=[
                {
                    "product_name": "KB 장기간부 도약적금",
                    "product_type": "적금",
                    "raw_text": "가입 대상: 군인, 만 19세 이상 가입 가능",
                    "eligibility_text": "군인 가입 가능",
                }
            ],
            user_constraints=constraints,
        )

        self.assertIsNone(constraints["period_months"])
        self.assertEqual(customer_profile.get("transaction_months"), 7)
        self.assertNotIn("period_mismatch", fallback_notes)
        self.assertTrue(all("period_mismatch" not in item.get("reasons", []) for item in results))

    def test_explicit_period_is_treated_as_period_constraint(self):
        constraints = eligibility_agent._extract_user_constraints({"user_query": "7개월짜리 적금 추천해줘"})
        self.assertEqual(constraints["period_months"], 7)

    def test_product_name_is_normalized_from_recommendation_heading(self):
        summary = "## 1️⃣ ① **‘장기간부 도약적금’** (36개월) – 추천 1위"
        products = extract_products_from_product_summary(summary)

        self.assertEqual(products[0]["product_name"], "장기간부 도약적금")
        self.assertEqual(products[0]["term_months"], 36)

    def test_parse_korean_money_normalizes_units(self):
        self.assertEqual(parse_korean_money("30만원"), 300000)
        self.assertEqual(parse_korean_money("10만원"), 100000)
        self.assertEqual(parse_korean_money("5만원"), 50000)
        self.assertEqual(parse_korean_money("100,000원"), 100000)
        self.assertEqual(parse_korean_money("5천만원"), 50000000)
        self.assertEqual(parse_korean_money("5억원"), 500000000)

    def test_savings_amount_limit_marks_eligible_when_within_range(self):
        result = {
            "eligible": True,
            "status": "eligible",
            "reasons": [],
        }
        fallback_notes = eligibility_agent._apply_user_constraints(
            result,
            {
                "product_name": "KB 장기간부 도약적금",
                "product_type": "적금",
                "min_monthly_amount": 100000,
                "max_monthly_amount": 300000,
                "amount_text": "최소 10만원, 최대 30만원",
            },
            {"monthly_amount": 100000, "period_months": None},
        )

        self.assertEqual(result["status"], "eligible")
        self.assertTrue(result["eligible"])
        self.assertEqual(result["amount_check"]["result"], "pass")
        self.assertNotIn("monthly_amount_exceeds_limit", fallback_notes)

    def test_amount_parse_uncertain_does_not_hard_reject(self):
        result = {
            "eligible": True,
            "status": "eligible",
            "reasons": [],
        }
        fallback_notes = eligibility_agent._apply_user_constraints(
            result,
            {
                "product_name": "KB 장기간부 도약적금",
                "product_type": "적금",
                "max_monthly_amount": 30,
                "amount_text": "최대 30만원",
            },
            {"monthly_amount": 100000, "period_months": None},
        )

        self.assertNotEqual(result["status"], "rejected")
        self.assertNotIn("monthly_amount_exceeds_limit", fallback_notes)
        self.assertIn(result["amount_check"]["result"], {"pass", "needs_check"})


    # --- 수정 요청 7: 새 테스트 A, B, C, D ---

    def test_A_recommendation_table_not_confused_with_situation_summary(self):
        """현재 상황 요약 heading이 있어도 추천 상품 표에서만 상품을 추출해야 한다."""
        summary = """\
## 1️⃣ 현재 상황 요약
| 항목 | 내용 |
|---|---|
| 연령 | 20대 |
| 직업 | 군인 |
| 월 저축 가능액 | 100,000원 |

## 2️⃣ 추천 상품
| 순위 | 상품명 | 상품 유형 | 기본 금리* | 우대 금리(가능 시) | 총 예상 금리 | 최소·최대 가입 금액 | 가입 가능 기간 | 우대조건 적용 여부 |
|---|---|---|---|---|---|---|---|---|
| 1 | KB 스타 적금(30만원 이하 자유적립식) | 적금 | 연 2.0% | +0.5%p | 연 2.5% | 최소 10만원 – 최대 30만원 | 12개월 / 24개월 / 36개월 | 급여이체 충족 |
"""
        products = extract_products_from_product_summary(summary)

        names = [p["product_name"] for p in products]
        self.assertIn("KB 스타 적금(30만원 이하 자유적립식)", names)
        self.assertNotIn("현재 상황 요약", names)
        self.assertEqual(len(products), 1)

    def test_B_only_situation_summary_returns_empty_products(self):
        """추천 상품 표 없이 현재 상황 요약만 있으면 products를 비워야 한다."""
        summary = """\
## 1️⃣ 현재 상황 요약
| 항목 | 내용 |
|---|---|
| 연령 | 20대 |
| 직업 | 군인 |
"""
        products = extract_products_from_product_summary(summary)

        self.assertEqual(products, [])

    def test_C_eligibility_rejects_non_product_section_name(self):
        """eligibility_agent가 섹션 제목을 product_name으로 받으면 invalid_product로 처리해야 한다."""
        results, fallback_notes = eligibility_agent._build_guarded_eligibility_results(
            customer_profile={
                "age": 21,
                "job": "군인",
                "income": 2900,
                "monthly_saving_amount": 100000,
                "salary_transfer": True,
                "auto_transfer": False,
            },
            customer_accounts=[],
            product_candidates=[{"product_name": "현재 상황 요약"}],
            user_constraints={},
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "invalid_product")
        self.assertFalse(results[0]["eligible"])
        self.assertIn("product_name", results[0]["invalid_fields"])
        self.assertIn("invalid_product_name", fallback_notes)

    def test_D_recommendation_table_parses_amounts_and_rates(self):
        """추천 상품 표에서 금액 범위와 금리를 정확히 파싱해야 한다."""
        summary = """\
| 순위 | 상품명 | 상품 유형 | 기본 금리* | 우대 금리(가능 시) | 총 예상 금리 | 최소·최대 가입 금액 | 가입 가능 기간 | 우대조건 적용 여부 |
|---|---|---|---|---|---|---|---|---|
| 1 | KB 스타 적금(30만원 이하 자유적립식) | 적금 | 연 2.0% | +0.5%p | 연 2.5% | 최소 10만원 – 최대 30만원 | 12개월 / 24개월 / 36개월 | 급여이체 충족 |
"""
        products = extract_products_from_product_summary(summary)

        self.assertEqual(len(products), 1)
        p = products[0]
        self.assertEqual(p["product_name"], "KB 스타 적금(30만원 이하 자유적립식)")
        self.assertEqual(p["product_type"], "적금")
        self.assertEqual(p["base_rate"], 2.0)
        self.assertEqual(p["min_monthly_amount"], 100000)
        self.assertEqual(p["max_monthly_amount"], 300000)
        self.assertEqual(p["source"], "recommendation_table")


if __name__ == "__main__":
    unittest.main()
