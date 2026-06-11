"""
Validation Agent 단독 테스트 파일

실행:
python scripts/test_validation.py
"""

import json
import os
import sys
from pprint import pprint

# 프로젝트 루트 import를 위해 경로 추가
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from agents.base import make_agent_result
from agents.validation.agent import validation_agent_node


def print_result(title: str, result: dict):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


def test_rule_only_customer_lookup():
    """
    단순 고객 조회 테스트
    - task_type이 customer_lookup이면 LLM 검증 없이 rule 기반 검증만 수행되는지 확인
    """

    state = {
        "user_query": "내 가입 상품 보여줘",
        "task_type": "customer_lookup",
        "plan": ["customer_agent", "validation_agent"],
        "current_step": 1,
        "current_agent": "customer_agent",
        "completed_agents": ["customer_agent"],
        "errors": [],

        "customer_result": make_agent_result(
            status="success",
            result={
                "customer_id": 101,
                "customer_name": "고객_101",
                "accounts": [
                    {
                        "account_id": 1,
                        "product_name": "KB Star 정기예금",
                        "current_balance": 3000000,
                        "account_status": "ACTIVE",
                    }
                ],
            },
            evidence=[],
            error=None,
        ),

        "agent_outputs": {
            "customer_agent": make_agent_result(
                status="success",
                result={
                    "customer_id": 101,
                    "customer_name": "고객_101",
                    "accounts": [
                        {
                            "account_id": 1,
                            "product_name": "KB Star 정기예금",
                            "current_balance": 3000000,
                            "account_status": "ACTIVE",
                        }
                    ],
                },
                evidence=[],
                error=None,
            )
        },
    }

    result = validation_agent_node(state)
    print_result("TEST 1: customer_lookup / rule only", result)


def test_missing_required_result():
    """
    필수 결과 누락 테스트
    - recommendation인데 recommend_result가 없으면 issue가 잡히는지 확인
    """

    state = {
        "user_query": "나한테 맞는 적금 추천해줘",
        "task_type": "recommendation",
        "plan": [
            "customer_agent",
            "financial_agent",
            "product_agent",
            "eligibility_agent",
            "recommend_agent",
            "validation_agent",
        ],
        "current_step": 4,
        "current_agent": "eligibility_agent",
        "completed_agents": [
            "customer_agent",
            "financial_agent",
            "product_agent",
            "eligibility_agent",
        ],
        "errors": [],

        "customer_result": make_agent_result(
            status="success",
            result={
                "customer_id": 101,
                "customer_name": "고객_101",
                "monthly_saving_capacity": 200000,
            },
            evidence=[],
            error=None,
        ),

        "financial_result": make_agent_result(
            status="success",
            result={
                "monthly_saving_capacity": 200000,
                "total_balance": 13000000,
            },
            evidence=[],
            error=None,
        ),

        "product_result": make_agent_result(
            status="success",
            result={
                "products": [
                    {
                        "product_name": "KB국민ONE적금",
                        "product_type": "적금",
                        "base_rate": 2.5,
                        "max_rate": 5.0,
                    }
                ]
            },
            evidence=[
                {
                    "source": "KB국민ONE적금 상품설명서",
                    "content": "우대금리 조건 및 가입 조건 근거",
                }
            ],
            error=None,
        ),

        "eligibility_result": make_agent_result(
            status="success",
            result={
                "eligibility_results": [
                    {
                        "product_name": "KB국민ONE적금",
                        "is_eligible": True,
                        "reason": "가입 조건 충족",
                    }
                ]
            },
            evidence=[],
            error=None,
        ),

        # 일부러 recommend_result 없음
        "agent_outputs": {},
    }

    result = validation_agent_node(state)
    print_result("TEST 2: recommendation / missing recommend_result", result)


def test_recommendation_with_llm_validation():
    """
    추천 검증 테스트
    - task_type이 recommendation이면 LLM verify_result가 실행되는지 확인
    - .env에 LLM API 설정이 있어야 정상 실행됨
    """

    state = {
        "user_query": "나한테 맞는 적금 추천해줘",
        "task_type": "recommendation",
        "plan": [
            "customer_agent",
            "financial_agent",
            "product_agent",
            "eligibility_agent",
            "recommend_agent",
            "validation_agent",
        ],
        "current_step": 5,
        "current_agent": "recommend_agent",
        "completed_agents": [
            "customer_agent",
            "financial_agent",
            "product_agent",
            "eligibility_agent",
            "recommend_agent",
        ],
        "errors": [],

        "customer_result": make_agent_result(
            status="success",
            result={
                "customer_id": 101,
                "customer_name": "고객_101",
                "monthly_saving_capacity": 200000,
                "salary_transfer_yn": True,
                "auto_transfer_yn": True,
                "card_usage_yn": False,
            },
            evidence=[],
            error=None,
        ),

        "financial_result": make_agent_result(
            status="success",
            result={
                "monthly_saving_capacity": 200000,
                "total_balance": 13000000,
                "active_accounts_count": 0,
            },
            evidence=[],
            error=None,
        ),

        "product_result": make_agent_result(
            status="success",
            result={
                "products": [
                    {
                        "product_name": "KB국민ONE적금",
                        "product_type": "적금",
                        "base_rate": 2.5,
                        "max_rate": 5.0,
                        "join_channel": "비대면",
                        "max_amount": 300000,
                    }
                ]
            },
            evidence=[
                {
                    "source": "KB국민ONE적금 상품설명서",
                    "content": "월 납입 한도, 우대금리, 가입 조건 관련 근거",
                }
            ],
            error=None,
        ),

        "eligibility_result": make_agent_result(
            status="success",
            result={
                "eligibility_results": [
                    {
                        "product_name": "KB국민ONE적금",
                        "is_eligible": True,
                        "reason": "월 납입 가능 금액과 가입 조건을 충족합니다.",
                    }
                ]
            },
            evidence=[],
            error=None,
        ),

        "recommend_result": make_agent_result(
            status="success",
            result={
                "recommendations": [
                    {
                        "product_name": "KB국민ONE적금",
                        "reason": "월 20만원 납입 여력에 적합하고 우대조건 일부 충족 가능성이 있습니다.",
                    }
                ]
            },
            evidence=[],
            error=None,
        ),
    }

    state["agent_outputs"] = {
        "customer_agent": state["customer_result"],
        "financial_agent": state["financial_result"],
        "product_agent": state["product_result"],
        "eligibility_agent": state["eligibility_result"],
        "recommend_agent": state["recommend_result"],
    }

    result = validation_agent_node(state)
    print_result("TEST 3: recommendation / LLM validation", result)


if __name__ == "__main__":
    test_rule_only_customer_lookup()
    test_missing_required_result()
    test_recommendation_with_llm_validation()