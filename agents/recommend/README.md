# Recommend Agent

Eligibility 결과를 바탕으로 실제로 추천 가능한 상품만 추려서 순위화하는 에이전트입니다.

## 역할

- `eligibility_results`를 기준으로 추천 가능한 상품만 선별합니다.
- 가입 가능 여부를 새로 판단하지 않고, Eligibility Agent의 결과를 그대로 사용합니다.

## 입력 데이터

| 항목 | 설명 |
|---|---|
| `eligibility_results` | 상품별 가입 가능 여부와 추가 확인 필요 여부가 담긴 결과 |

## 출력 데이터

| 항목 | 설명 |
|---|---|
| `recommendation_results` | 실제 추천 가능한 상품만 담은 결과 |

## 추천 규칙

| 구분 | 조건 | 처리 방식 |
|---|---|---|
| 추천 가능 상품 | `eligible=True` 이고 `check_required` 없음 | `recommendation_results`에 포함 |
| 추가 확인 필요 상품 | `eligible=True` 이지만 `check_required` 존재 | 추천 결과에는 넣지 않고 별도로 안내 |
| 가입 불가 상품 | `eligible=False` | 추천 대상에서 제외 |

## 설계 원칙

- Recommend Agent는 가입 가능 여부를 직접 판단하지 않습니다.
- 가입 가능 여부 판단은 Eligibility Agent의 책임입니다.
- Recommend Agent는 Eligibility 결과를 이용하여 순위화 및 추천만 수행합니다.

## 추천 흐름

Customer Agent  
→ Product Agent  
→ Eligibility Agent  
→ Recommend Agent

## 테스트 방법

```bash
python scripts/test_eligibility_recommend.py
python scripts/test_with_product_fixtures.py
```
