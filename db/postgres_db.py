"""
PostgreSQL 데이터베이스 연결 모듈
- 연결 관리 및 SQL 실행
- SELECT 쿼리만 허용하는 안전 장치 포함
"""
import os
import re
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

# ─── DB 접속 설정 ───
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "database": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "ghfj526127@"),
}

# ─── 위험한 SQL 키워드 ───
DANGEROUS_KEYWORDS = [
    "DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE",
    "CREATE", "GRANT", "REVOKE", "EXEC", "EXECUTE",
]

# ─── DB 스키마 정보 (LLM이 SQL 생성 시 참고) ───
DB_SCHEMA = """
## PostgreSQL 데이터베이스 스키마

### 1. customers (고객 정보)
| 컬럼 | 타입 | 설명 |
|---|---|---|
| customer_id | INT (PK) | 고객 고유 ID |
| customer_name | VARCHAR(50) | 고객명 (예: 고객_001) |
| birth_date | DATE | 생년월일 |
| customer_job | VARCHAR(50) | 직업 (회사원, 공무원, 자영업 등) |
| created_at | DATE | 고객 등록일 |
| annual_income | INT | 연소득 (만원 단위) |
| income_level | VARCHAR(20) | 소득 수준 (낮음/중간/높음) |
| main_bank_yn | BOOLEAN | 주거래 은행 여부 |
| salary_transfer_yn | BOOLEAN | 급여 이체 여부 |
| auto_transfer_yn | BOOLEAN | 자동 이체 여부 |
| card_usage_yn | BOOLEAN | 카드 사용 여부 |
| marketing_agree_yn | BOOLEAN | 마케팅 수신 동의 |
| transaction_months | INT | 거래 개월 수 |
| available_monthly_saving | INT | 월 가용 저축액 |
| updated_at | DATE | 정보 갱신일 |

### 3. customer_accounts (고객 계좌)
| 컬럼 | 타입 | 설명 |
|---|---|---|
| account_id | INT (PK) | 계좌 고유 ID |
| customer_id | INT (FK → customers) | 고객 ID |
| product_id | INT (FK → products) | 상품 ID |
| account_number | VARCHAR(30) | 계좌번호 |
| join_date | DATE | 가입일 |
| maturity_date | DATE | 만기일 |
| contract_months | INT | 계약 기간(개월) |
| deposit_amount | BIGINT | 예치 금액 |
| monthly_amount | BIGINT | 월 납입 금액 |
| current_balance | BIGINT | 현재 잔액 |
| account_status | VARCHAR(20) | 계좌 상태 (ACTIVE/MATURED) |
| applied_rate | NUMERIC(5,2) | 적용 금리(%) |
| created_at | DATE | 계좌 생성일 |

### 4. payment_history (납입 이력)
| 컬럼 | 타입 | 설명 |
|---|---|---|
| payment_id | INT (PK) | 납입 고유 ID |
| account_id | INT (FK → customer_accounts) | 계좌 ID |
| payment_date | DATE | 납입일 |
| payment_amount | BIGINT | 납입 금액 |
| payment_round | INT | 납입 회차 |

### 테이블 관계
- customers(1) → customer_accounts(N): 고객은 여러 계좌를 가질 수 있음
- products(1) → customer_accounts(N): 하나의 상품에 여러 계좌가 연결됨
- customer_accounts(1) → payment_history(N): 하나의 계좌에 여러 납입 이력이 있음
"""


def get_connection():
    """PostgreSQL 연결을 생성합니다."""
    return psycopg2.connect(**DB_CONFIG)


def validate_sql(sql: str) -> tuple[bool, str]:
    """
    SQL 쿼리의 안전성을 검증합니다.
    SELECT 쿼리만 허용하고, 위험한 키워드를 차단합니다.
    
    Returns:
        (is_safe, message): 안전 여부와 메시지
    """
    # 주석과 공백 제거 후 첫 키워드 확인
    cleaned = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)  # 라인 주석 제거
    cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)  # 블록 주석 제거
    cleaned = cleaned.strip().upper()

    if not cleaned.startswith("SELECT"):
        return False, "SELECT 쿼리만 실행할 수 있습니다."

    # 위험한 키워드 확인 (서브쿼리 내에서도 차단)
    for keyword in DANGEROUS_KEYWORDS:
        # 단어 경계로 정확한 매칭
        if re.search(rf'\b{keyword}\b', cleaned):
            return False, f"보안상 '{keyword}' 키워드가 포함된 쿼리는 실행할 수 없습니다."

    # 세미콜론 뒤에 추가 쿼리가 있는지 확인 (SQL Injection 방지)
    statements = [s.strip() for s in cleaned.split(";") if s.strip()]
    if len(statements) > 1:
        return False, "한 번에 하나의 쿼리만 실행할 수 있습니다."

    return True, "OK"


def execute_query(sql: str, max_rows: int = 50) -> str:
    """
    SQL 쿼리를 실행하고 결과를 포맷된 텍스트로 반환합니다.
    
    Args:
        sql: 실행할 SQL 쿼리 (SELECT만 허용)
        max_rows: 반환할 최대 행 수
        
    Returns:
        str: 포맷된 쿼리 결과
    """
    # 1. SQL 검증
    is_safe, message = validate_sql(sql)
    if not is_safe:
        return f"⚠️ 쿼리 차단: {message}"

    # 2. LIMIT 강제 추가 (없으면 추가)
    sql_upper = sql.strip().upper()
    if "LIMIT" not in sql_upper:
        sql = sql.strip().rstrip(";").strip() + f" LIMIT {max_rows};"

    # 3. 실행
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(sql)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        cursor.close()
        conn.close()

        if not rows:
            return "조회 결과가 없습니다."

        # 4. 결과를 표 형식으로 포맷
        lines = []
        lines.append(f"조회 결과: {len(rows)}건")
        lines.append("")

        # 헤더
        header = " | ".join(columns)
        lines.append(header)
        lines.append("-" * len(header))

        # 데이터
        for row in rows:
            values = []
            for col in columns:
                val = row[col]
                if val is None:
                    values.append("NULL")
                elif isinstance(val, bool):
                    values.append("예" if val else "아니오")
                else:
                    values.append(str(val))
            lines.append(" | ".join(values))

        return "\n".join(lines)

    except psycopg2.Error as e:
        return f"DB 오류: {e.pgerror or str(e)}"
    except Exception as e:
        return f"실행 오류: {str(e)}"


def test_connection() -> bool:
    """DB 연결 테스트. 연결 가능하면 True를 반환합니다."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1;")
        cursor.close()
        conn.close()
        return True
    except Exception:
        return False
