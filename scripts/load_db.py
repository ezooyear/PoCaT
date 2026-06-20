"""
PoCaT DB 적재 스크립트 (data/*.csv → PostgreSQL)

- data 폴더의 테이블 덤프 CSV 4종을 pocat 스키마(기본)에 적재한다.
- 앱의 SELECT 전용 가드(validate_sql)를 우회해 DDL/적재가 가능하도록 psycopg2로 직접 실행한다.
- ⚠️ search_path 함정 처리: 코드가 비한정 테이블명(FROM customers)으로 쿼리하므로,
  적재 후 ALTER DATABASE ... SET search_path TO <schema>, public 으로 영구 반영한다.

사용:
    python scripts/load_db.py
    # 스키마 이름 변경:  DB_SCHEMA_NAME=public python scripts/load_db.py

접속 정보는 .env(DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD)를 사용한다.
docker(다른 폴더)로 띄운 DB라면 .env의 값이 그 컨테이너의 5432와 일치해야 한다.
"""

import glob
import os
import sys
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

load_dotenv(PROJECT_ROOT / ".env")

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "database": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}
SCHEMA = os.getenv("DB_SCHEMA_NAME", "pocat3")

# ─── 테이블 정의 (CSV 헤더 순서와 동일하게 컬럼 나열) ───
# 적재 순서 = FK 안전 순서. 삭제는 역순.
TABLES = {
    "customers": {
        "pattern": "customers_*.csv",
        "ddl": """
            customer_id              INT PRIMARY KEY,
            customer_name            VARCHAR(50),
            birth_date               DATE,
            customer_job             VARCHAR(50),
            created_at               DATE,
            annual_income            INT,
            income_level             VARCHAR(20),
            main_bank_yn             BOOLEAN,
            salary_transfer_yn       BOOLEAN,
            auto_transfer_yn         BOOLEAN,
            card_usage_yn            BOOLEAN,
            marketing_agree_yn       BOOLEAN,
            transaction_months       INT,
            available_monthly_saving BIGINT,
            updated_at               DATE
        """,
        "columns": [
            "customer_id", "customer_name", "birth_date", "customer_job", "created_at",
            "annual_income", "income_level", "main_bank_yn", "salary_transfer_yn",
            "auto_transfer_yn", "card_usage_yn", "marketing_agree_yn",
            "transaction_months", "available_monthly_saving", "updated_at",
        ],
    },
    "products": {
        "pattern": "products_*.csv",
        "ddl": """
            product_id            INT PRIMARY KEY,
            product_name          VARCHAR(100),
            product_type          VARCHAR(20),
            join_channel          VARCHAR(100),
            min_amount            BIGINT,
            max_amount            BIGINT,
            min_period_months     INT,
            max_period_months     INT,
            partial_withdrawal_yn BOOLEAN,
            auto_redeposit_yn     BOOLEAN,
            additional_deposit_yn BOOLEAN,
            is_active             BOOLEAN,
            base_rate             NUMERIC(5,2),
            max_rate              NUMERIC(5,2),
            age_min               INT,
            age_max               INT,
            rag_document_key      VARCHAR(100)
        """,
        "columns": [
            "product_id", "product_name", "product_type", "join_channel",
            "min_amount", "max_amount", "min_period_months", "max_period_months",
            "partial_withdrawal_yn", "auto_redeposit_yn", "additional_deposit_yn",
            "is_active", "base_rate", "max_rate", "age_min", "age_max", "rag_document_key",
        ],
    },
    "customer_accounts": {
        "pattern": "customer_accounts_*.csv",
        "ddl": """
            account_id      INT PRIMARY KEY,
            customer_id     INT REFERENCES {schema}.customers(customer_id),
            product_id      INT REFERENCES {schema}.products(product_id),
            account_number  VARCHAR(30),
            join_date       DATE,
            maturity_date   DATE,
            contract_months INT,
            deposit_amount  BIGINT,
            monthly_amount  BIGINT,
            current_balance BIGINT,
            account_status  VARCHAR(20),
            applied_rate    NUMERIC(5,2),
            created_at      DATE
        """,
        "columns": [
            "account_id", "customer_id", "product_id", "account_number", "join_date",
            "maturity_date", "contract_months", "deposit_amount", "monthly_amount",
            "current_balance", "account_status", "applied_rate", "created_at",
        ],
    },
    "payment_history": {
        "pattern": "payment_history_*.csv",
        "ddl": """
            payment_id     INT PRIMARY KEY,
            account_id     INT REFERENCES {schema}.customer_accounts(account_id),
            payment_date   DATE,
            payment_amount BIGINT,
            payment_round  INT
        """,
        "columns": [
            "payment_id", "account_id", "payment_date", "payment_amount", "payment_round",
        ],
    },
}

LOAD_ORDER = ["customers", "products", "customer_accounts", "payment_history"]


def _latest_csv(pattern: str) -> str:
    matches = sorted(glob.glob(str(DATA_DIR / pattern)))
    if not matches:
        raise FileNotFoundError(f"CSV를 찾을 수 없습니다: data/{pattern}")
    return matches[-1]  # 타임스탬프 최신


def main() -> None:
    if not DB_CONFIG["password"]:
        print("⚠️  DB_PASSWORD가 비어 있습니다. .env를 확인하세요.")

    print(f"🔌 접속 시도: {DB_CONFIG['user']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}  schema={SCHEMA}")
    try:
        conn = psycopg2.connect(**DB_CONFIG)
    except Exception as e:
        print(f"❌ DB 연결 실패: {e}")
        print("   → docker DB가 떠 있는지, .env의 DB_HOST/PORT/PASSWORD가 맞는지 확인하세요.")
        sys.exit(1)

    conn.autocommit = False
    cur = conn.cursor()

    try:
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA};")

        # 기존 테이블 정리 (역순, FK 때문에 CASCADE)
        for name in reversed(LOAD_ORDER):
            cur.execute(f"DROP TABLE IF EXISTS {SCHEMA}.{name} CASCADE;")

        # 생성 + 적재 (정순)
        for name in LOAD_ORDER:
            spec = TABLES[name]
            ddl = spec["ddl"].format(schema=SCHEMA)
            cur.execute(f"CREATE TABLE {SCHEMA}.{name} ({ddl});")

            csv_path = _latest_csv(spec["pattern"])
            cols = ", ".join(spec["columns"])
            copy_sql = (
                f"COPY {SCHEMA}.{name} ({cols}) FROM STDIN "
                f"WITH (FORMAT csv, HEADER true, NULL '')"
            )
            with open(csv_path, "r", encoding="utf-8") as f:
                cur.copy_expert(copy_sql, f)

            cur.execute(f"SELECT COUNT(*) FROM {SCHEMA}.{name};")
            count = cur.fetchone()[0]
            print(f"   ✅ {name:<18} ← {os.path.basename(csv_path)}  ({count}행)")

        # ⚠️ search_path 영구 반영 — 코드가 비한정 테이블명으로 쿼리하므로 필수
        cur.execute(f'ALTER DATABASE "{DB_CONFIG["database"]}" SET search_path TO {SCHEMA}, public;')
        print(f"🛣️  search_path 설정: ALTER DATABASE {DB_CONFIG['database']} SET search_path TO {SCHEMA}, public")

        conn.commit()
        print("✅ 적재 완료. (새 연결부터 search_path 적용)")

        # 검증: search_path가 적용된 새 연결로 비한정 쿼리가 되는지 확인
        conn.close()
        conn2 = psycopg2.connect(**DB_CONFIG)
        cur2 = conn2.cursor()
        cur2.execute("SELECT customer_name FROM customers ORDER BY customer_id LIMIT 1;")
        sample = cur2.fetchone()
        print(f"🔎 검증(비한정 쿼리 FROM customers): {sample[0] if sample else '없음'}")
        cur2.close()
        conn2.close()

    except Exception as e:
        conn.rollback()
        print(f"❌ 적재 실패(롤백): {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
