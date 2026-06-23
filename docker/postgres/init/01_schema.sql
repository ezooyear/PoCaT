CREATE TABLE IF NOT EXISTS customers (
    customer_id INTEGER PRIMARY KEY,
    customer_name VARCHAR(50) NOT NULL UNIQUE,
    birth_date DATE,
    customer_job VARCHAR(50),
    created_at DATE DEFAULT CURRENT_DATE,
    annual_income INTEGER,
    income_level VARCHAR(20),
    main_bank_yn BOOLEAN DEFAULT FALSE,
    salary_transfer_yn BOOLEAN DEFAULT FALSE,
    auto_transfer_yn BOOLEAN DEFAULT FALSE,
    card_usage_yn BOOLEAN DEFAULT FALSE,
    marketing_agree_yn BOOLEAN DEFAULT FALSE,
    transaction_months INTEGER DEFAULT 0,
    available_monthly_saving INTEGER,
    updated_at DATE DEFAULT CURRENT_DATE
);

CREATE TABLE IF NOT EXISTS products (
    product_id INTEGER PRIMARY KEY,
    product_name VARCHAR(100) NOT NULL,
    product_type VARCHAR(50),
    base_rate NUMERIC(5,2),
    max_rate NUMERIC(5,2),
    created_at DATE DEFAULT CURRENT_DATE
);

CREATE TABLE IF NOT EXISTS customer_accounts (
    account_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    product_id INTEGER REFERENCES products(product_id),
    account_number VARCHAR(30) NOT NULL UNIQUE,
    join_date DATE,
    maturity_date DATE,
    contract_months INTEGER,
    deposit_amount BIGINT,
    monthly_amount BIGINT,
    current_balance BIGINT,
    account_status VARCHAR(20),
    applied_rate NUMERIC(5,2),
    created_at DATE DEFAULT CURRENT_DATE
);

CREATE TABLE IF NOT EXISTS payment_history (
    payment_id INTEGER PRIMARY KEY,
    account_id INTEGER NOT NULL REFERENCES customer_accounts(account_id),
    payment_date DATE,
    payment_amount BIGINT,
    payment_round INTEGER
);

CREATE INDEX IF NOT EXISTS idx_customers_name ON customers(customer_name);
CREATE INDEX IF NOT EXISTS idx_customer_accounts_customer_id ON customer_accounts(customer_id);
CREATE INDEX IF NOT EXISTS idx_payment_history_account_id ON payment_history(account_id);
