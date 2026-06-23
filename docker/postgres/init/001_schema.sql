CREATE TABLE IF NOT EXISTS customers (
    customer_id integer PRIMARY KEY,
    customer_name varchar(50) NOT NULL,
    birth_date date,
    customer_job varchar(50),
    created_at date DEFAULT CURRENT_DATE,
    annual_income integer,
    income_level varchar(20),
    main_bank_yn boolean DEFAULT false,
    salary_transfer_yn boolean DEFAULT false,
    auto_transfer_yn boolean DEFAULT false,
    card_usage_yn boolean DEFAULT false,
    marketing_agree_yn boolean DEFAULT false,
    transaction_months integer DEFAULT 0,
    available_monthly_saving integer DEFAULT 0,
    updated_at date DEFAULT CURRENT_DATE
);

CREATE TABLE IF NOT EXISTS products (
    product_id integer PRIMARY KEY,
    product_name varchar(200) NOT NULL,
    product_type varchar(50),
    min_amount bigint,
    max_amount bigint,
    min_period_months integer,
    max_period_months integer,
    base_rate numeric(5,2),
    max_rate numeric(5,2),
    age_min integer,
    age_max integer,
    is_active boolean DEFAULT true,
    rag_document_key varchar(255),
    created_at date DEFAULT CURRENT_DATE,
    updated_at date DEFAULT CURRENT_DATE
);

CREATE TABLE IF NOT EXISTS customer_accounts (
    account_id integer PRIMARY KEY,
    customer_id integer NOT NULL REFERENCES customers(customer_id),
    product_id integer REFERENCES products(product_id),
    account_number varchar(30) NOT NULL,
    join_date date,
    maturity_date date,
    contract_months integer,
    deposit_amount bigint,
    monthly_amount bigint,
    current_balance bigint,
    account_status varchar(20),
    applied_rate numeric(5,2),
    created_at date DEFAULT CURRENT_DATE
);

CREATE TABLE IF NOT EXISTS payment_history (
    payment_id integer PRIMARY KEY,
    account_id integer NOT NULL REFERENCES customer_accounts(account_id),
    payment_date date,
    payment_amount bigint,
    payment_round integer
);

INSERT INTO customers (
    customer_id,
    customer_name,
    birth_date,
    customer_job,
    annual_income,
    income_level,
    main_bank_yn,
    salary_transfer_yn,
    auto_transfer_yn,
    card_usage_yn,
    marketing_agree_yn,
    transaction_months,
    available_monthly_saving
) VALUES
    (1, 'customer_001', '1994-05-12', 'office_worker', 4800, 'middle', true, true, true, true, false, 36, 500000)
ON CONFLICT (customer_id) DO NOTHING;

INSERT INTO products (
    product_id,
    product_name,
    product_type,
    min_amount,
    max_amount,
    min_period_months,
    max_period_months,
    base_rate,
    max_rate,
    age_min,
    age_max,
    is_active,
    rag_document_key
) VALUES
    (1, 'KB Star Savings', 'savings', 10000, 1000000, 6, 36, 2.50, 4.20, 18, 65, true, 'kb_star_savings'),
    (2, 'KB Youth Dream Savings', 'savings', 10000, 500000, 12, 36, 3.00, 5.00, 19, 34, true, 'kb_youth_dream_savings'),
    (3, 'KB Flexible Deposit', 'deposit', 100000, 10000000, 3, 24, 2.20, 3.60, 18, 80, true, 'kb_flexible_deposit')
ON CONFLICT (product_id) DO NOTHING;

INSERT INTO customer_accounts (
    account_id,
    customer_id,
    product_id,
    account_number,
    join_date,
    maturity_date,
    contract_months,
    deposit_amount,
    monthly_amount,
    current_balance,
    account_status,
    applied_rate
) VALUES
    (1, 1, 1, '110-123-000001', '2025-01-15', '2027-01-15', 24, 0, 300000, 5400000, 'ACTIVE', 4.10)
ON CONFLICT (account_id) DO NOTHING;

INSERT INTO payment_history (
    payment_id,
    account_id,
    payment_date,
    payment_amount,
    payment_round
) VALUES
    (1, 1, '2025-01-15', 300000, 1),
    (2, 1, '2025-02-15', 300000, 2),
    (3, 1, '2025-03-15', 300000, 3)
ON CONFLICT (payment_id) DO NOTHING;
