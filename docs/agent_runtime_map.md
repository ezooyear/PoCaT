# Agent Runtime Map

## Actual runtime path

The graph is built in `graph/builder.py` and imports these packages:

- `customer_agent` -> `agents.customer.customer_agent_node`
- `product_agent` -> `agents.product.product_agent_node`
- `eligibility_agent` -> `agents.eligibility.eligibility_agent_node`
- `financial_agent` -> `agents.financial.financial_agent_node`
- `recommend_agent` -> `agents.recommend.recommend_agent_node`
- `validation_agent` -> `agents.validation.validation_agent_node`
- `supervisor` -> `agents.supervisor.supervisor_node`

Those package exports are confirmed through:

- `agents/customer/__init__.py`
- `agents/product/__init__.py`
- `agents/eligibility/__init__.py`
- `agents/recommend/__init__.py`
- `agents/validation/__init__.py`
- `agents/supervisor/__init__.py`

## Legacy or unused directories

The following directories are not imported by `graph/builder.py` and currently contain only `__pycache__` artifacts in this workspace:

- `agents/eligibility_agent`
- `agents/product_agent`
- `agents/recommend_agent`
- `agents/validator_agent`

These look like legacy paths rather than active runtime code.
