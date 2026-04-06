# SQL Agent Mind Guide

This document is a practical SQL reference used by the agent to reason deeply about data quality tasks.

## Core SQL command pattern
- Allowed: `SELECT`, `WITH` (CTEs)
- Blocked: destructive statements (`DROP`, `DELETE`, `UPDATE`, etc.)

## Most important SQL functions in this environment

### Aggregation
- `COUNT(*)`
- `SUM(...)`
- `AVG(...)`
- `MIN(...)`, `MAX(...)`

### Data quality checks
- `CASE WHEN ... THEN ... ELSE ... END`
- `IS NULL`
- `TRY_CAST(...)`
- `REPLACE(...)`

### Deduplication logic
- `GROUP BY ... HAVING COUNT(*) > 1`
- `SUM(c - 1)` where `c` is duplicate group count

### Drift analysis
- Baseline vs current mean comparison with subqueries
- `LEFT JOIN ... WHERE right_col IS NULL` for novelty/referential drift
- Distribution checks with `GROUP BY`

## Task-specific deep probe examples

### Task 1: Nulls + duplicates
```sql
SELECT SUM(CASE WHEN email IS NULL THEN 1 ELSE 0 END) AS null_email,
       SUM(CASE WHEN customer_id IS NULL THEN 1 ELSE 0 END) AS null_customer_id
FROM customers;
```

```sql
SELECT COALESCE(SUM(c - 1), 0) AS duplicate_rows
FROM (
  SELECT customer_id, email, name, signup_date, country, COUNT(*) AS c
  FROM customers
  GROUP BY 1,2,3,4,5
  HAVING COUNT(*) > 1
) t;
```

### Task 2: Schema and range violations
```sql
SELECT SUM(CASE WHEN quantity < 0 THEN 1 ELSE 0 END) AS negative_quantity_rows
FROM orders;
```

```sql
SELECT SUM(CASE WHEN try_cast(replace(amount, '$', '') AS DOUBLE) IS NULL THEN 1 ELSE 0 END) AS unparseable_amount_rows
FROM orders;
```

### Task 3: Silent drift
```sql
SELECT
  (SELECT AVG(amount) FROM transactions_baseline) AS baseline_mean,
  (SELECT AVG(amount) FROM transactions_current) AS current_mean;
```

```sql
SELECT DISTINCT c.category
FROM transactions_current c
LEFT JOIN (SELECT DISTINCT category FROM transactions_baseline) b
  ON c.category = b.category
WHERE b.category IS NULL
ORDER BY c.category;
```

```sql
SELECT AVG(CASE WHEN user_id >= 1000 THEN 1.0 ELSE 0.0 END) AS new_user_row_pct
FROM transactions_current;
```

## Deeper testing strategy
1. Run sample + aggregate checks first.
2. Validate each scoring dimension with one explicit probe.
3. Add distribution probes to avoid blind spots.
4. Submit report only after all dimensions are covered.
