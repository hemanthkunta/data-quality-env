from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SQLProbe:
    name: str
    purpose: str
    sql_template: str


TASK1_PROBES = [
    SQLProbe("sample_rows", "Quick table sanity sample", "SELECT * FROM {table} LIMIT 5"),
    SQLProbe("null_email", "Count null emails", "SELECT SUM(CASE WHEN email IS NULL THEN 1 ELSE 0 END) AS null_email FROM {table}"),
    SQLProbe("null_customer_id", "Count null customer IDs", "SELECT SUM(CASE WHEN customer_id IS NULL THEN 1 ELSE 0 END) AS null_customer_id FROM {table}"),
    SQLProbe(
        "duplicate_rows",
        "Estimate exact duplicate row count",
        "SELECT COALESCE(SUM(c-1),0) AS duplicate_rows FROM ("
        "SELECT customer_id, email, name, signup_date, country, COUNT(*) AS c "
        "FROM {table} GROUP BY 1,2,3,4,5 HAVING COUNT(*) > 1) t",
    ),
    SQLProbe("country_dist", "Distribution by country", "SELECT country, COUNT(*) AS n FROM {table} GROUP BY country ORDER BY n DESC"),
]

TASK2_PROBES = [
    SQLProbe("sample_rows", "Quick table sanity sample", "SELECT * FROM {table} LIMIT 5"),
    SQLProbe(
        "negative_quantity_rows",
        "Count negative quantity violations",
        "SELECT SUM(CASE WHEN quantity < 0 THEN 1 ELSE 0 END) AS negative_quantity_rows FROM {table}",
    ),
    SQLProbe(
        "unparseable_amount_rows",
        "Count unparseable amount values",
        "SELECT SUM(CASE WHEN try_cast(replace(amount, '$', '') AS DOUBLE) IS NULL THEN 1 ELSE 0 END) AS unparseable_amount_rows FROM {table}",
    ),
    SQLProbe(
        "amount_parse_preview",
        "Preview parsed amounts",
        "SELECT amount, try_cast(replace(amount, '$', '') AS DOUBLE) AS amount_num FROM {table} LIMIT 20",
    ),
    SQLProbe("status_dist", "Distribution by status", "SELECT status, COUNT(*) AS n FROM {table} GROUP BY status ORDER BY n DESC"),
]

TASK3_PROBES = [
    SQLProbe(
        "mean_shift",
        "Compare baseline/current amount means",
        "SELECT (SELECT AVG(amount) FROM transactions_baseline) AS baseline_mean, "
        "(SELECT AVG(amount) FROM transactions_current) AS current_mean",
    ),
    SQLProbe(
        "new_categories",
        "Find categories present only in current snapshot",
        "SELECT DISTINCT c.category FROM transactions_current c "
        "LEFT JOIN (SELECT DISTINCT category FROM transactions_baseline) b "
        "ON c.category=b.category WHERE b.category IS NULL ORDER BY c.category",
    ),
    SQLProbe(
        "new_user_row_pct",
        "Estimate referential drift on user_id",
        "SELECT AVG(CASE WHEN user_id >= 1000 THEN 1.0 ELSE 0.0 END) AS new_user_row_pct "
        "FROM transactions_current",
    ),
    SQLProbe(
        "mean_by_category",
        "Amount mean by category in current snapshot",
        "SELECT category, AVG(amount) AS avg_amount FROM transactions_current GROUP BY category ORDER BY avg_amount DESC",
    ),
]


def probes_for_task(task_id: int, table_name: str) -> list[str]:
    if task_id == 1:
        return [p.sql_template.format(table=table_name) for p in TASK1_PROBES]
    if task_id == 2:
        return [p.sql_template.format(table=table_name) for p in TASK2_PROBES]
    return [p.sql_template.format(table=table_name) for p in TASK3_PROBES]
