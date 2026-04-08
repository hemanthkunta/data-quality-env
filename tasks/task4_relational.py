from tasks.base import BaseTask
from env.models import AuditReport


class Task4(BaseTask):
    def get_description(self) -> str:
        return (
            "Audit three related tables: 'customers', 'orders', 'line_items'. "
            "Find: (1) orphaned orders (customer_id in orders not in customers), "
            "(2) temporal violations (ship_date before order_date), "
            "(3) aggregate mismatches (order.order_total != SUM(line_items.price * quantity) for that order). "
            "You MUST write JOIN queries across tables. Report count per issue type with confidence."
        )

    def get_table_names(self) -> list[str]:
        return ["customers", "orders", "line_items"]

    def grade(self, report: AuditReport, gold: dict) -> tuple[float, dict]:
        scores: dict[str, float] = {}

        def find_relational_issue(issue_type_keyword: str):
            for issue in report.relational_issues:
                if issue_type_keyword.lower() in str(issue.get("issue_type", "")).lower():
                    return issue
            return None

        orphan_issue = find_relational_issue("orphan") or find_relational_issue("foreign_key")
        if orphan_issue:
            acc = self.count_accuracy(int(orphan_issue.get("count", 0)), int(gold["orphaned_order_count"]))
            conf = float(orphan_issue.get("confidence", 0.5))
            scores["orphans"] = self.brier_adjust(acc, conf, acc > 0.5)
        else:
            scores["orphans"] = 0.0

        temp_issue = find_relational_issue("temporal") or find_relational_issue("ship_date")
        if temp_issue:
            acc = self.count_accuracy(int(temp_issue.get("count", 0)), int(gold["temporal_violation_count"]))
            conf = float(temp_issue.get("confidence", 0.5))
            scores["temporal"] = self.brier_adjust(acc, conf, acc > 0.5)
        else:
            scores["temporal"] = 0.0

        agg_issue = find_relational_issue("aggregate") or find_relational_issue("mismatch") or find_relational_issue("total")
        if agg_issue:
            acc = self.count_accuracy(int(agg_issue.get("count", 0)), int(gold["aggregate_mismatch_count"]))
            conf = float(agg_issue.get("confidence", 0.5))
            scores["aggregates"] = self.brier_adjust(acc, conf, acc > 0.5)
        else:
            scores["aggregates"] = 0.0

        scores = {k: self.strict_score(v) for k, v in scores.items()}

        weights = {"orphans": 0.40, "temporal": 0.35, "aggregates": 0.25}
        total = sum(scores[k] * weights[k] for k in weights)
        return self.strict_score(round(total, 4)), scores
