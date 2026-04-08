from tasks.base import BaseTask
from env.models import AuditReport


class Task1(BaseTask):
    def get_description(self) -> str:
        return (
            "Audit the 'customers' table. Find: (1) real NULL values in each column, "
            "(2) disguised nulls stored as strings like 'NULL','N/A','-' etc., "
            "(3) exact duplicate rows, and (4) near-duplicate rows (same record, 1-2 fields changed). "
            "Report counts per finding with your confidence (0.0-1.0) in each."
        )

    def get_table_names(self) -> list[str]:
        return ["customers"]

    def grade(self, report: AuditReport, gold: dict) -> tuple[float, dict]:
        scores: dict[str, float] = {}
        if "email" in report.null_issues:
            fc = report.null_issues["email"]
            acc = self.count_accuracy(int(fc.value), int(gold["null_email_total"]))
            scores["null_email"] = self.brier_adjust(acc, fc.confidence, acc > 0.6)
        else:
            scores["null_email"] = 0.0

        if "customer_id" in report.null_issues:
            fc = report.null_issues["customer_id"]
            acc = self.count_accuracy(int(fc.value), int(gold["null_customer_id"]))
            scores["null_cid"] = self.brier_adjust(acc, fc.confidence, acc > 0.6)
        else:
            scores["null_cid"] = 0.0

        fc_dup = report.duplicate_row_count
        dup_acc = self.count_accuracy(int(fc_dup.value), int(gold["exact_duplicate_rows"]))
        scores["exact_dups"] = self.brier_adjust(dup_acc, fc_dup.confidence, dup_acc > 0.6)

        near_detected = any("near" in str(v.get("issue_type", "")).lower() for v in report.schema_violations)
        scores["near_dups"] = 0.5 if near_detected else 0.0

        weights = {"null_email": 0.30, "null_cid": 0.25, "exact_dups": 0.30, "near_dups": 0.15}
        total = sum(scores[k] * weights[k] for k in weights)
        return self.strict_score(round(total, 4)), scores
