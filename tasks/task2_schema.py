from tasks.base import BaseTask
from env.models import AuditReport


class Task2(BaseTask):
    def get_description(self) -> str:
        return (
            "Audit the 'orders' table. Detect: (1) type violations (amounts stored as strings "
            "like '$12.50', dates in human-readable format), (2) range violations (negative "
            "quantity), (3) unparseable values in amount field. Report each violation type, "
            "an example value, and your confidence."
        )

    def get_table_names(self) -> list[str]:
        return ["orders"]

    def grade(self, report: AuditReport, gold: dict) -> tuple[float, dict]:
        scores: dict[str, float] = {}

        amt_detected = any(
            "amount" in str(v.get("column", "")).lower() and "type" in str(v.get("issue_type", "")).lower()
            for v in report.schema_violations
        )
        conf = next((float(v.get("confidence", 0.5)) for v in report.schema_violations if "amount" in str(v.get("column", "")).lower()), 0.5)
        scores["amount_type"] = self.brier_adjust(1.0 if amt_detected else 0.0, conf, amt_detected)

        date_detected = any("date" in str(v.get("column", "")).lower() for v in report.schema_violations)
        conf = next((float(v.get("confidence", 0.5)) for v in report.schema_violations if "date" in str(v.get("column", "")).lower()), 0.5)
        scores["date_format"] = self.brier_adjust(1.0 if date_detected else 0.0, conf, date_detected)

        neg_qty_violations = [
            v
            for v in report.schema_violations
            if "quantity" in str(v.get("column", "")).lower() and "negative" in str(v.get("issue_type", "")).lower()
        ]
        if neg_qty_violations:
            reported_count = int(neg_qty_violations[0].get("count", 0))
            acc = self.count_accuracy(reported_count, int(gold["negative_quantity_rows"]))
            conf = float(neg_qty_violations[0].get("confidence", 0.5))
            scores["neg_qty"] = self.brier_adjust(acc, conf, acc > 0.5)
        else:
            scores["neg_qty"] = 0.0

        bad_detected = any(
            "unparseable" in str(v.get("issue_type", "")).lower()
            or ("amount" in str(v.get("column", "")).lower() and "invalid" in str(v.get("issue_type", "")).lower())
            for v in report.schema_violations
        )
        scores["bad_amount"] = self.brier_adjust(0.8 if bad_detected else 0.0, 0.5, bad_detected)

        scores = {k: self.strict_score(v) for k, v in scores.items()}

        weights = {"amount_type": 0.25, "date_format": 0.25, "neg_qty": 0.25, "bad_amount": 0.25}
        total = sum(scores[k] * weights[k] for k in weights)
        return self.strict_score(round(total, 4)), scores
