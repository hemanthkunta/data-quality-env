from tasks.base import BaseTask
from env.models import AuditReport


class Task3(BaseTask):
    def get_description(self) -> str:
        return (
            "Compare 'transactions_baseline' (last month) with 'transactions_current' (this month). "
            "Detect silent data drift: mean/distribution shifts in numeric columns, new category "
            "values not present in baseline, and referential drift (new user_ids not in baseline). "
            "Nothing is explicitly labelled wrong — you must find it statistically."
        )

    def get_table_names(self) -> list[str]:
        return ["transactions_baseline", "transactions_current"]

    def grade(self, report: AuditReport, gold: dict) -> tuple[float, dict]:
        scores: dict[str, float] = {}

        amount_drift = report.drift_details.get("amount")
        if amount_drift:
            detected = "shift" in str(amount_drift.value).lower() or "mean" in str(amount_drift.value).lower()
            scores["mean_shift"] = self.brier_adjust(1.0 if detected else 0.0, amount_drift.confidence, detected)
        else:
            scores["mean_shift"] = 0.0

        new_cat_mentioned = any(
            "categor" in str(v).lower() or "crypto" in str(v).lower() or "nft" in str(v).lower()
            for v in [report.drift_details, report.recommended_fixes]
        )
        cat_drift = report.drift_details.get("category")
        if cat_drift:
            reported_cats = {x.strip() for x in str(cat_drift.value).split(",") if x.strip()}
            actual_cats = set(gold["new_categories"])
            precision = len(reported_cats & actual_cats) / max(len(reported_cats), 1)
            recall = len(reported_cats & actual_cats) / max(len(actual_cats), 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-6)
            scores["new_cats"] = self.brier_adjust(f1, cat_drift.confidence, f1 > 0.4)
        else:
            scores["new_cats"] = 0.3 if new_cat_mentioned else 0.0

        ref_drift = report.drift_details.get("user_id")
        if ref_drift:
            try:
                cleaned = str(ref_drift.value).replace("%", " ").strip()
                token = cleaned.split()[0]
                reported_pct = float(token)
                if reported_pct > 1:
                    reported_pct /= 100.0
                actual_pct = float(gold["referential_drift_pct"])
                within_5pct = abs(reported_pct - actual_pct) <= 0.05
                scores["ref_drift"] = self.brier_adjust(1.0 if within_5pct else 0.5, ref_drift.confidence, within_5pct)
            except Exception:
                scores["ref_drift"] = 0.2
        else:
            scores["ref_drift"] = 0.0

        weights = {"mean_shift": 0.40, "new_cats": 0.35, "ref_drift": 0.25}
        total = sum(scores[k] * weights[k] for k in weights)
        return round(min(1.0, total), 4), scores
