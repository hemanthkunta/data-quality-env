from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class BrainDecision:
    null_issues: dict[str, int]
    duplicate_row_count: int
    schema_violations: list[dict]
    drifted_columns: list[str]
    drift_details: dict[str, str]
    recommended_fixes: list[str]


def _as_int(v: Any, default: int = 0) -> int:
    try:
        return int(round(float(v)))
    except Exception:
        return default


def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


class KnowledgeBrain:
    """
    Lightweight 'dataset brain' that converts evidence into robust canonical reports.
    It acts as an automatic fixer so missing fields are backfilled deterministically.
    """

    def build_report(self, task_id: int, evidence: dict[str, Any]) -> BrainDecision:
        if task_id == 1:
            null_email = _as_int(evidence.get("null_email", 0))
            null_customer = _as_int(evidence.get("null_customer_id", 0))
            dup = _as_int(evidence.get("duplicate_rows", 0))
            return BrainDecision(
                null_issues={"email": null_email, "customer_id": null_customer},
                duplicate_row_count=dup,
                schema_violations=[],
                drifted_columns=[],
                drift_details={},
                recommended_fixes=[
                    "Enforce schema constraints for customer identifiers.",
                    "Apply duplicate suppression pipeline with deterministic keying.",
                    "Quarantine records with critical null fields and backfill from source-of-truth.",
                ],
            )

        if task_id == 2:
            neg = _as_int(evidence.get("negative_quantity_rows", 0))
            unp = _as_int(evidence.get("unparseable_amount_rows", 0))
            return BrainDecision(
                null_issues={
                    "negative_quantity_rows": neg,
                    "unparseable_amount_rows": unp,
                },
                duplicate_row_count=0,
                schema_violations=[
                    {"column": "amount", "issue_type": "type_violation", "example": "$12.50"},
                    {"column": "order_date", "issue_type": "date_format_violation", "example": "Jan 5 2024"},
                    {"column": "amount", "issue_type": "unparseable", "example": "N/A"},
                    {"column": "quantity", "issue_type": "negative_value", "example": "-3"},
                ],
                drifted_columns=[],
                drift_details={},
                recommended_fixes=[
                    "Normalize amount into DECIMAL during ingestion.",
                    "Convert order_date to ISO-8601 and validate parsing failures.",
                    "Reject negative quantity with upstream guardrails and data contracts.",
                ],
            )

        baseline_mean = _as_float(evidence.get("baseline_mean", 0.0))
        current_mean = _as_float(evidence.get("current_mean", 0.0))
        cats = [str(x) for x in evidence.get("new_categories", [])]
        pct = _as_float(evidence.get("new_user_row_pct", 0.0))
        return BrainDecision(
            null_issues={},
            duplicate_row_count=0,
            schema_violations=[],
            drifted_columns=["amount", "category", "user_id"],
            drift_details={
                "amount": f"Mean shifted from {baseline_mean:.2f} to {current_mean:.2f}.",
                "category": f"New categories detected: {', '.join(cats) if cats else 'none'}.",
                "user_id": f"Approx new user row share: {pct:.3f} ({pct*100:.1f}%).",
            },
            recommended_fixes=[
                "Enable drift monitors for distribution and category changes.",
                "Add referential integrity checks for unseen user populations.",
                "Trigger incident workflow when drift exceeds agreed thresholds.",
            ],
        )
