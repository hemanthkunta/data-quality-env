from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class FindingConfidence(BaseModel):
    """A single audit finding with agent-reported confidence."""

    value: Any
    confidence: float = Field(ge=0.0, le=1.0)


class AuditReport(BaseModel):
    """Structured audit report submitted by the agent."""

    null_issues: dict[str, FindingConfidence]
    duplicate_row_count: FindingConfidence
    schema_violations: list[dict[str, Any]]
    drifted_columns: list[str]
    drift_details: dict[str, FindingConfidence]
    relational_issues: list[dict[str, Any]]
    recommended_fixes: list[str]


class Action(BaseModel):
    action_type: Literal["query", "submit_report", "fix_sql"]
    sql: str | None = None
    report: AuditReport | None = None


class Observation(BaseModel):
    task_id: int
    task_description: str
    tables: dict[str, dict[str, str]]
    row_counts: dict[str, int]
    step: int
    max_steps: int
    query_credits_remaining: int
    phase: Literal["audit", "fix"]
    last_query_result: list[dict] | None
    last_action_error: str | None
    last_fix_score: float | None


class RewardBreakdown(BaseModel):
    base_audit_score: float
    confidence_brier_adjustment: float
    budget_efficiency_bonus: float
    fix_verification_bonus: float
    total: float


class Reward(BaseModel):
    value: float = Field(ge=-0.5, le=1.25)
    breakdown: RewardBreakdown
    done: bool
    info: dict[str, Any]


class EpisodeState(BaseModel):
    task_id: int
    seed: int
    step: int = 0
    max_steps: int = 12
    query_credits: int = 10
    phase: Literal["audit", "fix"] = "audit"
    fix_steps_remaining: int = 3
    report_submitted: bool = False
    done: bool = False
    gold_faults: dict[str, Any] = {}
    audit_score: float = 0.0
    fix_bonus: float = 0.0
