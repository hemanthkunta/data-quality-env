from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


SAFE_SQL_RE = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)
BLOCKED_SQL_RE = re.compile(r"\b(drop|truncate|delete|insert|update|alter|create)\b", re.IGNORECASE)


@dataclass
class PlanBundle:
    hypotheses: list[str]
    extra_queries: list[str]


def safe_query_filter(queries: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for q in queries:
        s = (q or "").strip().rstrip(";")
        if not s:
            continue
        if not SAFE_SQL_RE.match(s):
            continue
        if BLOCKED_SQL_RE.search(s):
            continue
        key = re.sub(r"\s+", " ", s.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def parse_plan_json(raw: str) -> PlanBundle:
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            return PlanBundle(hypotheses=[], extra_queries=[])
        hypotheses = payload.get("hypotheses", [])
        extra_queries = payload.get("extra_queries", [])
        return PlanBundle(
            hypotheses=[str(x) for x in hypotheses][:6],
            extra_queries=safe_query_filter([str(x) for x in extra_queries])[:3],
        )
    except Exception:
        return PlanBundle(hypotheses=[], extra_queries=[])


def build_plan_prompt(task_id: int, table_name: str, schema: dict[str, str], base_queries: list[str]) -> str:
    prompt = {
        "task_id": task_id,
        "table_name": table_name,
        "schema": schema,
        "base_queries": base_queries,
        "instruction": (
            "Propose short investigation hypotheses and at most 3 additional safe SELECT queries. "
            "Return JSON only with keys: hypotheses (list[str]) and extra_queries (list[str])."
        ),
    }
    return json.dumps(prompt)


def validate_and_repair_report(report: dict[str, Any]) -> dict[str, Any]:
    fixed = dict(report)
    fixed.setdefault("null_issues", {})
    fixed.setdefault("duplicate_row_count", 0)
    fixed.setdefault("schema_violations", [])
    fixed.setdefault("drifted_columns", [])
    fixed.setdefault("drift_details", {})
    fixed.setdefault("recommended_fixes", [])

    if not isinstance(fixed["null_issues"], dict):
        fixed["null_issues"] = {}
    if not isinstance(fixed["duplicate_row_count"], int):
        try:
            fixed["duplicate_row_count"] = int(fixed["duplicate_row_count"])
        except Exception:
            fixed["duplicate_row_count"] = 0
    if not isinstance(fixed["schema_violations"], list):
        fixed["schema_violations"] = []
    if not isinstance(fixed["drifted_columns"], list):
        fixed["drifted_columns"] = []
    if not isinstance(fixed["drift_details"], dict):
        fixed["drift_details"] = {}
    if not isinstance(fixed["recommended_fixes"], list):
        fixed["recommended_fixes"] = []

    return fixed
