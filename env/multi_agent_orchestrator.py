from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from env.agent_memory import MemoryStore
from env.knowledge_brain import KnowledgeBrain
from env.reasoning_stack import build_plan_prompt, parse_plan_json, safe_query_filter, validate_and_repair_report

API_BASE_URL = os.environ.get("API_BASE_URL", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "")
API_KEY = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY", "")


def _get_client() -> OpenAI | None:
    if not API_BASE_URL or not MODEL_NAME or not API_KEY:
        return None
    try:
        return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception:
        return None


@dataclass
class OrchestratorPlan:
    assistant_message: str
    action: dict[str, Any]
    hypotheses: list[str]
    selected_queries: list[str]


class MultiAgentOrchestrator:
    """
    Planner -> Critic -> Executor -> Fixer stack.

    Designed to feel closer to a modern assistant product while still only
    using safe OpenEnv actions.
    """

    def __init__(self, memory: MemoryStore | None = None) -> None:
        self.client = _get_client()
        self.memory = memory
        self.brain = KnowledgeBrain()

    def _llm_json(self, system: str, user: dict[str, Any], max_tokens: int = 600) -> dict[str, Any]:
        if self.client is None:
            return {}
        try:
            c = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user)},
                ],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            raw = (c.choices[0].message.content or "").strip()
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    def plan_queries(
        self,
        task_id: int,
        obs: dict[str, Any],
        base_queries: list[str],
        reasoning_hints: list[str] | None = None,
    ) -> tuple[list[str], list[str]]:
        reasoning_hints = reasoning_hints or []
        user = {
            "task_id": task_id,
            "table_name": obs.get("table_name"),
            "schema": obs.get("schema", {}),
            "base_queries": base_queries,
            "reasoning_hints": reasoning_hints,
            "instruction": "Return JSON with hypotheses and extra_queries only.",
        }
        system = (
            "You are a planning module for SQL auditing. Return JSON only with keys hypotheses and extra_queries. "
            "extra_queries must be safe SELECT/WITH only and bounded to at most 3."
        )
        parsed = self._llm_json(system, user, max_tokens=350)
        plan = parse_plan_json(json.dumps(parsed)) if parsed else parse_plan_json("{}")
        extra_queries = safe_query_filter(plan.extra_queries)[:3]
        hypotheses = plan.hypotheses[:6]
        return hypotheses, extra_queries

    def critique_report(self, task_id: int, report: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
        report = validate_and_repair_report(report)
        # deterministic brain first
        brain_report = self.brain.build_report(task_id, evidence)
        merged = {
            "null_issues": dict(brain_report.null_issues),
            "duplicate_row_count": brain_report.duplicate_row_count,
            "schema_violations": list(brain_report.schema_violations),
            "drifted_columns": list(brain_report.drifted_columns),
            "drift_details": dict(brain_report.drift_details),
            "recommended_fixes": list(brain_report.recommended_fixes),
        }
        # preserve user/LLM-added details where safe
        merged["null_issues"].update(report.get("null_issues", {}))
        if int(report.get("duplicate_row_count", 0)) > merged["duplicate_row_count"]:
            merged["duplicate_row_count"] = int(report["duplicate_row_count"])
        merged["schema_violations"].extend(report.get("schema_violations", []))
        for c in report.get("drifted_columns", []):
            if c not in merged["drifted_columns"]:
                merged["drifted_columns"].append(c)
        merged["drift_details"].update(report.get("drift_details", {}))
        for fix in report.get("recommended_fixes", []):
            if fix not in merged["recommended_fixes"]:
                merged["recommended_fixes"].append(fix)
        return validate_and_repair_report(merged)

    def build_chat_response(
        self,
        user_text: str,
        obs: dict[str, Any],
        task_id: int,
        base_queries: list[str],
        reasoning_hints: list[str] | None = None,
    ) -> OrchestratorPlan:
        hypotheses, extra_queries = self.plan_queries(task_id, obs, base_queries, reasoning_hints)
        selected_queries = base_queries + extra_queries
        assistant_message = self._assistant_message(user_text, hypotheses, selected_queries, obs)

        action: dict[str, Any]
        lower = user_text.lower().strip()
        if any(word in lower for word in ["final", "submit", "report", "done", "finish"]):
            action = {"action_type": "submit_report", "report": self._fallback_report(task_id)}
        else:
            action = {"action_type": "query", "sql": selected_queries[0] if selected_queries else f"SELECT COUNT(*) AS n FROM {obs['table_name']}"}

        return OrchestratorPlan(
            assistant_message=assistant_message,
            action=action,
            hypotheses=hypotheses,
            selected_queries=selected_queries,
        )

    def _assistant_message(self, user_text: str, hypotheses: list[str], queries: list[str], obs: dict[str, Any]) -> str:
        if hypotheses:
            lead = hypotheses[0]
        else:
            lead = "I will inspect the data with a targeted SQL probe."
        if queries:
            return f"{lead} Next I’ll run a focused query and keep the plan safe and deterministic."
        return "I’ll use the available evidence to produce the final audit report."

    def _fallback_report(self, task_id: int) -> dict[str, Any]:
        if task_id == 1:
            return {
                "null_issues": {},
                "duplicate_row_count": 0,
                "schema_violations": [],
                "drifted_columns": [],
                "drift_details": {},
                "recommended_fixes": [],
            }
        if task_id == 2:
            return {
                "null_issues": {},
                "duplicate_row_count": 0,
                "schema_violations": [],
                "drifted_columns": [],
                "drift_details": {},
                "recommended_fixes": [],
            }
        return {
            "null_issues": {},
            "duplicate_row_count": 0,
            "schema_violations": [],
            "drifted_columns": [],
            "drift_details": {},
            "recommended_fixes": [],
        }
