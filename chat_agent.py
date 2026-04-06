"""
Chat-style AI auditor for DataQualityEnv.

This wrapper now behaves like a modern assistant stack:
- planner produces hypotheses and safe probe ideas
- executor runs OpenEnv tool calls
- critic normalizes/repairs the final report
- memory influences future turns
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import requests
from openai import OpenAI

from env.agent_memory import MemoryStore
from env.multi_agent_orchestrator import MultiAgentOrchestrator

API_BASE_URL = os.environ.get("API_BASE_URL", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "")
API_KEY = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY", "")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")
MEMORY_PATH = os.environ.get("AGENT_MEMORY_PATH", "outputs/agent_memory.json")


SYSTEM_PROMPT = """You are a data quality auditing assistant.
You can investigate data via SQL and then submit a final JSON report.

Return valid JSON only in this schema:
{
  "assistant_message": "short natural language reply",
  "action": {
    "action_type": "query" | "submit_report",
    "sql": "... optional when query ...",
    "report": {
      "null_issues": {"col": 0},
      "duplicate_row_count": 0,
      "schema_violations": [],
      "drifted_columns": [],
      "drift_details": {},
      "recommended_fixes": []
    }
  }
}

Rules:
- If user asks to inspect, use action_type=query with safe SELECT/WITH SQL.
- If enough evidence exists or user asks to finalize, use action_type=submit_report.
- Keep assistant_message concise and helpful.
"""


class ChatAuditor:
    def __init__(self, task_id: int, seed: int) -> None:
        if not API_BASE_URL or not MODEL_NAME or not API_KEY:
            raise RuntimeError("Set API_BASE_URL, MODEL_NAME, and HF_TOKEN/OPENAI_API_KEY.")
        self.client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        self.memory = MemoryStore(MEMORY_PATH)
        self.orchestrator = MultiAgentOrchestrator(memory=self.memory)
        self.task_id = task_id
        self.seed = seed
        self.history: list[dict[str, Any]] = []
        self.obs = self.call_env("reset", {"task_id": task_id, "seed": seed})

    def call_env(self, endpoint: str, payload: dict | None = None, method: str = "POST") -> dict:
        url = f"{ENV_URL}/{endpoint}"
        if method == "POST":
            r = requests.post(url, json=payload or {}, timeout=30)
        else:
            r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.json()

    def build_user_payload(self, user_text: str) -> str:
        view = {
            "user_request": user_text,
            "task_id": self.obs.get("task_id"),
            "task_description": self.obs.get("task_description"),
            "table_name": self.obs.get("table_name"),
            "schema": self.obs.get("schema"),
            "row_count": self.obs.get("row_count"),
            "step": self.obs.get("step"),
            "max_steps": self.obs.get("max_steps"),
            "last_query_result": (self.obs.get("last_query_result") or [])[:5],
            "last_action_error": self.obs.get("last_action_error"),
            "recent_history": self.history[-6:],
        }
        return json.dumps(view)

    def decide(self, user_text: str) -> dict:
        base_queries = [
            f"SELECT COUNT(*) AS n FROM {self.obs['table_name']}",
            f"SELECT * FROM {self.obs['table_name']} LIMIT 5",
        ]
        plan = self.orchestrator.build_chat_response(
            user_text=user_text,
            obs=self.obs,
            task_id=self.task_id,
            base_queries=base_queries,
            reasoning_hints=[],
        )
        return {
            "assistant_message": plan.assistant_message,
            "action": plan.action,
            "hypotheses": plan.hypotheses,
            "selected_queries": plan.selected_queries,
        }

    def step(self, user_text: str) -> tuple[str, dict]:
        decision = self.decide(user_text)
        assistant_message = str(decision.get("assistant_message", ""))
        action = decision.get("action", {"action_type": "query", "sql": f"SELECT COUNT(*) FROM {self.obs['table_name']}"})

        out = self.call_env("step", {"action": action})
        self.obs = out.get("observation", self.obs)
        reward = out.get("reward", {})

        self.history.append(
            {
                "user": user_text,
                "assistant_message": assistant_message,
                "action_type": action.get("action_type"),
                "reward": reward.get("value", 0.0),
                "done": reward.get("done", False),
                "selected_queries": decision.get("selected_queries", []),
            }
        )
        self.memory.save()
        return assistant_message, out


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat-like AI auditor for DataQualityEnv")
    parser.add_argument("--task-id", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    auditor = ChatAuditor(task_id=args.task_id, seed=args.seed)
    print(f"Chat auditor ready for task {args.task_id}. Type 'finalize' to submit, 'exit' to quit.")

    while True:
        user_text = input("you> ").strip()
        if user_text.lower() in {"exit", "quit"}:
            break
        if user_text.lower() == "finalize":
            user_text = "Finalize and submit the best report now."

        msg, result = auditor.step(user_text)
        reward = result.get("reward", {})
        print(f"agent> {msg}")
        print(f"reward={reward.get('value', 0.0)} done={reward.get('done', False)}")
        if reward.get("done"):
            print("Episode complete.")
            break


if __name__ == "__main__":
    main()
