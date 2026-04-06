import json
import os
import requests

BASE = os.environ.get("ENV_URL", "http://localhost:7860").rstrip("/")


def get(path: str):
    r = requests.get(f"{BASE}{path}", timeout=30)
    r.raise_for_status()
    return r.json()


def post(path: str, payload: dict):
    r = requests.post(f"{BASE}{path}", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def reset(task_id: int, seed: int = 42):
    return post("/reset", {"task_id": task_id, "seed": seed})


def step(action: dict):
    return post("/step", {"action": action})


def must(cond: bool, msg: str):
    if not cond:
        raise SystemExit(f"[FAIL] {msg}")


def main():
    print("[1] /health")
    h = get("/health")
    must(h.get("status") == "ok", "/health status must be ok")
    print("  ok")

    print("[2] /reset task1")
    obs = reset(1, 42)
    must(obs["task_id"] == 1, "task_id mismatch")
    must("schema" in obs and "row_count" in obs, "invalid observation")
    print("  ok")

    print("[3] /step query")
    out = step({"action_type": "query", "sql": "SELECT COUNT(*) AS n FROM customers"})
    must("reward" in out and "observation" in out, "step response malformed")
    must(out["reward"]["done"] is False, "query should not end episode")
    print("  ok")

    print("[4] safety guard")
    out = step({"action_type": "query", "sql": "DROP TABLE customers"})
    must(out["reward"]["value"] == -0.2, "DROP should be penalized -0.2")
    print("  ok")

    print("[5] grader dynamics")
    empty = {
        "action_type": "submit_report",
        "report": {
            "null_issues": {},
            "duplicate_row_count": 0,
            "schema_violations": [],
            "drifted_columns": [],
            "drift_details": {},
            "recommended_fixes": [],
        },
    }
    reset(1, 42)
    s0 = step(empty)["reward"]["value"]

    better = {
        "action_type": "submit_report",
        "report": {
            "null_issues": {"email": 10, "customer_id": 4},
            "duplicate_row_count": 15,
            "schema_violations": [],
            "drifted_columns": [],
            "drift_details": {},
            "recommended_fixes": ["fill nulls", "deduplicate"],
        },
    }
    reset(1, 42)
    s1 = step(better)["reward"]["value"]
    must(s1 >= s0, "better report should not score worse")
    print(f"  ok (empty={s0:.3f}, better={s1:.3f})")

    print("[PASS] local QA complete")
    print(json.dumps({"base_url": BASE}, indent=2))


if __name__ == "__main__":
    main()
