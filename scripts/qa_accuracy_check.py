from __future__ import annotations

import json
import time
import requests

BASE = "http://localhost:7860"


def post(path: str, payload: dict) -> dict:
    r = requests.post(BASE + path, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def task1_bad_vs_good(seed: int = 42) -> dict:
    post("/reset", {"task_id": 1, "seed": seed})
    bad = {
        "action": {
            "action_type": "submit_report",
            "report": {
                "null_issues": {},
                "duplicate_row_count": {"value": 0, "confidence": 0.1},
                "schema_violations": [],
                "drifted_columns": [],
                "drift_details": {},
                "relational_issues": [],
                "recommended_fixes": [],
            },
        }
    }
    bad_score = post("/step", bad)["reward"]["value"]

    post("/reset", {"task_id": 1, "seed": seed})
    q1 = post(
        "/step",
        {
            "action": {
                "action_type": "query",
                "sql": "SELECT SUM(CASE WHEN email IS NULL OR lower(trim(cast(email as varchar))) IN ('null','n/a','unknown','-','','0','none') THEN 1 ELSE 0 END) AS email_null_total, SUM(CASE WHEN customer_id IS NULL THEN 1 ELSE 0 END) AS cid_nulls FROM customers",
            }
        },
    )
    q2 = post(
        "/step",
        {
            "action": {
                "action_type": "query",
                "sql": "SELECT COALESCE(SUM(c-1),0) AS exact_duplicate_rows FROM (SELECT customer_id,email,name,signup_date,country, COUNT(*) c FROM customers GROUP BY 1,2,3,4,5 HAVING COUNT(*)>1) t",
            }
        },
    )
    r1 = (q1.get("observation", {}).get("last_query_result") or [{}])[0]
    r2 = (q2.get("observation", {}).get("last_query_result") or [{}])[0]
    good = {
        "action": {
            "action_type": "submit_report",
            "report": {
                "null_issues": {
                    "email": {"value": int(r1.get("email_null_total", 0) or 0), "confidence": 0.92},
                    "customer_id": {"value": int(r1.get("cid_nulls", 0) or 0), "confidence": 0.92},
                },
                "duplicate_row_count": {"value": int(r2.get("exact_duplicate_rows", 0) or 0), "confidence": 0.9},
                "schema_violations": [
                    {
                        "column": "customers",
                        "issue_type": "near_duplicate_pattern",
                        "example": "country changed",
                        "count": 1,
                        "confidence": 0.6,
                    }
                ],
                "drifted_columns": [],
                "drift_details": {},
                "relational_issues": [],
                "recommended_fixes": ["dedupe and normalize disguised nulls"],
            },
        }
    }
    good_score = post("/step", good)["reward"]["value"]
    return {"task1_bad_score": bad_score, "task1_good_score": good_score}


def task3_bad_vs_good(seed: int = 42) -> dict:
    post("/reset", {"task_id": 3, "seed": seed})
    bad = {
        "action": {
            "action_type": "submit_report",
            "report": {
                "null_issues": {},
                "duplicate_row_count": {"value": 0, "confidence": 0.1},
                "schema_violations": [],
                "drifted_columns": [],
                "drift_details": {},
                "relational_issues": [],
                "recommended_fixes": [],
            },
        }
    }
    bad_score = post("/step", bad)["reward"]["value"]

    post("/reset", {"task_id": 3, "seed": seed})
    m = post(
        "/step",
        {
            "action": {
                "action_type": "query",
                "sql": "SELECT (SELECT AVG(amount) FROM transactions_baseline) AS baseline_mean, (SELECT AVG(amount) FROM transactions_current) AS current_mean",
            }
        },
    )
    c = post(
        "/step",
        {
            "action": {
                "action_type": "query",
                "sql": "SELECT DISTINCT c.category FROM transactions_current c LEFT JOIN (SELECT DISTINCT category FROM transactions_baseline) b ON c.category=b.category WHERE b.category IS NULL ORDER BY c.category",
            }
        },
    )
    u = post(
        "/step",
        {
            "action": {
                "action_type": "query",
                "sql": "SELECT AVG(CASE WHEN user_id >= 3000 THEN 1.0 ELSE 0.0 END) AS new_user_row_pct FROM transactions_current",
            }
        },
    )
    mr = (m.get("observation", {}).get("last_query_result") or [{}])[0]
    ur = (u.get("observation", {}).get("last_query_result") or [{}])[0]
    cats = [str(x.get("category")) for x in (c.get("observation", {}).get("last_query_result") or []) if x.get("category") is not None]
    good = {
        "action": {
            "action_type": "submit_report",
            "report": {
                "null_issues": {},
                "duplicate_row_count": {"value": 0, "confidence": 0.6},
                "schema_violations": [],
                "drifted_columns": ["amount", "category", "user_id"],
                "drift_details": {
                    "amount": {"value": f"mean shift from {float(mr.get('baseline_mean', 0.0) or 0.0):.2f} to {float(mr.get('current_mean', 0.0) or 0.0):.2f}", "confidence": 0.9},
                    "category": {"value": ",".join(cats), "confidence": 0.88},
                    "user_id": {"value": f"{float(ur.get('new_user_row_pct', 0.0) or 0.0)*100:.1f}%", "confidence": 0.87},
                },
                "relational_issues": [],
                "recommended_fixes": ["enable drift monitors"],
            },
        }
    }
    good_score = post("/step", good)["reward"]["value"]
    return {"task3_bad_score": bad_score, "task3_good_score": good_score}


def main() -> None:
    t0 = time.time()
    health = requests.get(BASE + "/health", timeout=10).json()
    t1 = task1_bad_vs_good(42)
    t3 = task3_bad_vs_good(42)
    out = {
        "health": health,
        "scorer_sensitivity": {**t1, **t3},
        "elapsed_sec": round(time.time() - t0, 3),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
