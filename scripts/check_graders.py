import requests

BASE = "http://localhost:7860"

def post(path, payload):
    r = requests.post(f"{BASE}{path}", json=payload, timeout=20)
    r.raise_for_status()
    return r.json()

def score(task_id, report):
    post("/reset", {"task_id": task_id, "seed": 42})
    out = post("/step", {"action": {"action_type": "submit_report", "report": report}})
    return out["reward"]["value"], out["reward"].get("breakdown", {})

EMPTY = {
    "null_issues": {},
    "duplicate_row_count": 0,
    "schema_violations": [],
    "drifted_columns": [],
    "drift_details": {},
    "recommended_fixes": []
}

BETTER_T1 = {
    "null_issues": {"email": 10, "customer_id": 4},
    "duplicate_row_count": 15,
    "schema_violations": [],
    "drifted_columns": [],
    "drift_details": {},
    "recommended_fixes": ["dedupe rows", "fill nulls"]
}

BETTER_T2 = {
    "null_issues": {"negative_quantity_rows": 7},
    "duplicate_row_count": 0,
    "schema_violations": [
        {"column": "amount", "issue_type": "type_violation", "example": "$12.50"},
        {"column": "order_date", "issue_type": "date_format_violation", "example": "Jan 5 2024"},
        {"column": "amount", "issue_type": "unparseable", "example": "N/A"},
        {"column": "quantity", "issue_type": "negative_value", "example": "-3"}
    ],
    "drifted_columns": [],
    "drift_details": {},
    "recommended_fixes": ["parse amount", "normalize date", "clamp quantity"]
}

BETTER_T3 = {
    "null_issues": {},
    "duplicate_row_count": 0,
    "schema_violations": [{"column": "category", "issue_type": "new_values", "example": "crypto, NFT"}],
    "drifted_columns": ["amount"],
    "drift_details": {"amount": "mean shifted from ~50 to ~78", "user_id": "new users around 15%"},
    "recommended_fixes": ["monitor drift", "update reference sets"]
}

def main():
    for task_id, better in [(1, BETTER_T1), (2, BETTER_T2), (3, BETTER_T3)]:
        s0, _ = score(task_id, EMPTY)
        s1, b1 = score(task_id, better)
        print(f"task {task_id}: empty={s0:.3f} better={s1:.3f} breakdown={b1}")
        if s1 < s0:
            raise SystemExit(f"Unexpected scoring regression on task {task_id}")
    print("grader dynamics check passed")

if __name__ == "__main__":
    main()
