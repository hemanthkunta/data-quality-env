import os
import requests

BASE = os.environ.get("ENV_URL", "http://localhost:7860").rstrip("/")

def call(method, path, payload=None):
    url = f"{BASE}{path}"
    if method == "GET":
        r = requests.get(url, timeout=20)
    else:
        r = requests.post(url, json=payload or {}, timeout=20)
    r.raise_for_status()
    return r.json()

def main():
    print("health:", call("GET", "/health"))
    obs = call("POST", "/reset", {"task_id": 1, "seed": 42})
    print("reset.table:", obs["table_name"], "rows:", obs["row_count"])
    out = call("POST", "/step", {"action": {"action_type": "query", "sql": "SELECT * FROM customers LIMIT 3"}})
    print("step.reward:", out["reward"]["value"], "done:", out["reward"]["done"])
    print("smoke test passed")

if __name__ == "__main__":
    main()
