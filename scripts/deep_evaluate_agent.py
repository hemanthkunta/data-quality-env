from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / "high_grade_agent.py"


def parse_scores(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in text.splitlines():
        m = re.search(r"task_(\d)\s*:\s*([0-9.]+)", line)
        if m:
            out[f"task_{m.group(1)}"] = float(m.group(2))
        m2 = re.search(r"mean\s*:\s*([0-9.]+)", line)
        if m2:
            out["mean"] = float(m2.group(1))
    return out


def run_once(seed: int, env: dict[str, str]) -> dict[str, float]:
    env2 = dict(env)
    env2["SEED"] = str(seed)
    p = subprocess.run([sys.executable, str(AGENT)], cwd=str(ROOT), env=env2, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr or p.stdout)
    return parse_scores(p.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deeper multi-seed evaluator for high-grade agent")
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    env = os.environ.copy()
    env.setdefault("RL_POLICY_PATH", str(ROOT / "outputs" / "rl_policy.json"))
    env.setdefault("AGENT_MEMORY_PATH", str(ROOT / "outputs" / "agent_memory.json"))
    env.setdefault("ENV_URL", "http://localhost:7860")

    rows: list[dict[str, float]] = []
    for i in range(args.runs):
        seed = args.seed_start + i
        score = run_once(seed, env)
        score["seed"] = float(seed)
        rows.append(score)

    agg: dict[str, float] = {}
    keys = ["task_1", "task_2", "task_3", "mean"]
    for k in keys:
        vals = [r.get(k, 0.0) for r in rows]
        agg[f"{k}_avg"] = sum(vals) / max(1, len(vals))

    payload = {"runs": rows, "aggregate": agg}
    out_file = ROOT / "outputs" / "deep_eval_summary.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(payload, indent=2))

    print(json.dumps(payload, indent=2))
    print(f"saved: {out_file}")


if __name__ == "__main__":
    main()
