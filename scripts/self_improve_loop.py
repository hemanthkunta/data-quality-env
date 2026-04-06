from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], env: dict[str, str] | None = None) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
    return p.returncode, (p.stdout + "\n" + p.stderr).strip()


def parse_scores(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip().lower()
        if line.startswith("task_") and ":" in line:
            k, v = line.split(":", 1)
            try:
                out[k.strip()] = float(v.strip())
            except Exception:
                pass
        if line.startswith("mean:"):
            try:
                out["mean"] = float(line.split(":", 1)[1].strip())
            except Exception:
                pass
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-improvement loop: RL tune + evaluate advanced agent")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--episodes-per-cycle", type=int, default=200)
    parser.add_argument("--policy-path", type=str, default="outputs/rl_policy.json")
    parser.add_argument("--memory-path", type=str, default="outputs/agent_memory.json")
    args = parser.parse_args()

    env = os.environ.copy()
    env["RL_POLICY_PATH"] = args.policy_path
    env["AGENT_MEMORY_PATH"] = args.memory_path

    best = {"mean": -1.0}

    for c in range(1, args.cycles + 1):
        print(f"\n=== self-improve cycle {c}/{args.cycles} ===")
        rc, txt = run(
            [
                sys.executable,
                "scripts/train_rl_agent.py",
                "train",
                "--episodes",
                str(args.episodes_per_cycle),
                "--output",
                args.policy_path,
            ],
            env=env,
        )
        print(txt)
        if rc != 0:
            print("train step failed; aborting")
            break

        rc, txt = run([sys.executable, "high_grade_agent.py"], env=env)
        print(txt)
        if rc != 0:
            print("agent eval failed; aborting")
            break

        scores = parse_scores(txt)
        if scores.get("mean", -1.0) > best.get("mean", -1.0):
            best = scores

    summary_path = ROOT / "outputs" / "self_improve_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({"best": best}, indent=2))
    print(f"\nSaved summary -> {summary_path}")
    print(json.dumps({"best": best}, indent=2))


if __name__ == "__main__":
    main()
