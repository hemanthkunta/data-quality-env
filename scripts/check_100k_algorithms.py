from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.algorithm_bank import algorithm_rule_check, generate_100k_algorithms


def main() -> None:
    algos = generate_100k_algorithms()
    assert len(algos) == 100_000, f"Expected 100000 algorithms, got {len(algos)}"

    # Representative safe probe set aligned with environment constraints.
    queries = [
        "SELECT * FROM customers LIMIT 5",
        "SELECT COUNT(*) FROM orders",
        "WITH t AS (SELECT AVG(amount) a FROM transactions_current) SELECT * FROM t",
    ]

    valid = sum(1 for a in algos if algorithm_rule_check(a, queries, max_steps=10))
    print({"total_algorithms": len(algos), "valid_algorithms": valid})


if __name__ == "__main__":
    main()
