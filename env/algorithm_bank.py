from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from hashlib import sha1


_ALGO_BANK: list["AlgorithmSpec"] | None = None
_BEST_SPEC_CACHE: dict[str, "AlgorithmSpec"] = {}


@dataclass(frozen=True)
class AlgorithmSpec:
    algorithm_id: int
    w_coverage: float
    w_stat: float
    w_risk: float
    w_novelty: float
    w_limit: float
    w_prior: float
    repeat_penalty: float


def generate_100k_algorithms() -> list[AlgorithmSpec]:
    """Generate exactly 100,000 deterministic algorithm specs."""
    global _ALGO_BANK
    if _ALGO_BANK is not None:
        return _ALGO_BANK

    out: list[AlgorithmSpec] = []
    # 10 * 10 * 10 * 10 * 5 * 2 = 100,000
    grids = [
        [i / 10 for i in range(10)],
        [i / 10 for i in range(10)],
        [i / 10 for i in range(10)],
        [i / 10 for i in range(10)],
        [i / 5 for i in range(5)],
        [0.0, 1.0],
    ]

    idx = 0
    for a, b, c, d, e, f in itertools.product(*grids):
        out.append(
            AlgorithmSpec(
                algorithm_id=idx,
                w_coverage=a,
                w_stat=b,
                w_risk=c,
                w_novelty=d,
                w_limit=e,
                w_prior=(idx % 5) / 5,
                repeat_penalty=f * 0.03,
            )
        )
        idx += 1

    _ALGO_BANK = out
    return _ALGO_BANK


def _query_features(sql: str) -> dict[str, float]:
    s = (sql or "").lower()
    return {
        "coverage": float(any(k in s for k in ["count(", "sum(", "avg(", "group by", "distinct"])),
        "stat": float(any(k in s for k in ["avg(", "stddev", "variance", "percentile", "try_cast", "strptime"])),
        "risk": float(any(k in s for k in ["drop", "truncate", "delete", "insert", "update", "alter", "create"])),
        "novelty": float(any(k in s for k in ["left join", "except", "not in", "having", "case when"])),
        "has_limit": float("limit" in s),
    }


def _task_relevance(task_id: int, sql: str) -> float:
    s = (sql or "").lower()
    if task_id == 1:
        keys = ["null", "email", "customer_id", "duplicate", "group by"]
    elif task_id == 2:
        keys = ["quantity", "amount", "n/a", "try_cast", "order_date"]
    else:
        keys = ["transactions_baseline", "transactions_current", "category", "user_id", "avg(amount)"]
    hits = sum(1 for k in keys if k in s)
    return hits / max(1, len(keys))


def algorithm_rule_check(spec: AlgorithmSpec, queries: list[str], max_steps: int = 10) -> bool:
    """
    Enforces constraints aligned with hackathon rules for this environment:
    - non-destructive SQL preference
    - bounded steps
    - deterministic finite parameters
    """
    if max_steps <= 0 or max_steps > 10:
        return False
    if spec.w_risk < 0.0 or spec.w_risk > 1.0:
        return False
    if spec.repeat_penalty < 0.0 or spec.repeat_penalty > 0.03:
        return False

    for q in queries:
        s = (q or "").strip()
        if not s:
            return False
        if re.search(r"\b(drop|truncate|delete|insert|update|alter|create)\b", s, flags=re.IGNORECASE):
            return False
        if not re.match(r"^\s*(select|with)\b", s, flags=re.IGNORECASE):
            return False
    return True


def rank_queries(task_id: int, queries: list[str], priors: list[float], spec: AlgorithmSpec) -> list[int]:
    scored: list[tuple[int, float]] = []
    for i, q in enumerate(queries):
        f = _query_features(q)
        prior = priors[i] if i < len(priors) else 0.0
        relevance = _task_relevance(task_id, q)
        score = (
            spec.w_coverage * f["coverage"]
            + spec.w_stat * f["stat"]
            + spec.w_novelty * f["novelty"]
            + spec.w_limit * f["has_limit"]
            + spec.w_prior * prior
            + 0.8 * relevance
            - spec.w_risk * f["risk"]
        )
        scored.append((i, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in scored]


def choose_best_algorithm(task_id: int, queries: list[str], priors: list[float], max_algorithms: int = 100_000) -> AlgorithmSpec:
    key_payload = f"t={task_id}|n={len(queries)}|m={max_algorithms}|q={'||'.join(queries)}|p={','.join(f'{x:.4f}' for x in priors)}"
    cache_key = sha1(key_payload.encode("utf-8")).hexdigest()
    if cache_key in _BEST_SPEC_CACHE:
        return _BEST_SPEC_CACHE[cache_key]

    algorithms = generate_100k_algorithms()
    n = min(max_algorithms, len(algorithms))

    best = algorithms[0]
    best_obj = -1e18

    for spec in algorithms[:n]:
        if not algorithm_rule_check(spec, queries, max_steps=10):
            continue
        ranking = rank_queries(task_id, queries, priors, spec)
        top = ranking[:2]
        obj = 0.0
        for pos, i in enumerate(top):
            base = 2.0 - pos
            rel = _task_relevance(task_id, queries[i])
            obj += base * rel
        # Prefer slight risk aversion
        obj -= 0.1 * spec.w_risk
        if obj > best_obj:
            best_obj = obj
            best = spec

    _BEST_SPEC_CACHE[cache_key] = best
    return best


def order_queries_with_100k_algorithms(task_id: int, queries: list[str], priors: list[float]) -> list[str]:
    spec = choose_best_algorithm(task_id, queries, priors, max_algorithms=100_000)
    ranked_idx = rank_queries(task_id, queries, priors, spec)
    return [queries[i] for i in ranked_idx]
