from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class AlgoConfig:
    w_coverage: float
    w_stat: float
    w_risk: float
    w_novelty: float
    limit_bonus: float
    repeat_penalty: float


def _query_features(sql: str) -> dict[str, float]:
    s = (sql or "").lower()
    return {
        "coverage": float(any(k in s for k in ["count(", "sum(", "avg(", "group by", "distinct"])),
        "stat": float(any(k in s for k in ["avg(", "stddev", "variance", "percentile", "try_cast", "strptime"])),
        "risk": float(any(k in s for k in ["drop", "truncate", "delete", "insert", "update", "alter", "create"])),
        "novelty": float(any(k in s for k in ["left join", "except", "not in", "having", "case when"])),
        "has_limit": float("limit" in s),
    }


def _task_keywords(task_id: int) -> list[str]:
    if task_id == 1:
        return ["null", "email", "customer_id", "duplicate", "group by"]
    if task_id == 2:
        return ["quantity", "amount", "n/a", "try_cast", "order_date"]
    return ["transactions_baseline", "transactions_current", "category", "user_id", "avg(amount)"]


def _task_relevance(task_id: int, sql: str) -> float:
    s = (sql or "").lower()
    keys = _task_keywords(task_id)
    hits = sum(1 for k in keys if k in s)
    return hits / max(1, len(keys))


def _sql_shape_penalty(sql: str) -> float:
    # Penalize very long and likely redundant SQL in a constrained step budget.
    length = len(sql or "")
    if length < 120:
        return 0.0
    if length < 300:
        return 0.02
    return 0.05


def algorithm_config_stream() -> Iterable[AlgoConfig]:
    # 11^4 * 7^2 = 717,409 total algorithm configurations.
    grid_a = [i / 10 for i in range(0, 11)]
    grid_b = [i / 20 for i in range(0, 7)]
    for a, b, c, d, e, f in itertools.product(grid_a, grid_a, grid_a, grid_a, grid_b, grid_b):
        yield AlgoConfig(
            w_coverage=a,
            w_stat=b,
            w_risk=c,
            w_novelty=d,
            limit_bonus=e,
            repeat_penalty=f,
        )


def _config_query_score(task_id: int, sql: str, cfg: AlgoConfig, q_prior: float) -> float:
    f = _query_features(sql)
    relevance = _task_relevance(task_id, sql)
    penalty_len = _sql_shape_penalty(sql)
    score = (
        cfg.w_coverage * f["coverage"]
        + cfg.w_stat * f["stat"]
        + cfg.w_novelty * f["novelty"]
        + cfg.limit_bonus * f["has_limit"]
        + 0.6 * relevance
        + 0.4 * q_prior
        - cfg.w_risk * f["risk"]
        - penalty_len
    )
    return score


def _ranking_for_config(task_id: int, queries: list[str], cfg: AlgoConfig, priors: list[float]) -> list[int]:
    pairs = []
    for i, q in enumerate(queries):
        pairs.append((i, _config_query_score(task_id, q, cfg, priors[i])))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in pairs]


def select_best_config(task_id: int, queries: list[str], priors: list[float], max_configs: int = 100_000) -> AlgoConfig:
    best_cfg = None
    best_obj = -10**9

    for idx, cfg in enumerate(algorithm_config_stream()):
        if idx >= max_configs:
            break
        ranking = _ranking_for_config(task_id, queries, cfg, priors)

        # Objective: prioritize top-2 quality and diversity in SQL intent.
        top = ranking[:2]
        top_score = sum(_config_query_score(task_id, queries[i], cfg, priors[i]) for i in top)

        intents = set()
        for i in top:
            s = queries[i].lower()
            intent = "join" if any(k in s for k in ["join", "except", "not in"]) else "agg"
            intents.add(intent)
        diversity_bonus = 0.05 if len(intents) > 1 else 0.0

        obj = top_score + diversity_bonus
        if obj > best_obj:
            best_obj = obj
            best_cfg = cfg

    return best_cfg if best_cfg is not None else AlgoConfig(0.5, 0.5, 1.0, 0.5, 0.0, 0.0)


def ensemble_order(task_id: int, queries: list[str], priors: list[float], max_configs: int = 100_000) -> list[str]:
    cfg = select_best_config(task_id, queries, priors, max_configs=max_configs)
    ranking = _ranking_for_config(task_id, queries, cfg, priors)

    # De-prioritize unsafe SQL just in case external user-provided probes are included.
    safe = []
    unsafe = []
    for i in ranking:
        if re.search(r"\b(drop|truncate|delete|insert|update|alter|create)\b", queries[i], re.IGNORECASE):
            unsafe.append(queries[i])
        else:
            safe.append(queries[i])
    return safe + unsafe
