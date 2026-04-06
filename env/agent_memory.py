from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class MemoryItem:
    task_id: int
    seed: int
    score: float
    query_plan: list[str]
    evidence: dict[str, Any]


class MemoryStore:
    """Simple persistent memory for agent self-improvement."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._items: list[MemoryItem] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self._items = []
            return
        try:
            payload = json.loads(self.path.read_text())
            raw = payload.get("items", []) if isinstance(payload, dict) else []
            items: list[MemoryItem] = []
            for r in raw:
                items.append(
                    MemoryItem(
                        task_id=int(r.get("task_id", 0)),
                        seed=int(r.get("seed", 0)),
                        score=float(r.get("score", 0.0)),
                        query_plan=[str(x) for x in r.get("query_plan", [])],
                        evidence=dict(r.get("evidence", {})),
                    )
                )
            self._items = items
        except Exception:
            self._items = []

    def save(self) -> None:
        payload = {
            "version": 1,
            "items": [
                {
                    "task_id": i.task_id,
                    "seed": i.seed,
                    "score": i.score,
                    "query_plan": i.query_plan,
                    "evidence": i.evidence,
                }
                for i in self._items
            ],
        }
        self.path.write_text(json.dumps(payload))

    def add(self, item: MemoryItem, max_items: int = 500) -> None:
        self._items.append(item)
        # keep highest-scoring memories per task
        self._items.sort(key=lambda x: (x.task_id, x.score), reverse=True)
        self._items = self._items[:max_items]

    def top_for_task(self, task_id: int, k: int = 5) -> list[MemoryItem]:
        rows = [i for i in self._items if i.task_id == task_id]
        rows.sort(key=lambda x: x.score, reverse=True)
        return rows[:k]

    def query_bias(self, task_id: int, queries: list[str], k: int = 5) -> list[float]:
        """Returns additive prior bias per query from successful memories."""
        top = self.top_for_task(task_id, k=k)
        if not top:
            return [0.0 for _ in queries]

        bias = [0.0 for _ in queries]
        for mem in top:
            for rank, q in enumerate(mem.query_plan):
                if q in queries:
                    i = queries.index(q)
                    # Earlier query in successful run gets stronger weight.
                    bias[i] += max(0.0, 0.08 - 0.02 * rank) * max(0.0, mem.score)
        return bias
