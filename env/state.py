from __future__ import annotations

from typing import Any

from env.models import EpisodeState


def export_state(st: EpisodeState | None) -> dict[str, Any]:
    if st is None:
        return {"task_id": None, "seed": None, "step": 0, "done": False}
    return st.model_dump()
