from __future__ import annotations

import threading
from typing import Any

from env import app as env_app


class InProcessEnvBackend:
    def __init__(self) -> None:
        self._lock = threading.Lock()

    def call(self, endpoint: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload or {}
        with self._lock:
            if endpoint == "reset":
                return env_app.reset(payload)
            if endpoint == "step":
                return env_app.step(payload)
            if endpoint == "state":
                return env_app.get_state()
            if endpoint == "health":
                return env_app.health()
        raise ValueError(f"Unsupported endpoint: {endpoint}")

    def reset(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.call("reset", payload)

    def step(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.call("step", payload)

    def state(self) -> dict[str, Any]:
        return self.call("state")

    def health(self) -> dict[str, Any]:
        return self.call("health")


BACKEND = InProcessEnvBackend()
