from __future__ import annotations

import re
import threading
from typing import Any

import duckdb

BLOCKED = re.compile(
    r"\b(DROP|TRUNCATE|DELETE|INSERT|UPDATE|CREATE|ALTER|ATTACH|COPY|EXPORT|IMPORT)\b",
    re.IGNORECASE,
)
MAX_ROWS = 100
_lock = threading.Lock()


class SQLEngine:
    def __init__(self) -> None:
        self.conn = duckdb.connect(":memory:")

    def load_tables(self, tables: dict[str, Any]) -> None:
        with _lock:
            for name, df in tables.items():
                self.conn.register(name, df)
                self.conn.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM {name}")
                self.conn.unregister(name)

    def execute(self, sql: str) -> list[dict] | str:
        s = (sql or "").strip()
        if BLOCKED.search(s):
            return "ERROR: Destructive SQL (DROP/DELETE/UPDATE/etc.) is not permitted."
        with _lock:
            try:
                rel = self.conn.execute(s)
                cols = [d[0] for d in rel.description]
                rows = rel.fetchmany(MAX_ROWS)
                return [dict(zip(cols, row)) for row in rows]
            except Exception as e:
                return f"ERROR: {e}"

    def run_fix_sql(self, sql: str, gold_clean: dict[str, Any] | None = None) -> float:
        s = (sql or "").strip()
        # Only allow UPDATE during fix phase.
        if re.search(r"\b(DROP|TRUNCATE|DELETE|INSERT|CREATE|ALTER|ATTACH|COPY|EXPORT|IMPORT)\b", s, re.IGNORECASE):
            return 0.0
        if not re.search(r"\bUPDATE\b", s, re.IGNORECASE):
            return 0.0
        with _lock:
            try:
                self.conn.execute(s)
                # Lightweight deterministic scoring placeholder.
                return 0.5
            except Exception:
                return 0.0

    def get_table_schemas(self, tables: list[str]) -> dict[str, dict[str, str]]:
        out: dict[str, dict[str, str]] = {}
        with _lock:
            for t in tables:
                rows = self.conn.execute(f"PRAGMA table_info('{t}')").fetchall()
                out[t] = {r[1]: str(r[2]) for r in rows}
        return out

    def get_row_counts(self, tables: list[str]) -> dict[str, int]:
        out: dict[str, int] = {}
        with _lock:
            for t in tables:
                out[t] = int(self.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
        return out

    def close(self) -> None:
        self.conn.close()
