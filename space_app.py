from __future__ import annotations

import json
import os
import re
from typing import Any

import gradio as gr
from fastapi import Body, FastAPI, HTTPException

from env.inprocess_backend import BACKEND

SESSION = BACKEND


def health() -> dict[str, str]:
    return {"status": "ok", "env": "DataQualityEnv", "mode": "space-ui"}


def session_status(obs: dict[str, Any] | None) -> str:
    if not obs:
        return "No active episode. Choose a task and click Reset."
    return (
        f"Task {obs.get('task_id')} | phase={obs.get('phase')} | step={obs.get('step')}/{obs.get('max_steps')} | "
        f"credits={obs.get('query_credits_remaining')}"
    )


def initial_chat() -> list[dict[str, str]]:
    return []


def format_observation(obs: dict[str, Any] | None) -> str:
    return json.dumps(obs or {}, indent=2, default=str)


def format_reward(reward: dict[str, Any] | None) -> str:
    return json.dumps(reward or {}, indent=2, default=str)


def task_hint(task_id: int) -> str:
    if task_id == 1:
        return "Try null-like value checks and duplicate-row grouping on the customers table."
    if task_id == 2:
        return "Try type parsing, negative values, and date-format checks on orders."
    if task_id == 3:
        return "Try baseline/current comparisons, new categories, and user population drift."
    return "Try orphaned foreign keys, temporal checks, and aggregate consistency."


def heuristic_queries(task_id: int) -> list[str]:
    if task_id == 1:
        return [
            "SELECT COUNT(*) AS total_rows FROM customers",
            "SELECT SUM(CASE WHEN email IS NULL OR lower(trim(cast(email as varchar))) IN ('null','n/a','unknown','-','','0','none') THEN 1 ELSE 0 END) AS email_null_total, SUM(CASE WHEN customer_id IS NULL THEN 1 ELSE 0 END) AS cid_nulls FROM customers",
            "SELECT COALESCE(SUM(c-1),0) AS exact_duplicate_rows FROM (SELECT customer_id,email,name,signup_date,country, COUNT(*) c FROM customers GROUP BY 1,2,3,4,5 HAVING COUNT(*)>1) t",
        ]
    if task_id == 2:
        return [
            "SELECT SUM(CASE WHEN quantity < 0 THEN 1 ELSE 0 END) AS neg_qty, SUM(CASE WHEN try_cast(replace(amount,'$','') AS DOUBLE) IS NULL THEN 1 ELSE 0 END) AS bad_amt FROM orders",
            "SELECT amount, order_date FROM orders LIMIT 10",
        ]
    if task_id == 3:
        return [
            "SELECT (SELECT AVG(amount) FROM transactions_baseline) AS baseline_mean, (SELECT AVG(amount) FROM transactions_current) AS current_mean",
            "SELECT DISTINCT c.category FROM transactions_current c LEFT JOIN (SELECT DISTINCT category FROM transactions_baseline) b ON c.category=b.category WHERE b.category IS NULL ORDER BY c.category",
        ]
    return [
        "SELECT COUNT(*) AS orphan_count FROM orders o LEFT JOIN customers c ON o.customer_id=c.customer_id WHERE c.customer_id IS NULL",
        "SELECT COUNT(*) AS temporal_count FROM orders WHERE try_cast(ship_date AS TIMESTAMP) < try_cast(order_date AS TIMESTAMP)",
        "SELECT COUNT(*) AS aggregate_count FROM (SELECT o.order_id, o.order_total, SUM(li.subtotal) AS s FROM orders o JOIN line_items li ON o.order_id=li.order_id GROUP BY o.order_id, o.order_total HAVING abs(o.order_total - SUM(li.subtotal)) > 1e-6) x",
    ]


def current_tables(obs: dict[str, Any] | None) -> set[str]:
    tables = (obs or {}).get("tables") or {}
    return {str(name).lower() for name in tables.keys()}


def referenced_tables(sql_text: str) -> set[str]:
    sql = normalize_command(sql_text)
    matches = re.finditer(r"\b(?:from|join)\s+([a-zA-Z_][\w\.]*)", sql, flags=re.IGNORECASE)
    refs: set[str] = set()
    for match in matches:
        identifier = match.group(1).split(".")[-1].lower()
        if identifier:
            refs.add(identifier)
    return refs


def validate_query_tables(sql_text: str, obs: dict[str, Any] | None) -> str | None:
    allowed = current_tables(obs)
    if not allowed:
        return None
    refs = referenced_tables(sql_text)
    if not refs:
        return None
    unknown = sorted(refs - allowed)
    if unknown:
        available = ", ".join(sorted(allowed))
        return f"This task only exposes: {available}. Please query one of those tables instead of: {', '.join(unknown)}."
    return None


def normalize_command(text: str) -> str:
    return (text or "").strip()


def parse_json_fragment(text: str) -> dict[str, Any] | None:
    raw = normalize_command(text)
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                return None
    return None


def fallback_report_from_obs(obs: dict[str, Any] | None) -> dict[str, Any]:
    task_id = int((obs or {}).get("task_id", 1) or 1)
    base = {
        "null_issues": {},
        "duplicate_row_count": {"value": 0, "confidence": 0.5},
        "schema_violations": [],
        "drifted_columns": [],
        "drift_details": {},
        "relational_issues": [],
        "recommended_fixes": [
            "Auto-submitted fallback report to avoid max_steps termination",
            "Run additional targeted probes in earlier steps for higher confidence",
        ],
    }
    if task_id == 1:
        base["schema_violations"] = [
            {
                "column": "customers",
                "issue_type": "partial_audit",
                "example": "auto_submit_guard",
                "count": 1,
                "confidence": 0.4,
            }
        ]
    return base


def reset_ui(task_id: int, seed: int):
    obs = SESSION.reset({"task_id": task_id, "seed": seed})
    chat = initial_chat()
    chat.append({"role": "assistant", "content": f"Reset complete for task {task_id}. {task_hint(task_id)}"})
    return chat, format_observation(obs), session_status(obs), format_reward({"value": 0.0, "done": False}), obs


def run_query(sql_text: str, current_obs: dict[str, Any] | None, chat: list[dict[str, str]]):
    if current_obs:
        step = int(current_obs.get("step", 0) or 0)
        max_steps = int(current_obs.get("max_steps", 12) or 12)
        if step >= max_steps - 1:
            chat = chat + [
                {
                    "role": "assistant",
                    "content": "Step budget is almost exhausted. Submit your report now (`submit: {...}`) to avoid `max_steps` termination.",
                }
            ]
            return chat, format_observation(current_obs), session_status(current_obs), format_reward({}), current_obs

    sql = normalize_command(sql_text)
    if not sql:
        chat = chat + [{"role": "assistant", "content": "Send a SQL query first."}]
        return chat, format_observation(current_obs), session_status(current_obs), format_reward({}), current_obs

    table_error = validate_query_tables(sql, current_obs)
    if table_error:
        chat = chat + [{"role": "assistant", "content": table_error}]
        return chat, format_observation(current_obs), session_status(current_obs), format_reward({"value": 0.0, "done": False}), current_obs

    out = SESSION.step({"action": {"action_type": "query", "sql": sql}})
    obs = out.get("observation")
    reward = out.get("reward")
    chat = chat + [
        {"role": "user", "content": f"query: {sql}"},
        {"role": "assistant", "content": f"Ran query. reward={reward.get('value', 0.0)}"},
    ]
    return chat, format_observation(obs), session_status(obs), format_reward(reward), obs


def submit_report(report_text: str, current_obs: dict[str, Any] | None, chat: list[dict[str, str]]):
    report = parse_json_fragment(report_text)
    if report is None:
        chat = chat + [{"role": "assistant", "content": "I couldn’t parse that as JSON. Paste a valid report object."}]
        return chat, format_observation(current_obs), session_status(current_obs), format_reward({}), current_obs

    out = SESSION.step({"action": {"action_type": "submit_report", "report": report}})
    obs = out.get("observation")
    reward = out.get("reward")
    chat = chat + [
        {"role": "user", "content": "submit report"},
        {"role": "assistant", "content": f"Submitted report. reward={reward.get('value', 0.0)}"},
    ]
    return chat, format_observation(obs), session_status(obs), format_reward(reward), obs


def auto_audit(current_obs: dict[str, Any] | None, chat: list[dict[str, str]]):
    if not current_obs:
        chat = chat + [{"role": "assistant", "content": "Reset a task before running auto audit."}]
        return chat, format_observation(current_obs), session_status(current_obs), format_reward({}), current_obs

    task_id = int(current_obs.get("task_id", 1) or 1)
    queries = heuristic_queries(task_id)
    running_chat = chat + [{"role": "assistant", "content": f"Running {len(queries)} diagnostic probes..."}]
    obs = current_obs
    reward = None
    for sql in queries:
        table_error = validate_query_tables(sql, obs)
        if table_error:
            running_chat.append({"role": "assistant", "content": table_error})
            continue
        out = SESSION.step({"action": {"action_type": "query", "sql": sql}})
        obs = out.get("observation")
        reward = out.get("reward")
        running_chat.append({"role": "user", "content": sql})
        running_chat.append({"role": "assistant", "content": f"reward={reward.get('value', 0.0)}"})
    return running_chat, format_observation(obs), session_status(obs), format_reward(reward), obs


def handle_command(user_text: str, current_obs: dict[str, Any] | None, chat: list[dict[str, str]], task_id: int, seed: int):
    text = normalize_command(user_text)
    if not text:
        return chat, format_observation(current_obs), session_status(current_obs), format_reward({}), current_obs

    lower = text.lower()
    if lower in {"help", "?"}:
        chat = chat + [{"role": "assistant", "content": "Commands: `reset`, `query: SELECT ...`, `submit: {...json...}`, `auto`, or `state`."}]
        return chat, format_observation(current_obs), session_status(current_obs), format_reward({}), current_obs

    if current_obs and not (lower.startswith("submit") or lower.startswith("reset") or lower == "state"):
        step = int(current_obs.get("step", 0) or 0)
        max_steps = int(current_obs.get("max_steps", 12) or 12)
        if step >= max_steps - 1:
            fallback = fallback_report_from_obs(current_obs)
            out = SESSION.step({"action": {"action_type": "submit_report", "report": fallback}})
            obs = out.get("observation", current_obs)
            reward = out.get("reward", {})
            chat = chat + [
                {
                    "role": "assistant",
                    "content": "Step budget exhausted. I auto-submitted a fallback report to prevent `max_steps` zero-output failure.",
                }
            ]
            return chat, format_observation(obs), session_status(obs), format_reward(reward), obs

    if lower.startswith("reset"):
        return reset_ui(task_id=task_id, seed=seed)

    if lower == "state":
        chat = chat + [{"role": "assistant", "content": session_status(current_obs)}]
        return chat, format_observation(current_obs), session_status(current_obs), format_reward({}), current_obs

    if lower.startswith("auto"):
        return auto_audit(current_obs, chat)

    if lower.startswith("submit"):
        payload = text.split(":", 1)[1].strip() if ":" in text else text[len("submit"):].strip()
        return submit_report(payload, current_obs, chat)

    if lower.startswith("query"):
        payload = text.split(":", 1)[1].strip() if ":" in text else text[len("query"):].strip()
        return run_query(payload, current_obs, chat)

    if re.search(r"\bselect\b|\bwith\b", lower):
        return run_query(text, current_obs, chat)

    chat = chat + [{"role": "assistant", "content": "I can help with `reset`, `query`, `submit`, `auto`, or `state`."}]
    return chat, format_observation(current_obs), session_status(current_obs), format_reward({}), current_obs


fastapi_app = FastAPI(title="DataQualityEnv Space")


@fastapi_app.get("/health")
def _health() -> dict[str, str]:
    return health()


@fastapi_app.post("/reset")
def _reset(payload: dict = Body(default_factory=dict)) -> dict:
    payload = payload or {}
    payload.setdefault("task_id", 1)
    payload.setdefault("seed", 42)
    return SESSION.reset(payload)


@fastapi_app.post("/step")
def _step(payload: dict = Body(default_factory=dict)) -> dict:
    payload = payload or {}
    return SESSION.step(payload)


@fastapi_app.get("/state")
def _state() -> dict:
    return SESSION.state()


with gr.Blocks(title="DataQualityEnv") as demo:
    gr.Markdown(
        "# DataQualityEnv\n"
        "A self-contained Hugging Face Space demo. No `ENV_URL`, no localhost dependency, no external API hop for the environment."
    )
    with gr.Row():
        with gr.Column(scale=1):
            task_id = gr.Dropdown(choices=[1, 2, 3, 4], value=1, label="Task")
            seed = gr.Number(value=42, precision=0, label="Seed")
            reset_btn = gr.Button("Reset episode", variant="primary")
            auto_btn = gr.Button("Auto audit")
            gr.Markdown("### Session status")
            status_box = gr.Markdown("No active episode. Choose a task and click Reset.")
            reward_box = gr.Textbox(label="Last reward", lines=8, interactive=False)
            obs_box = gr.Textbox(label="Observation JSON", lines=22, interactive=False)
        with gr.Column(scale=2):
            chat = gr.Chatbot(label="Chat", height=520)
            user_text = gr.Textbox(
                label="Command or SQL",
                placeholder="Type reset, query: SELECT ..., submit: {...}, auto, or state",
                lines=3,
            )
            with gr.Row():
                send_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear chat")

    current_obs = gr.State(None)

    reset_btn.click(
        reset_ui,
        inputs=[task_id, seed],
        outputs=[chat, obs_box, status_box, reward_box, current_obs],
    )
    auto_btn.click(
        auto_audit,
        inputs=[current_obs, chat],
        outputs=[chat, obs_box, status_box, reward_box, current_obs],
    )
    send_btn.click(
        handle_command,
        inputs=[user_text, current_obs, chat, task_id, seed],
        outputs=[chat, obs_box, status_box, reward_box, current_obs],
    )
    user_text.submit(
        handle_command,
        inputs=[user_text, current_obs, chat, task_id, seed],
        outputs=[chat, obs_box, status_box, reward_box, current_obs],
    )
    clear_btn.click(lambda: [], inputs=None, outputs=chat)


app = gr.mount_gradio_app(fastapi_app, demo, path="/")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("space_app:app", host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))