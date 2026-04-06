from __future__ import annotations

import threading
from typing import Any

from fastapi import FastAPI, HTTPException

from env.dataset_gen import generate_dataset
from env.engine import SQLEngine
from env.models import Action, EpisodeState, Observation, Reward, RewardBreakdown
from tasks.task1_nulls import Task1
from tasks.task2_schema import Task2
from tasks.task3_drift import Task3
from tasks.task4_relational import Task4

app = FastAPI(title="DataQualityEnv")

_lock = threading.Lock()

TASKS = {1: Task1(), 2: Task2(), 3: Task3(), 4: Task4()}
MAX_STEPS = 12
FIX_STEPS = 3

state: EpisodeState | None = None
engine: SQLEngine | None = None
gold: dict[str, Any] = {}
table_names: list[str] = []


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "env": "DataQualityEnv", "version": "2.0.0"}


@app.post("/reset")
def reset(payload: dict):
    global state, engine, gold, table_names
    task_id = int(payload.get("task_id", 1))
    seed = int(payload.get("seed", 42))
    if task_id not in TASKS:
        raise HTTPException(400, f"task_id must be 1-4, got {task_id}")

    with _lock:
        if engine:
            engine.close()
        engine = SQLEngine()
        tables, gold = generate_dataset(task_id, seed)
        engine.load_tables(tables)
        table_names = list(tables.keys())

        state = EpisodeState(task_id=task_id, seed=seed, gold_faults=gold, max_steps=MAX_STEPS, fix_steps_remaining=FIX_STEPS)

        task = TASKS[task_id]
        obs = _make_observation(task, state, engine, table_names, None, None, None)
        return obs.model_dump()


@app.post("/step")
def step(payload: dict):
    global state
    if state is None or state.done:
        raise HTTPException(400, "Call /reset first.")

    try:
        action = Action(**payload.get("action", payload))
    except Exception as e:
        raise HTTPException(400, f"Invalid action: {e}")

    task = TASKS[state.task_id]
    assert engine is not None

    with _lock:
        state.step += 1

        if state.step > MAX_STEPS:
            state.done = True
            total = round(min(1.25, state.audit_score + state.fix_bonus), 4)
            rb = RewardBreakdown(
                base_audit_score=state.audit_score,
                confidence_brier_adjustment=0.0,
                budget_efficiency_bonus=0.0,
                fix_verification_bonus=round(state.fix_bonus, 4),
                total=total,
            )
            obs = _make_observation(task, state, engine, table_names, None, "max_steps", None)
            return _step_response(obs, Reward(value=total, breakdown=rb, done=True, info={"reason": "max_steps"}))

        if action.action_type == "query":
            if state.phase == "fix":
                obs = _make_observation(task, state, engine, table_names, None, "Use fix_sql action in fix phase, not query.", None)
                reward = Reward(value=0.0, breakdown=_zero_breakdown(), done=False, info={})
                return _step_response(obs, reward)
            if state.query_credits <= 0:
                obs = _make_observation(task, state, engine, table_names, None, "No query credits remaining.", None)
                reward = Reward(value=0.0, breakdown=_zero_breakdown(), done=False, info={})
                return _step_response(obs, reward)
            if not action.sql:
                raise HTTPException(400, "sql is required for query action")

            result = engine.execute(action.sql)
            if isinstance(result, str) and result.startswith("ERROR"):
                obs = _make_observation(task, state, engine, table_names, None, result, None)
                reward = Reward(value=-0.1, breakdown=_zero_breakdown(destructive=-0.1), done=False, info={"error": result})
            else:
                state.query_credits -= 1
                obs = _make_observation(task, state, engine, table_names, result if isinstance(result, list) else None, None, None)
                reward = Reward(value=0.0, breakdown=_zero_breakdown(), done=False, info={})
            return _step_response(obs, reward)

        if action.action_type == "submit_report":
            if action.report is None:
                raise HTTPException(400, "report is required for submit_report")
            if state.report_submitted:
                raise HTTPException(400, "Report already submitted. Use fix_sql or reset.")

            base_score, score_breakdown = task.grade(action.report, gold)
            budget_bonus = round(min(0.10, state.query_credits * 0.01), 4)
            total = round(min(1.0, base_score + budget_bonus), 4)

            state.audit_score = total
            state.report_submitted = True
            state.phase = "fix"

            rb = RewardBreakdown(
                base_audit_score=float(base_score),
                confidence_brier_adjustment=0.0,
                budget_efficiency_bonus=budget_bonus,
                fix_verification_bonus=0.0,
                total=total,
            )
            done = state.fix_steps_remaining == 0
            if done:
                state.done = True

            obs = _make_observation(task, state, engine, table_names, None, None, None)
            return _step_response(obs, Reward(value=total, breakdown=rb, done=done, info={"score_breakdown": score_breakdown, "fix_steps_available": FIX_STEPS}))

        if action.action_type == "fix_sql":
            if not state.report_submitted:
                raise HTTPException(400, "Submit report before using fix_sql.")
            if not action.sql:
                raise HTTPException(400, "sql is required for fix_sql")

            if state.fix_steps_remaining <= 0:
                state.done = True
                total = round(min(1.25, state.audit_score + state.fix_bonus), 4)
                rb = RewardBreakdown(
                    base_audit_score=state.audit_score,
                    confidence_brier_adjustment=0.0,
                    budget_efficiency_bonus=0.0,
                    fix_verification_bonus=round(state.fix_bonus, 4),
                    total=total,
                )
                obs = _make_observation(task, state, engine, table_names, None, None, 0.0)
                return _step_response(obs, Reward(value=total, breakdown=rb, done=True, info={}))

            fix_score = engine.run_fix_sql(action.sql, gold)
            state.fix_bonus = min(0.25, state.fix_bonus + fix_score * 0.08)
            state.fix_steps_remaining -= 1
            done = state.fix_steps_remaining == 0
            if done:
                state.done = True

            total = round(min(1.25, state.audit_score + state.fix_bonus), 4)
            rb = RewardBreakdown(
                base_audit_score=state.audit_score,
                confidence_brier_adjustment=0.0,
                budget_efficiency_bonus=0.0,
                fix_verification_bonus=round(state.fix_bonus, 4),
                total=total,
            )
            obs = _make_observation(task, state, engine, table_names, None, None, fix_score)
            return _step_response(obs, Reward(value=total, breakdown=rb, done=done, info={}))

        raise HTTPException(400, f"Unsupported action_type: {action.action_type}")


@app.get("/state")
def get_state():
    if state is None:
        raise HTTPException(400, "No active episode.")
    return state.model_dump()


def _make_observation(task, st: EpisodeState, eng: SQLEngine, tables: list[str], query_result, error, last_fix_score) -> Observation:
    schemas = eng.get_table_schemas(tables)
    row_counts = eng.get_row_counts(tables)
    trimmed = query_result[:50] if isinstance(query_result, list) else None
    return Observation(
        task_id=st.task_id,
        task_description=task.get_description(),
        tables=schemas,
        row_counts=row_counts,
        step=st.step,
        max_steps=MAX_STEPS,
        query_credits_remaining=st.query_credits,
        phase=st.phase,
        last_query_result=trimmed,
        last_action_error=error,
        last_fix_score=last_fix_score,
    )


def _step_response(obs: Observation, reward: Reward) -> dict[str, Any]:
    return {"observation": obs.model_dump(), "reward": reward.model_dump()}


def _zero_breakdown(destructive: float = 0.0) -> RewardBreakdown:
    return RewardBreakdown(
        base_audit_score=0.0,
        confidence_brier_adjustment=0.0,
        budget_efficiency_bonus=0.0,
        fix_verification_bonus=destructive,
        total=destructive,
    )
