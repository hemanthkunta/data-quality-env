#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://localhost:7860}"
PY_BIN="${PY_BIN:-}"

if [[ -z "${PY_BIN}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PY_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PY_BIN="python"
  else
    echo "No Python interpreter found (python3/python)." >&2
    exit 127
  fi
fi

echo "[1/6] Health check"
curl -fsS "${BASE_URL}/health" | grep -q '"status":"ok"' && echo "  OK"

echo "[2/6] Reset check"
RESET_JSON="$(curl -fsS -X POST "${BASE_URL}/reset" \
  -H 'content-type: application/json' \
  -d '{"task_id":1,"seed":42}')"
echo "${RESET_JSON}" | grep -q '"task_id":1' && echo "  OK"

echo "[3/6] Step query check"
STEP_JSON="$(curl -fsS -X POST "${BASE_URL}/step" \
  -H 'content-type: application/json' \
  -d '{"action":{"action_type":"query","sql":"SELECT COUNT(*) AS n FROM customers"}}')"
echo "${STEP_JSON}" | grep -q '"reward"' && echo "  OK"

echo "[4/6] SQL safety guard check (DROP blocked)"
SAFE_JSON="$(curl -fsS -X POST "${BASE_URL}/step" \
  -H 'content-type: application/json' \
  -d '{"action":{"action_type":"query","sql":"DROP TABLE customers"}}')"
echo "${SAFE_JSON}" | grep -q '"value":-0.2' && echo "  OK"

echo "[5/6] OpenEnv schema validation"
"${PY_BIN}" -m pip install -q openenv-core
if ! command -v openenv >/dev/null 2>&1; then
  echo "openenv CLI not found on PATH after installation" >&2
  exit 1
fi
openenv validate
echo "  OK"

echo "[6/6] Inference dry run (optional, requires model env vars)"
if [[ -n "${API_BASE_URL:-}" && -n "${MODEL_NAME:-}" && ( -n "${HF_TOKEN:-}" || -n "${OPENAI_API_KEY:-}" ) ]]; then
  "${PY_BIN}" inference.py
else
  echo "  Skipped (set API_BASE_URL, MODEL_NAME, HF_TOKEN/OPENAI_API_KEY to run)"
fi

echo "All checks completed."
