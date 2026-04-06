#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${DIR}/.."

exec "${ROOT}/run_high_grade_agent.sh"