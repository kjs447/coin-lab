#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
UVICORN_ARGS=("app.main:app" "--host" "0.0.0.0" "--port" "8000" "--reload" "--app-dir" "${ROOT_DIR}")

if [[ ! -d "${VENV_DIR}" ]]; then
    echo "Virtual environment not found at ${VENV_DIR}" >&2
    echo "Run scripts/setup_env.sh first." >&2
    exit 1
fi

source "${VENV_DIR}/bin/activate"
exec uvicorn "${UVICORN_ARGS[@]}"
