#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="${PYTHON:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Error: Python interpreter '${PYTHON_BIN}' not found. Set PYTHON env var if needed." >&2
    exit 1
fi

if [[ ! -d "${VENV_DIR}" || ! -f "${VENV_DIR}/bin/activate" ]]; then
    if [[ -d "${VENV_DIR}" ]]; then
        echo "Existing virtual environment at ${VENV_DIR} is incomplete. Recreating..."
        rm -rf "${VENV_DIR}"
    else
        echo "Creating virtual environment at ${VENV_DIR}"
    fi
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
else
    echo "Reusing existing virtual environment at ${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "${ROOT_DIR}/requirements.txt"

echo "Virtual environment is ready. Activate with: source ${VENV_DIR}/bin/activate"
