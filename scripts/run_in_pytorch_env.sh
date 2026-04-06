#!/usr/bin/env bash
# Usage:
#   scripts/run_in_pytorch_env.sh python -V
#   scripts/run_in_pytorch_env.sh pytest -q

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${CONDA_PROJECT_ENV_NAME:-pytorch_env}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda command not found."
  exit 1
fi

if [[ $# -eq 0 ]]; then
  echo "No command provided."
  echo "Example: scripts/run_in_pytorch_env.sh python -V"
  exit 1
fi

conda run -n "${ENV_NAME}" --cwd "${PROJECT_ROOT}" "$@"
