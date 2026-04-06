#!/usr/bin/env bash
# Usage (must be sourced):
#   source scripts/activate_pytorch_env.sh

set -euo pipefail

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Please source this script:"
  echo "  source scripts/activate_pytorch_env.sh"
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${CONDA_PROJECT_ENV_NAME:-pytorch_env}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda command not found."
  return 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"

conda activate "${ENV_NAME}"
cd "${PROJECT_ROOT}"
export PROJECT_ROOT

echo "Activated conda env: ${ENV_NAME}"
echo "Project root: ${PROJECT_ROOT}"
