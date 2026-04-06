#!/usr/bin/env bash
# Register a dedicated Jupyter kernel backed by pytorch_env.

set -euo pipefail

ENV_NAME="${CONDA_PROJECT_ENV_NAME:-pytorch_env}"
KERNEL_NAME="michigancast-pytorch"
DISPLAY_NAME="Python (MichiganCast pytorch_env)"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda command not found."
  exit 1
fi

conda run -n "${ENV_NAME}" python -m ipykernel install --user \
  --name "${KERNEL_NAME}" \
  --display-name "${DISPLAY_NAME}"

echo "Registered kernel: ${DISPLAY_NAME}"
