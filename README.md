# MichiganCast

`MichiganCast: A Production-Grade Multimodal Lake-Effect Precipitation Forecasting Pipeline`

Engineering-style project layout for data cleaning, analysis, model training, and deployment.

## Conda Environment

This project is bound to `pytorch_env` (`/opt/anaconda3/envs/pytorch_env`).

- Activate in current shell: `source scripts/activate_pytorch_env.sh`
- Run any command in env: `scripts/run_in_pytorch_env.sh <command ...>`
- Register notebook kernel once: `scripts/register_jupyter_kernel.sh`

Environment reproducibility file:
- `configs/environment/conda-pytorch_env.from-history.yml`

## Main Folders

- `data/`: reference and processed datasets
- `notebooks/`: staged exploratory and legacy training notebooks
- `src/`: production code modules (data/features/train/eval/serve)
- `models/`: trained model weights by framework
- `artifacts/`: generated outputs (figures, logs)
- `configs/`: experiment configuration files
- `tests/`: automated tests
- `docs/`: architecture and planning documents
- `reports/`: presentation and report outputs

For the full structure and naming rules, see:
- `docs/architecture/project_structure.md`
