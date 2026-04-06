# MichiganCast: Multimodal Lake-Effect Precipitation Forecasting Pipeline

> A production-oriented portfolio project for leakage-safe forecasting (`t+6h`, `t+24h`) using satellite image sequences and meteorological time-series.

[![Python](https://img.shields.io/badge/Python-3.9-3776AB?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikitlearn)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas)](https://pandas.pydata.org/)
[![Conda Env](https://img.shields.io/badge/Conda-pytorch_env-44A833?logo=anaconda)](https://docs.conda.io/)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Core Capabilities](#core-capabilities)
- [Pipeline Surface](#pipeline-surface)
- [Getting Started](#getting-started)
- [Engineering Depth (Portfolio Signals)](#engineering-depth-portfolio-signals)
- [Repository Structure](#repository-structure)
- [Documentation](#documentation)

---

## Overview

MichiganCast focuses on binary precipitation-event forecasting over the Lake Michigan region with strict anti-leakage rules:

- Inputs are built from historical windows ending at anchor time `t`
- Labels are generated at future timestamp `y_time = t + horizon`
- Forecast horizons currently supported: `6h` and `24h`
- Time-based train/validation/test split is enforced by year range

Current codebase implements:

- Data contract checks and data quality validation
- Reproducible cleaning pipeline and EDA report generation
- Traditional ML baselines (Logistic Regression, Random Forest, Gradient Boosting)
- Modular multimodal PyTorch training stack (`Dataset`, `Model`, `Train Loop`, CLI entry)

---

## Architecture

```text
raw tabular CSV + satellite PNG
            │
            ▼
  src/data/contracts.py     -> schema/range/missing-rule validation
  src/data/validate.py      -> alignment, continuity, image inventory checks
            │
            ▼
  src/data/clean.py         -> reproducible cleaning pipeline
            │
            ▼
  src/features/labeling.py  -> forecast sample index + leakage guards (X_time < y_time)
            │
            ▼
  src/data/split.py         -> strict time-based split by year (train/val/test)
            │
   ┌────────┴───────────┐
   ▼                    ▼
src/models/baselines.py src/models/multimodal/train.py
traditional ML          ConvLSTM + LSTM multimodal network
   │                    │
   └────────┬───────────┘
            ▼
 artifacts/reports + artifacts/figures + models/pytorch
```

---

## Core Capabilities

| Concern | Implementation | Output |
|---|---|---|
| Data contract validation | `src/data/contracts.py` | `artifacts/reports/data_contract_report*.json` |
| Data quality checks | `src/data/validate.py` | `artifacts/data_quality_report.json` |
| Reproducible cleaning | `src/data/clean.py` | `data/interim/*clean*.csv`, cleaning summary JSON |
| Leakage-safe labeling | `src/features/labeling.py` | Forecast sample index with temporal constraints |
| Time-based split | `src/data/split.py` | Year-range split summaries / split CSVs |
| EDA automation | `src/analysis/eda_report.py` | EDA JSON + figures (distribution/correlation/seasonality) |
| Baseline modeling | `src/models/baselines.py` | PR-AUC, F1, Recall, Recall@Precision, Brier, confusion matrices |
| Multimodal training | `src/models/multimodal/*` | Best checkpoint + train summary JSON |

---

## Pipeline Surface

All commands below can be run with the project-bound conda environment wrapper:

`scripts/run_in_pytorch_env.sh <command ...>`

| Stage | Command |
|---|---|
| Data contract | `python -m src.data.contracts --input data/processed/tabular/traverse_city_daytime_meteo_preprocessed.csv` |
| Data quality | `python -m src.data.validate --input data/processed/tabular/traverse_city_daytime_meteo_preprocessed.csv --image-dir data/processed/images/lake_michigan_64_png` |
| Cleaning pipeline | `python -m src.data.clean --input data/processed/tabular/traverse_city_daytime_meteo_preprocessed.csv --output data/interim/traverse_city_daytime_clean_v1.csv` |
| EDA report | `python -m src.analysis.eda_report --input data/interim/traverse_city_daytime_clean_v1.csv --summary artifacts/reports/eda_summary.json --fig-dir artifacts/figures` |
| Baselines | `python -m src.models.baselines --input data/interim/traverse_city_daytime_clean_v1.csv --report artifacts/reports/baseline_results.json --fig-dir artifacts/figures` |
| Multimodal train | `python -m src.models.multimodal.train --input-csv data/interim/traverse_city_daytime_clean_v1.csv --image-dir data/processed/images/lake_michigan_64_png --output-dir artifacts/reports --checkpoint-path models/pytorch/michigancast_multimodal_best.pth` |

---

## Getting Started

### Prerequisites

- Conda available in shell
- Local conda env: `pytorch_env`
- Dataset files under `data/processed/` (tabular + images)

### Environment

```bash
source scripts/activate_pytorch_env.sh
```

Or run directly without activating shell:

```bash
scripts/run_in_pytorch_env.sh python -V
```

Optional Jupyter kernel registration:

```bash
scripts/register_jupyter_kernel.sh
```

### Quick Smoke Run

```bash
scripts/run_in_pytorch_env.sh python -m src.data.clean --nrows 5000
scripts/run_in_pytorch_env.sh python -m src.analysis.eda_report --nrows 5000
scripts/run_in_pytorch_env.sh python -m src.models.baselines --nrows 5000
scripts/run_in_pytorch_env.sh python -m src.models.multimodal.train --nrows 5000 --max-samples-per-split 256 --epochs 2
```

---

## Engineering Depth (Portfolio Signals)

1. Data cleaning + analysis + ML:
`contracts.py` / `validate.py` / `clean.py` / `eda_report.py` / `baselines.py` form a reproducible tabular ML pipeline with explicit quality gates and rare-event metrics.

2. Manual neural-network tuning + local training:
`src/models/multimodal/` decouples dataset construction, architecture, and training loop, enabling controlled local experiments on horizon/lookback/loss-threshold parameters.

3. Data engineering:
The repository uses layered data directories (`raw/interim/processed/reference`), versioned artifacts, environment scripts, and CLI-first pipelines to reduce notebook-only workflow risk.

---

## Repository Structure

```text
.
├── artifacts/      # generated reports/figures
├── configs/        # task + environment + experiment configs
├── data/           # raw / interim / processed / reference
├── docs/           # architecture and planning
├── models/         # trained checkpoints (.pth/.h5)
├── notebooks/      # exploratory and legacy notebook workflows
├── scripts/        # env helpers
├── src/            # production Python modules
└── tests/          # test placeholders (to be expanded)
```

---

## Documentation

- [Project Structure](docs/architecture/project_structure.md)
- [Portfolio Task List](docs/planning/michigancast_portfolio_task_list.md)
- [Task Definition](configs/task.yaml)
