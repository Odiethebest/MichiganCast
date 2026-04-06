# MichiganCast Project Structure (T30)

The repository follows a production-style ML structure with clear ownership boundaries.

```text
.
├── artifacts/                  # generated outputs (reports, figures, experiment logs, exported models)
│   ├── experiments/
│   ├── figures/
│   ├── models/
│   └── reports/
├── configs/
│   ├── data/
│   │   ├── versioning.yaml
│   │   └── versions/
│   ├── environment/
│   └── experiments/
├── data/
│   ├── raw/
│   ├── external/
│   ├── interim/
│   ├── processed/
│   ├── features/
│   ├── reference/
│   └── README.md
├── docs/
│   ├── architecture/
│   └── planning/
├── models/                     # local model checkpoints (framework-specific)
│   ├── keras/
│   └── pytorch/
├── notebooks/                  # exploratory or legacy notebooks (not production entrypoints)
├── scripts/                    # environment/helper shell scripts
├── src/
│   ├── analysis/
│   ├── data/
│   ├── eval/
│   ├── features/
│   ├── models/
│   │   └── multimodal/
│   ├── serve/
│   └── train/
└── tests/
```

## Ownership Rules

1. Production logic must live under `src/`, not in notebooks.
2. Runtime outputs must go to `artifacts/`, not source folders.
3. Datasets must stay in `data/` with explicit layer semantics.
4. Experiment definitions belong to `configs/experiments/`.
5. Dataset version manifests belong to `configs/data/versions/`.

## T30 Completion Criteria Mapping

- Core logic is in `src/` modules and callable via CLI scripts.
- Notebooks are retained for exploration only.
- Root directory has no business logic scripts outside `src/`/`scripts/`.
