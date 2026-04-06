# MichiganCast Project Structure (Engineering Standard)

This project is organized as a production-oriented ML repository, with clear separation of concerns:

```text
.
├── artifacts/
│   ├── figures/
│   │   ├── eval_confusion_matrix.png
│   │   └── train_validation_metrics.png
│   ├── logs/
│   │   └── tensorboard/
│   └── reports/
├── configs/
│   └── experiments/
├── data/
│   ├── external/
│   ├── interim/
│   ├── processed/
│   │   ├── images/
│   │   │   └── lake_michigan_64_png/
│   │   └── tabular/
│   │       └── traverse_city_daytime_meteo_preprocessed.csv
│   ├── raw/
│   └── reference/
│       └── lake_michigan_lat_lon_index.csv
├── docs/
│   ├── architecture/
│   │   └── project_structure.md
│   └── planning/
│       └── michigancast_portfolio_task_list.md
├── models/
│   ├── keras/
│   │   └── rain_multimodal_img6_meteo18_best.h5
│   └── pytorch/
│       ├── rain_multimodal_baseline_best.pth
│       ├── rain_multimodal_img8_meteo24_best.pth
│       └── rain_multimodal_img16_meteo48_best.pth
├── notebooks/
│   ├── 00_util/
│   │   └── read_large_weather_csv_experiments.ipynb
│   ├── 01_eda/
│   │   └── lake_effect_precipitation_exploration.ipynb
│   ├── 02_preprocessing/
│   │   └── lake_michigan_satellite_preprocessing.ipynb
│   └── 03_training/
│       └── multimodal_rainfall_training_pipeline.ipynb
├── reports/
│   └── slides/
│       └── info6106_final_presentation_group3.pptx
├── scripts/
├── src/
│   ├── data/
│   ├── eval/
│   ├── features/
│   ├── models/
│   ├── serve/
│   └── train/
└── tests/
```

## Naming Rules

1. Use snake_case for files and folders.
2. Prefix notebooks by stage (`00_`, `01_`, `02_`, `03_`) to preserve execution flow.
3. Store only immutable or versioned datasets in `data/processed` and `data/reference`.
4. Store all trained weights in `models/` by framework (`pytorch/`, `keras/`).
5. Store generated outputs in `artifacts/` (figures, logs, reports).
6. Keep implementation code in `src/` and tests in `tests/`.

## Local-Only Folders

- `.idea/` and `.DS_Store` are local environment files and should not be treated as project artifacts.
