from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

from src.data.clean import CleaningConfig, run_cleaning_pipeline
from src.data.split import TimeSplitConfig, split_samples_by_label_year
from src.features.labeling import (
    ForecastSamplingConfig,
    attach_utc_timestamp,
    build_forecast_sample_index,
    build_meteorological_feature_columns,
)

DEFAULT_INPUT = "data/interim/traverse_city_daytime_clean_v1.csv"
DEFAULT_REPORT = "artifacts/reports/baseline_results.json"
DEFAULT_FIG_DIR = "artifacts/figures"


def _ensure_input_exists(input_csv: str, auto_clean: bool) -> None:
    path = Path(input_csv)
    if path.exists():
        return
    if not auto_clean:
        raise FileNotFoundError(f"Input not found: {input_csv}")
    cfg = CleaningConfig(output_csv=input_csv)
    run_cleaning_pipeline(cfg)


def _evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, object]:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "confusion_matrix": cm.tolist(),
    }


def _save_confusion_matrix_plot(cm: np.ndarray, title: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=["No Rain", "Rain"],
        yticklabels=["No Rain", "Rain"],
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    thresh = cm.max() / 2.0 if cm.size else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _build_sample_dataset(
    df: pd.DataFrame,
    *,
    horizon_hours: int,
    meteo_lookback_steps: int,
    image_lookback_steps: int,
    target_col: str = "target_rain",
) -> Tuple[pd.DataFrame, list[str]]:
    sample_cfg = ForecastSamplingConfig(
        horizons_hours=(horizon_hours,),
        meteo_lookback_steps=meteo_lookback_steps,
        image_lookback_steps=image_lookback_steps,
        target_col=target_col,
    )
    sample_index = build_forecast_sample_index(df, config=sample_cfg)
    candidate_feature_cols = build_meteorological_feature_columns(df.columns)
    feature_cols = [
        c
        for c in candidate_feature_cols
        if c not in {"source_row_id"} and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not feature_cols:
        raise ValueError("No numeric feature columns available for baseline models.")
    anchor_features = df.iloc[sample_index["anchor_idx"].to_numpy()][feature_cols].reset_index(drop=True)
    sample_dataset = pd.concat(
        [
            sample_index[["sample_id", "y_time", target_col, "horizon_hours"]].reset_index(drop=True),
            anchor_features,
        ],
        axis=1,
    )
    sample_dataset[feature_cols] = sample_dataset[feature_cols].replace([np.inf, -np.inf], np.nan)
    return sample_dataset, feature_cols


def train_and_evaluate_baselines(
    *,
    input_csv: str = DEFAULT_INPUT,
    report_json: str = DEFAULT_REPORT,
    fig_dir: str = DEFAULT_FIG_DIR,
    horizon_hours: int = 24,
    meteo_lookback_steps: int = 48,
    image_lookback_steps: int = 16,
    threshold: float = 0.5,
    auto_clean: bool = True,
    nrows: int | None = None,
) -> Dict[str, object]:
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.utils.extmath")
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.linear_model._linear_loss")

    _ensure_input_exists(input_csv, auto_clean=auto_clean)
    df = pd.read_csv(input_csv, nrows=nrows, low_memory=False)
    df = attach_utc_timestamp(df)

    sample_dataset, feature_cols = _build_sample_dataset(
        df,
        horizon_hours=horizon_hours,
        meteo_lookback_steps=meteo_lookback_steps,
        image_lookback_steps=image_lookback_steps,
    )

    split_cfg = TimeSplitConfig(train_years=(2006, 2012), val_years=(2013, 2014), test_years=(2015, 2015))
    split_frames = split_samples_by_label_year(
        sample_dataset.rename(columns={"target_rain": "target_rain", "y_time": "y_time"}),
        config=split_cfg,
        label_time_col="y_time",
        verbose=True,
    )

    train_df = split_frames["train"]
    val_df = split_frames["val"]
    test_df = split_frames["test"]
    if train_df.empty or val_df.empty:
        raise ValueError("Train/validation split is empty. Check input range and split years.")

    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df["target_rain"].astype(int).to_numpy()
    X_val = val_df[feature_cols].to_numpy(dtype=float)
    y_val = val_df["target_rain"].astype(int).to_numpy()
    X_test = (
        test_df[feature_cols].to_numpy(dtype=float)
        if not test_df.empty
        else np.empty((0, len(feature_cols)), dtype=float)
    )
    y_test = test_df["target_rain"].astype(int).to_numpy() if not test_df.empty else np.empty((0,), dtype=int)

    models = {
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("variance", VarianceThreshold(threshold=0.0)),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=12,
                        class_weight="balanced_subsample",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", GradientBoostingClassifier(random_state=42)),
            ]
        ),
    }

    fig_root = Path(fig_dir)
    fig_root.mkdir(parents=True, exist_ok=True)
    results: Dict[str, object] = {
        "input_csv": input_csv,
        "horizon_hours": horizon_hours,
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "split_sizes": {"train": int(len(train_df)), "val": int(len(val_df)), "test": int(len(test_df))},
        "models": {},
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        val_prob = model.predict_proba(X_val)[:, 1]
        val_metrics = _evaluate_predictions(y_val, val_prob, threshold=threshold)
        cm_val = np.array(val_metrics["confusion_matrix"], dtype=int)
        val_cm_path = fig_root / f"baseline_cm_{name}_val.png"
        _save_confusion_matrix_plot(cm_val, f"{name} (val)", val_cm_path)

        model_result = {
            "validation": {**val_metrics, "confusion_matrix_figure": str(val_cm_path)},
        }

        if len(test_df) > 0:
            test_prob = model.predict_proba(X_test)[:, 1]
            test_metrics = _evaluate_predictions(y_test, test_prob, threshold=threshold)
            cm_test = np.array(test_metrics["confusion_matrix"], dtype=int)
            test_cm_path = fig_root / f"baseline_cm_{name}_test.png"
            _save_confusion_matrix_plot(cm_test, f"{name} (test)", test_cm_path)
            model_result["test"] = {**test_metrics, "confusion_matrix_figure": str(test_cm_path)}

        results["models"][name] = model_result

    out_json = Path(report_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MichiganCast traditional ML baselines")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input cleaned CSV")
    parser.add_argument("--report", default=DEFAULT_REPORT, help="Output JSON results path")
    parser.add_argument("--fig-dir", default=DEFAULT_FIG_DIR, help="Confusion matrix output directory")
    parser.add_argument("--horizon-hours", type=int, default=24, help="Forecast horizon in hours")
    parser.add_argument("--meteo-lookback", type=int, default=48, help="Meteorological lookback steps")
    parser.add_argument("--image-lookback", type=int, default=16, help="Image lookback steps")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
    parser.add_argument("--nrows", type=int, default=None, help="Optional row limit for quick runs")
    parser.add_argument("--no-auto-clean", action="store_true", help="Disable auto clean fallback")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    results = train_and_evaluate_baselines(
        input_csv=args.input,
        report_json=args.report,
        fig_dir=args.fig_dir,
        horizon_hours=args.horizon_hours,
        meteo_lookback_steps=args.meteo_lookback,
        image_lookback_steps=args.image_lookback,
        threshold=args.threshold,
        auto_clean=not args.no_auto_clean,
        nrows=args.nrows,
    )
    print(
        f"[baselines] horizon={results['horizon_hours']} "
        f"train={results['split_sizes']['train']} val={results['split_sizes']['val']} "
        f"test={results['split_sizes']['test']} report={args.report}"
    )
    for name, result in results["models"].items():
        val = result["validation"]
        print(
            f"[baselines] {name}: val_pr_auc={val['pr_auc']:.4f} "
            f"val_f1={val['f1']:.4f} val_recall={val['recall']:.4f}"
        )


if __name__ == "__main__":
    main()
