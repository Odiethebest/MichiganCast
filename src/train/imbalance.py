from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.clean import CleaningConfig, run_cleaning_pipeline
from src.data.split import TimeSplitConfig, split_samples_by_label_year
from src.eval.metrics import evaluate_binary_predictions
from src.features.labeling import (
    ForecastSamplingConfig,
    attach_utc_timestamp,
    build_forecast_sample_index,
    build_meteorological_feature_columns,
)

DEFAULT_INPUT = "data/interim/traverse_city_daytime_clean_v1.csv"
DEFAULT_OUTPUT_DIR = "artifacts/reports"


@dataclass(frozen=True)
class RunConfig:
    input_csv: str = DEFAULT_INPUT
    output_dir: str = DEFAULT_OUTPUT_DIR
    horizon_hours: int = 24
    meteo_lookback_steps: int = 48
    image_lookback_steps: int = 16
    epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 1e-3
    threshold: float = 0.5
    precision_threshold: float = 0.7
    threshold_moving_precision_floor: float = 0.35
    device: str = "cpu"
    seed: int = 42
    nrows: int | None = None
    auto_clean: bool = True


class TabularRainNet(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.20) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal = -alpha_t * ((1 - pt) ** self.gamma) * torch.log(torch.clamp(pt, min=1e-6, max=1.0))
        return focal.mean()


def _ensure_input_exists(input_csv: str, auto_clean: bool) -> None:
    path = Path(input_csv)
    if path.exists():
        return
    if not auto_clean:
        raise FileNotFoundError(f"Input not found: {input_csv}")
    cfg = CleaningConfig(output_csv=input_csv)
    run_cleaning_pipeline(cfg)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_tabular_samples(
    df: pd.DataFrame,
    *,
    horizon_hours: int,
    meteo_lookback_steps: int,
    image_lookback_steps: int,
) -> tuple[pd.DataFrame, list[str]]:
    sample_cfg = ForecastSamplingConfig(
        horizons_hours=(horizon_hours,),
        meteo_lookback_steps=meteo_lookback_steps,
        image_lookback_steps=image_lookback_steps,
        target_col="target_rain",
    )
    sample_index = build_forecast_sample_index(df, config=sample_cfg)
    feature_cols = [
        c
        for c in build_meteorological_feature_columns(df.columns)
        if pd.api.types.is_numeric_dtype(df[c]) and c != "source_row_id"
    ]

    anchor_features = df.iloc[sample_index["anchor_idx"].to_numpy()][feature_cols].reset_index(drop=True)
    sample_df = pd.concat(
        [
            sample_index[["sample_id", "y_time", "target_rain", "horizon_hours"]].reset_index(drop=True),
            anchor_features,
        ],
        axis=1,
    )
    sample_df[feature_cols] = sample_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    sample_df = sample_df.dropna(subset=feature_cols).reset_index(drop=True)
    return sample_df, feature_cols


def _split_dataset(sample_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    split_cfg = TimeSplitConfig(train_years=(2006, 2012), val_years=(2013, 2014), test_years=(2015, 2015))
    return split_samples_by_label_year(sample_df, config=split_cfg, label_time_col="y_time", verbose=True)


def _standardize(
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (train_x - mean) / std, (val_x - mean) / std, (test_x - mean) / std


def _to_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _tune_threshold_for_recall(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    precision_floor: float,
) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return 0.5

    candidates = []
    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        if p >= precision_floor:
            candidates.append((float(r), float(t)))
    if candidates:
        candidates.sort(reverse=True, key=lambda x: x[0])
        return float(candidates[0][1])

    best_idx = int(np.argmax(recall[:-1])) if len(recall) > 1 else 0
    return float(thresholds[best_idx])


def _train_one_strategy(
    *,
    strategy: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_x: np.ndarray,
    test_y: np.ndarray,
    input_dim: int,
    cfg: RunConfig,
    pos_weight_value: float,
) -> Dict[str, object]:
    device = torch.device(cfg.device)
    _set_seed(cfg.seed)

    model = TabularRainNet(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    if strategy == "weighted_bce":
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=device))
    elif strategy == "focal":
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
    else:
        criterion = nn.BCEWithLogitsLoss()

    for _ in range(cfg.epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_prob_list = []
        val_true_list = []
        for xb, yb in val_loader:
            xb = xb.to(device)
            logits = model(xb)
            prob = torch.sigmoid(logits).cpu().numpy()
            val_prob_list.append(prob)
            val_true_list.append(yb.numpy())
        val_prob = np.concatenate(val_prob_list).astype(float)
        val_true = np.concatenate(val_true_list).astype(int)

        test_tensor = torch.tensor(test_x, dtype=torch.float32, device=device)
        test_prob = torch.sigmoid(model(test_tensor)).cpu().numpy().astype(float)

    decision_threshold = cfg.threshold
    if strategy == "threshold_moving":
        decision_threshold = _tune_threshold_for_recall(
            val_true,
            val_prob,
            precision_floor=cfg.threshold_moving_precision_floor,
        )

    metrics = evaluate_binary_predictions(
        y_true=test_y.astype(int),
        y_prob=test_prob,
        threshold=decision_threshold,
        precision_threshold=cfg.precision_threshold,
    )
    return {
        "strategy": strategy,
        "threshold_used": decision_threshold,
        "metrics": metrics,
    }


def run_imbalance_experiments(cfg: RunConfig) -> Dict[str, object]:
    _ensure_input_exists(cfg.input_csv, auto_clean=cfg.auto_clean)
    _set_seed(cfg.seed)

    df = pd.read_csv(cfg.input_csv, nrows=cfg.nrows, low_memory=False)
    df = attach_utc_timestamp(df)
    sample_df, feature_cols = _build_tabular_samples(
        df,
        horizon_hours=cfg.horizon_hours,
        meteo_lookback_steps=cfg.meteo_lookback_steps,
        image_lookback_steps=cfg.image_lookback_steps,
    )
    splits = _split_dataset(sample_df)

    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("train/val/test split has empty set, cannot run imbalance experiments.")

    train_x = train_df[feature_cols].to_numpy(dtype=float)
    val_x = val_df[feature_cols].to_numpy(dtype=float)
    test_x = test_df[feature_cols].to_numpy(dtype=float)
    train_y = train_df["target_rain"].to_numpy(dtype=float)
    val_y = val_df["target_rain"].to_numpy(dtype=float)
    test_y = test_df["target_rain"].to_numpy(dtype=float)

    train_x, val_x, test_x = _standardize(train_x, val_x, test_x)
    train_loader = _to_loader(train_x, train_y, batch_size=cfg.batch_size, shuffle=True)
    val_loader = _to_loader(val_x, val_y, batch_size=cfg.batch_size, shuffle=False)

    pos_count = float(np.sum(train_y == 1.0))
    neg_count = float(np.sum(train_y == 0.0))
    pos_weight_value = max(neg_count / max(pos_count, 1.0), 1.0)

    strategies = ["baseline_bce", "weighted_bce", "focal", "threshold_moving"]
    strategy_results = []
    for strategy in strategies:
        result = _train_one_strategy(
            strategy=strategy,
            train_loader=train_loader,
            val_loader=val_loader,
            test_x=test_x,
            test_y=test_y,
            input_dim=len(feature_cols),
            cfg=cfg,
            pos_weight_value=pos_weight_value,
        )
        strategy_results.append(result)

    baseline_metrics = next(x for x in strategy_results if x["strategy"] == "baseline_bce")["metrics"]
    rows = []
    for item in strategy_results:
        m = item["metrics"]
        rows.append(
            {
                "strategy": item["strategy"],
                "threshold_used": float(item["threshold_used"]),
                "pr_auc": float(m["pr_auc"]),
                "f1": float(m["f1"]),
                "recall": float(m["recall"]),
                "precision": float(m["precision"]),
                "recall_at_precision": float(m["recall_at_precision"]),
                "brier": float(m["brier"]),
                "recall_delta_vs_baseline": float(m["recall"] - baseline_metrics["recall"]),
                "f1_delta_vs_baseline": float(m["f1"] - baseline_metrics["f1"]),
                "pr_auc_delta_vs_baseline": float(m["pr_auc"] - baseline_metrics["pr_auc"]),
            }
        )

    comparison_df = pd.DataFrame(rows).sort_values(by="recall_delta_vs_baseline", ascending=False)
    improved = comparison_df[comparison_df["recall_delta_vs_baseline"] > 0.05]
    positive_metric_improved = not improved.empty

    summary = {
        "config": cfg.__dict__,
        "split_sizes": {k: int(len(v)) for k, v in splits.items()},
        "positive_rate": {
            "train": float(train_df["target_rain"].mean()),
            "val": float(val_df["target_rain"].mean()),
            "test": float(test_df["target_rain"].mean()),
        },
        "pos_weight_value": float(pos_weight_value),
        "comparison": comparison_df.to_dict(orient="records"),
        "has_significant_positive_metric_improvement": bool(positive_metric_improved),
        "improvement_criterion": "recall_delta_vs_baseline > 0.05",
    }

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "imbalance_strategy_comparison.csv"
    json_path = out_dir / "imbalance_strategy_comparison.json"
    comparison_df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[imbalance] output_csv={csv_path}")
    print(f"[imbalance] output_json={json_path}")
    print(f"[imbalance] significant_improvement={summary['has_significant_positive_metric_improvement']}")
    top = comparison_df.iloc[0].to_dict()
    print(
        f"[imbalance] best_recall_strategy={top['strategy']} "
        f"recall={top['recall']:.4f} delta={top['recall_delta_vs_baseline']:.4f}"
    )
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run class-imbalance strategy experiments (T23)")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--horizon-hours", type=int, default=24)
    parser.add_argument("--meteo-lookback", type=int, default=48)
    parser.add_argument("--image-lookback", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--precision-threshold", type=float, default=0.7)
    parser.add_argument("--threshold-moving-precision-floor", type=float, default=0.35)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--no-auto-clean", action="store_true")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    cfg = RunConfig(
        input_csv=args.input,
        output_dir=args.output_dir,
        horizon_hours=args.horizon_hours,
        meteo_lookback_steps=args.meteo_lookback,
        image_lookback_steps=args.image_lookback,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        threshold=args.threshold,
        precision_threshold=args.precision_threshold,
        threshold_moving_precision_floor=args.threshold_moving_precision_floor,
        device=args.device,
        seed=args.seed,
        nrows=args.nrows,
        auto_clean=not args.no_auto_clean,
    )
    run_imbalance_experiments(cfg)


if __name__ == "__main__":
    main()
