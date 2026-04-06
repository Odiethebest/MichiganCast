from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.clean import CleaningConfig, run_cleaning_pipeline
from src.models.multimodal.dataset import DatasetBuildConfig, build_multimodal_datasets
from src.models.multimodal.model import MichiganCastMultimodalNet
from src.models.multimodal.train_loop import TrainingConfig, evaluate_loader, fit_multimodal_model


@dataclass(frozen=True)
class StabilityConfig:
    input_csv: str = "data/interim/traverse_city_daytime_clean_v1.csv"
    image_dir: str = "data/processed/images/lake_michigan_64_png"
    output_json: str = "artifacts/reports/stability_train_report.json"
    checkpoint_prefix: str = "models/pytorch/michigancast_stability"
    horizon_hours: int = 24
    meteo_lookback_steps: int = 48
    image_lookback_steps: int = 16
    image_size: int = 64
    batch_size: int = 64
    epochs: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    threshold: float = 0.5
    precision_threshold: float = 0.7
    patience: int = 4
    conv_hidden_dim: int = 64
    meteo_hidden_dim: int = 128
    fusion_hidden_dim: int = 64
    dropout: float = 0.50
    device: str = "auto"
    seed: int = 42
    stability_runs: int = 2
    acceptable_delta: float = 0.03
    nrows: int | None = None
    max_samples_per_split: int | None = None
    auto_clean: bool = True


def _ensure_input_exists(input_csv: str, auto_clean: bool) -> None:
    path = Path(input_csv)
    if path.exists():
        return
    if not auto_clean:
        raise FileNotFoundError(f"Input not found: {input_csv}")
    cfg = CleaningConfig(output_csv=input_csv)
    run_cleaning_pipeline(cfg)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if device == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS is unavailable.")
        return torch.device("mps")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable.")
        return torch.device("cuda")
    return torch.device("cpu")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _build_loaders(datasets: Dict[str, torch.utils.data.Dataset], batch_size: int, seed: int) -> Dict[str, DataLoader]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, num_workers=0, generator=generator),
        "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, num_workers=0),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, num_workers=0),
    }


def _single_run(cfg: StabilityConfig, run_idx: int, device: torch.device) -> Dict[str, object]:
    _set_seed(cfg.seed)

    df = pd.read_csv(cfg.input_csv, nrows=cfg.nrows, low_memory=False)
    dataset_cfg = DatasetBuildConfig(
        image_dir=cfg.image_dir,
        horizon_hours=cfg.horizon_hours,
        meteo_lookback_steps=cfg.meteo_lookback_steps,
        image_lookback_steps=cfg.image_lookback_steps,
        image_size=cfg.image_size,
        max_samples_per_split=cfg.max_samples_per_split,
        cache_images_in_memory=False,
        train_years=(2006, 2012),
        val_years=(2013, 2014),
        test_years=(2015, 2015),
    )
    datasets, feature_columns, dataset_meta = build_multimodal_datasets(df, dataset_cfg)
    if len(datasets["train"]) == 0 or len(datasets["val"]) == 0:
        raise ValueError("Empty train/val dataset.")

    loaders = _build_loaders(datasets, cfg.batch_size, cfg.seed)
    model = MichiganCastMultimodalNet(
        image_channels=1,
        meteo_feature_count=len(feature_columns),
        conv_hidden_dim=cfg.conv_hidden_dim,
        meteo_hidden_dim=cfg.meteo_hidden_dim,
        fusion_hidden_dim=cfg.fusion_hidden_dim,
        dropout=cfg.dropout,
    ).to(device)

    checkpoint_path = f"{cfg.checkpoint_prefix}_run{run_idx}.pth"
    train_cfg = TrainingConfig(
        epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        threshold=cfg.threshold,
        precision_threshold=cfg.precision_threshold,
        early_stopping_patience=cfg.patience,
        checkpoint_path=checkpoint_path,
        use_scheduler=True,
    )
    fit_result = fit_multimodal_model(
        model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        device=device,
        config=train_cfg,
    )

    criterion = torch.nn.BCEWithLogitsLoss()
    test_loss, test_metrics = evaluate_loader(
        model,
        loaders["test"],
        criterion,
        device,
        threshold=cfg.threshold,
        precision_threshold=cfg.precision_threshold,
    )

    history = fit_result.get("history", {})
    result = {
        "run_index": run_idx,
        "seed": cfg.seed,
        "checkpoint_path": checkpoint_path,
        "epochs_ran": fit_result.get("epochs_ran"),
        "best_val_loss": fit_result.get("best_val_loss"),
        "final_val_pr_auc": float(history.get("val_pr_auc", [0.0])[-1]) if history.get("val_pr_auc") else 0.0,
        "final_val_f1": float(history.get("val_f1", [0.0])[-1]) if history.get("val_f1") else 0.0,
        "test_loss": float(test_loss),
        "test_pr_auc": float(test_metrics["pr_auc"]),
        "test_f1": float(test_metrics["f1"]),
        "test_recall": float(test_metrics["recall"]),
        "dataset_metadata": dataset_meta,
    }
    return result


def run_stability_training(cfg: StabilityConfig) -> Dict[str, object]:
    _ensure_input_exists(cfg.input_csv, auto_clean=cfg.auto_clean)
    device = _resolve_device(cfg.device)
    if device.type in {"cuda", "mps"}:
        torch.set_float32_matmul_precision("high")

    runs = []
    for i in range(1, cfg.stability_runs + 1):
        print(f"[stability] run={i}/{cfg.stability_runs} seed={cfg.seed}")
        runs.append(_single_run(cfg, run_idx=i, device=device))

    keys = ["best_val_loss", "final_val_pr_auc", "final_val_f1", "test_pr_auc", "test_f1", "test_recall"]
    deltas = {}
    for key in keys:
        values = [float(r[key]) for r in runs]
        deltas[key] = float(max(values) - min(values))

    max_delta = max(deltas.values()) if deltas else 0.0
    is_stable = max_delta <= cfg.acceptable_delta

    report = {
        "config": asdict(cfg),
        "device": str(device),
        "runs": runs,
        "metric_deltas": deltas,
        "max_delta": float(max_delta),
        "acceptable_delta": float(cfg.acceptable_delta),
        "is_stable": bool(is_stable),
    }

    out = Path(cfg.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[stability] output={out}")
    print(f"[stability] is_stable={is_stable} max_delta={max_delta:.6f}")
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stability-focused multimodal training (T24)")
    parser.add_argument("--input-csv", default="data/interim/traverse_city_daytime_clean_v1.csv")
    parser.add_argument("--image-dir", default="data/processed/images/lake_michigan_64_png")
    parser.add_argument("--output-json", default="artifacts/reports/stability_train_report.json")
    parser.add_argument("--checkpoint-prefix", default="models/pytorch/michigancast_stability")
    parser.add_argument("--horizon-hours", type=int, default=24)
    parser.add_argument("--meteo-lookback", type=int, default=48)
    parser.add_argument("--image-lookback", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--precision-threshold", type=float, default=0.7)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--conv-hidden-dim", type=int, default=64)
    parser.add_argument("--meteo-hidden-dim", type=int, default=128)
    parser.add_argument("--fusion-hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stability-runs", type=int, default=2)
    parser.add_argument("--acceptable-delta", type=float, default=0.03)
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    parser.add_argument("--no-auto-clean", action="store_true")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    cfg = StabilityConfig(
        input_csv=args.input_csv,
        image_dir=args.image_dir,
        output_json=args.output_json,
        checkpoint_prefix=args.checkpoint_prefix,
        horizon_hours=args.horizon_hours,
        meteo_lookback_steps=args.meteo_lookback,
        image_lookback_steps=args.image_lookback,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        threshold=args.threshold,
        precision_threshold=args.precision_threshold,
        patience=args.patience,
        conv_hidden_dim=args.conv_hidden_dim,
        meteo_hidden_dim=args.meteo_hidden_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        dropout=args.dropout,
        device=args.device,
        seed=args.seed,
        stability_runs=args.stability_runs,
        acceptable_delta=args.acceptable_delta,
        nrows=args.nrows,
        max_samples_per_split=args.max_samples_per_split,
        auto_clean=not args.no_auto_clean,
    )
    run_stability_training(cfg)


if __name__ == "__main__":
    main()
