from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.clean import CleaningConfig, run_cleaning_pipeline
from src.models.multimodal.dataset import DatasetBuildConfig, build_multimodal_datasets
from src.models.multimodal.model import MichiganCastMultimodalNet
from src.models.multimodal.train_loop import TrainingConfig, evaluate_loader, fit_multimodal_model


def _parse_year_range(text: str) -> Tuple[int, int]:
    start, end = text.split(":")
    return int(start), int(end)


def _ensure_clean_input_exists(input_csv: str, auto_clean: bool) -> None:
    path = Path(input_csv)
    if path.exists():
        return
    if not auto_clean:
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    print("[train] cleaned input missing, running cleaning pipeline first")
    cfg = CleaningConfig(output_csv=input_csv)
    run_cleaning_pipeline(cfg)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MichiganCast multimodal model (T20)")
    parser.add_argument("--input-csv", default="data/interim/traverse_city_daytime_clean_v1.csv")
    parser.add_argument("--image-dir", default="data/processed/images/lake_michigan_64_png")
    parser.add_argument("--output-dir", default="artifacts/reports")
    parser.add_argument("--checkpoint-path", default="models/pytorch/michigancast_multimodal_best.pth")
    parser.add_argument("--horizon-hours", type=int, default=24)
    parser.add_argument("--meteo-lookback", type=int, default=48)
    parser.add_argument("--image-lookback", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--precision-threshold", type=float, default=0.7)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--train-years", default="2006:2012")
    parser.add_argument("--val-years", default="2013:2014")
    parser.add_argument("--test-years", default="2015:2015")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--no-auto-clean", action="store_true")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    _ensure_clean_input_exists(args.input_csv, auto_clean=not args.no_auto_clean)

    df = pd.read_csv(args.input_csv, nrows=args.nrows, low_memory=False)
    dataset_cfg = DatasetBuildConfig(
        image_dir=args.image_dir,
        horizon_hours=args.horizon_hours,
        meteo_lookback_steps=args.meteo_lookback,
        image_lookback_steps=args.image_lookback,
        image_size=args.image_size,
        max_samples_per_split=args.max_samples_per_split,
        train_years=_parse_year_range(args.train_years),
        val_years=_parse_year_range(args.val_years),
        test_years=_parse_year_range(args.test_years),
    )

    datasets, feature_columns, dataset_meta = build_multimodal_datasets(df, dataset_cfg)
    if len(datasets["train"]) == 0 or len(datasets["val"]) == 0:
        raise ValueError("Empty train/val dataset. Adjust split years or input range.")

    train_loader = DataLoader(
        datasets["train"],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[train] device={device}")

    model = MichiganCastMultimodalNet(
        image_channels=1,
        meteo_feature_count=len(feature_columns),
    ).to(device)

    train_cfg = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        threshold=args.threshold,
        precision_threshold=args.precision_threshold,
        early_stopping_patience=args.patience,
        checkpoint_path=args.checkpoint_path,
        use_scheduler=True,
    )
    fit_result = fit_multimodal_model(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=train_cfg,
    )

    criterion = torch.nn.BCEWithLogitsLoss()
    test_loss, test_metrics = evaluate_loader(
        model,
        test_loader,
        criterion,
        device,
        threshold=args.threshold,
        precision_threshold=args.precision_threshold,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "multimodal_train_summary.json"
    summary = {
        "dataset_config": asdict(dataset_cfg),
        "training_config": asdict(train_cfg),
        "dataset_metadata": dataset_meta,
        "feature_columns": feature_columns,
        "fit_result": fit_result,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"[train] done checkpoint={fit_result['checkpoint_path']} "
        f"val_best_loss={fit_result['best_val_loss']:.4f}"
    )
    print(
        f"[train] test_pr_auc={test_metrics['pr_auc']:.4f} "
        f"test_f1={test_metrics['f1']:.4f} test_recall={test_metrics['recall']:.4f}"
    )
    print(f"[train] summary={summary_path}")


if __name__ == "__main__":
    main()
