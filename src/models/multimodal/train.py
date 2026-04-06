from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.clean import CleaningConfig, run_cleaning_pipeline
from src.models.multimodal.dataset import DatasetBuildConfig, build_multimodal_datasets
from src.models.multimodal.model import MichiganCastMultimodalNet
from src.models.multimodal.train_loop import TrainingConfig, evaluate_loader, fit_multimodal_model


@dataclass(frozen=True)
class HardwareProfile:
    platform: str
    machine: str
    cpu_count: int
    total_memory_gb: float
    is_apple_silicon: bool


@dataclass(frozen=True)
class RuntimeProfile:
    device: str
    batch_size: int
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int | None
    cache_images_in_memory: bool


def _parse_year_range(text: str) -> Tuple[int, int]:
    start, end = text.split(":")
    return int(start), int(end)


def _detect_total_memory_gb() -> float:
    if sys.platform == "darwin":
        try:
            raw = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            return round(int(raw) / (1024 ** 3), 2)
        except Exception:
            pass

    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return round((pages * page_size) / (1024 ** 3), 2)
    except (ValueError, OSError, AttributeError):
        pass
    return 0.0


def _build_hardware_profile() -> HardwareProfile:
    machine = platform.machine().lower()
    return HardwareProfile(
        platform=sys.platform,
        machine=machine,
        cpu_count=max(os.cpu_count() or 1, 1),
        total_memory_gb=_detect_total_memory_gb(),
        is_apple_silicon=(sys.platform == "darwin" and machine in {"arm64", "aarch64"}),
    )


def _mps_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if _mps_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if requested == "mps":
        if not _mps_available():
            raise RuntimeError("Requested --device mps but MPS is not available in this PyTorch runtime.")
        return torch.device("mps")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda but CUDA is not available.")
        return torch.device("cuda")

    return torch.device("cpu")


def _auto_batch_size_for_mps(total_memory_gb: float) -> int:
    if total_memory_gb >= 40.0:
        return 64
    if total_memory_gb >= 24.0:
        return 48
    return 32


def _resolve_runtime_profile(
    args: argparse.Namespace,
    *,
    device: torch.device,
    hardware: HardwareProfile,
) -> tuple[RuntimeProfile, list[str]]:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.prefetch_factor is not None and args.prefetch_factor <= 0:
        raise ValueError("--prefetch-factor must be positive when provided.")
    if args.num_workers < -1:
        raise ValueError("--num-workers must be >= -1 (-1 means auto).")

    notes: list[str] = []
    batch_size = args.batch_size
    num_workers = args.num_workers
    prefetch_factor = args.prefetch_factor
    cache_images = args.cache_images == "on"

    if args.cache_images == "auto":
        cache_images = False

    if args.apple_metal_opt and device.type == "mps" and hardware.is_apple_silicon:
        if args.cache_images == "auto" and hardware.total_memory_gb >= 40.0:
            cache_images = True
            notes.append("Enabled in-memory image cache (Apple Silicon + large unified memory).")

        if args.batch_size == 32:
            tuned_batch_size = _auto_batch_size_for_mps(hardware.total_memory_gb)
            batch_size = tuned_batch_size
            if tuned_batch_size != 32:
                notes.append(f"Auto-raised batch size from 32 to {tuned_batch_size} for MPS.")

        if num_workers == -1:
            # When image caching is enabled, worker processes duplicate cache memory.
            num_workers = 0 if cache_images else min(max(hardware.cpu_count // 2, 2), 8)
        if prefetch_factor is None and num_workers > 0:
            prefetch_factor = 4
    else:
        if num_workers == -1:
            num_workers = min(max(hardware.cpu_count // 2, 2), 8)
        if prefetch_factor is None and num_workers > 0:
            prefetch_factor = 2

    if num_workers <= 0:
        num_workers = 0
        persistent_workers = False
        prefetch_factor = None
    else:
        persistent_workers = not args.no_persistent_workers

    runtime = RuntimeProfile(
        device=device.type,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        cache_images_in_memory=cache_images,
    )
    return runtime, notes


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
    parser.add_argument("--num-workers", type=int, default=-1, help="-1 for auto worker tuning")
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument(
        "--cache-images",
        choices=["auto", "on", "off"],
        default="auto",
        help="In-memory image caching strategy for sequence loading",
    )
    parser.add_argument(
        "--max-cached-images-per-split",
        type=int,
        default=None,
        help="Optional limit for in-memory cached image count per split",
    )
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--train-years", default="2006:2012")
    parser.add_argument("--val-years", default="2013:2014")
    parser.add_argument("--test-years", default="2015:2015")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--apple-metal-opt", dest="apple_metal_opt", action="store_true")
    parser.add_argument("--no-apple-metal-opt", dest="apple_metal_opt", action="store_false")
    parser.set_defaults(apple_metal_opt=True)
    parser.add_argument("--no-auto-clean", action="store_true")
    return parser


def _build_loader(dataset, *, runtime: RuntimeProfile, shuffle: bool) -> DataLoader:
    kwargs = {
        "batch_size": runtime.batch_size,
        "shuffle": shuffle,
        "num_workers": runtime.num_workers,
        "pin_memory": runtime.pin_memory,
    }
    if runtime.num_workers > 0:
        kwargs["persistent_workers"] = runtime.persistent_workers
        if runtime.prefetch_factor is not None:
            kwargs["prefetch_factor"] = runtime.prefetch_factor
    return DataLoader(dataset, **kwargs)


def main() -> None:
    args = _build_arg_parser().parse_args()
    _ensure_clean_input_exists(args.input_csv, auto_clean=not args.no_auto_clean)

    hardware = _build_hardware_profile()
    device = _resolve_device(args.device)
    runtime, runtime_notes = _resolve_runtime_profile(args, device=device, hardware=hardware)

    if device.type in {"cuda", "mps"}:
        torch.set_float32_matmul_precision("high")

    print(
        f"[train] device={device.type} platform={hardware.platform}/{hardware.machine} "
        f"cpu={hardware.cpu_count} mem_gb={hardware.total_memory_gb:.2f}"
    )
    print(
        f"[train] runtime batch_size={runtime.batch_size} workers={runtime.num_workers} "
        f"pin_memory={runtime.pin_memory} prefetch={runtime.prefetch_factor} "
        f"cache_images={runtime.cache_images_in_memory}"
    )
    for note in runtime_notes:
        print(f"[train] runtime_note={note}")

    df = pd.read_csv(args.input_csv, nrows=args.nrows, low_memory=False)
    dataset_cfg = DatasetBuildConfig(
        image_dir=args.image_dir,
        horizon_hours=args.horizon_hours,
        meteo_lookback_steps=args.meteo_lookback,
        image_lookback_steps=args.image_lookback,
        image_size=args.image_size,
        max_samples_per_split=args.max_samples_per_split,
        cache_images_in_memory=runtime.cache_images_in_memory,
        max_cached_images_per_split=args.max_cached_images_per_split,
        train_years=_parse_year_range(args.train_years),
        val_years=_parse_year_range(args.val_years),
        test_years=_parse_year_range(args.test_years),
    )

    datasets, feature_columns, dataset_meta = build_multimodal_datasets(df, dataset_cfg)
    if len(datasets["train"]) == 0 or len(datasets["val"]) == 0:
        raise ValueError("Empty train/val dataset. Adjust split years or input range.")

    train_loader = _build_loader(datasets["train"], runtime=runtime, shuffle=True)
    val_loader = _build_loader(datasets["val"], runtime=runtime, shuffle=False)
    test_loader = _build_loader(datasets["test"], runtime=runtime, shuffle=False)

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
        "hardware_profile": asdict(hardware),
        "runtime_profile": asdict(runtime),
        "runtime_notes": runtime_notes,
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
