from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict

from src.analysis.eda_report import build_eda_report
from src.data.build_parquet import ParquetBuildConfig, build_parquet_assets
from src.data.clean import CleaningConfig, run_cleaning_pipeline
from src.data.contracts import validate_dataset_file
from src.data.validate import ValidationConfig, build_data_quality_report, write_report
from src.models.baselines import train_and_evaluate_baselines


def _run_full_pipeline(args: argparse.Namespace) -> Dict[str, object]:
    summary: Dict[str, object] = {"steps": []}

    contract_report = validate_dataset_file(
        args.raw_csv,
        output_path=args.contract_report,
        nrows=args.nrows,
    )
    summary["steps"].append("contracts")
    summary["contracts"] = {"status": contract_report["status"], "failed_checks": len(contract_report["failed_checks"])}

    val_cfg = ValidationConfig(
        input_csv=args.raw_csv,
        image_dir=args.image_dir,
        output_json=args.quality_report,
        nrows=args.nrows,
    )
    quality_report = build_data_quality_report(val_cfg)
    write_report(quality_report, val_cfg.output_json)
    summary["steps"].append("quality_validation")
    summary["quality_validation"] = {"status": quality_report["status"], "issues": quality_report["issues"]}

    clean_cfg = CleaningConfig(
        input_csv=args.raw_csv,
        output_csv=args.clean_csv,
        summary_json=args.clean_summary,
        nrows=args.nrows,
    )
    clean_summary = run_cleaning_pipeline(clean_cfg)
    summary["steps"].append("cleaning")
    summary["cleaning"] = {"output_rows": clean_summary["output_rows"], "contract_status": clean_summary["contract_status"]}

    parquet_cfg = ParquetBuildConfig(
        input_csv=args.clean_csv,
        output_parquet=args.parquet_output,
        index_mapping_parquet=args.parquet_index_mapping,
        report_json=args.parquet_report,
        engine=args.parquet_engine,
        compression=args.parquet_compression,
        nrows=args.nrows,
        dry_run=args.parquet_dry_run,
    )
    try:
        parquet_report = build_parquet_assets(parquet_cfg)
    except RuntimeError as exc:
        if args.strict_parquet:
            raise
        parquet_cfg = ParquetBuildConfig(
            input_csv=args.clean_csv,
            output_parquet=args.parquet_output,
            index_mapping_parquet=args.parquet_index_mapping,
            report_json=args.parquet_report,
            engine=args.parquet_engine,
            compression=args.parquet_compression,
            nrows=args.nrows,
            dry_run=True,
        )
        parquet_report = build_parquet_assets(parquet_cfg)
        summary["parquet_warning"] = f"parquet conversion skipped: {exc}"
    summary["steps"].append("parquet_build")
    summary["parquet_build"] = {
        "dry_run": parquet_report["dry_run"],
        "rows": parquet_report["rows"],
    }

    eda_report = build_eda_report(
        input_csv=args.clean_csv,
        summary_json=args.eda_summary,
        fig_dir=args.fig_dir,
        auto_clean=False,
        nrows=args.nrows,
    )
    summary["steps"].append("eda")
    summary["eda"] = {
        "rows_analyzed": eda_report["rows_analyzed"],
        "positive_rate": eda_report["class_summary"]["positive_rate"],
    }

    baseline_report = train_and_evaluate_baselines(
        input_csv=args.clean_csv,
        report_json=args.baseline_report,
        fig_dir=args.fig_dir,
        horizon_hours=args.horizon_hours,
        meteo_lookback_steps=args.meteo_lookback,
        image_lookback_steps=args.image_lookback,
        threshold=args.threshold,
        precision_threshold=args.precision_threshold,
        auto_clean=False,
        nrows=args.nrows,
    )
    summary["steps"].append("baselines")
    summary["baselines"] = {
        "models": list(baseline_report["models"].keys()),
        "split_sizes": baseline_report["split_sizes"],
    }

    if args.include_multimodal:
        cmd = [
            sys.executable,
            "-m",
            "src.models.multimodal.train",
            "--input-csv",
            args.clean_csv,
            "--image-dir",
            args.image_dir,
            "--output-dir",
            args.multimodal_output_dir,
            "--checkpoint-path",
            args.multimodal_checkpoint,
            "--horizon-hours",
            str(args.horizon_hours),
            "--meteo-lookback",
            str(args.meteo_lookback),
            "--image-lookback",
            str(args.image_lookback),
            "--threshold",
            str(args.threshold),
            "--precision-threshold",
            str(args.precision_threshold),
            "--epochs",
            str(args.multimodal_epochs),
            "--device",
            args.multimodal_device,
        ]
        if args.nrows is not None:
            cmd.extend(["--nrows", str(args.nrows)])
        subprocess.run(cmd, check=True)
        summary["steps"].append("multimodal_train")
        summary["multimodal_train"] = {
            "checkpoint": args.multimodal_checkpoint,
            "epochs": args.multimodal_epochs,
        }

    out_path = Path(args.pipeline_summary)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MichiganCast one-command pipeline runner (T33)")
    sub = parser.add_subparsers(dest="command", required=True)

    full = sub.add_parser("full", help="Run validate -> clean -> parquet -> eda -> baselines pipeline")
    full.add_argument("--raw-csv", default="data/processed/tabular/traverse_city_daytime_meteo_preprocessed.csv")
    full.add_argument("--image-dir", default="data/processed/images/lake_michigan_64_png")
    full.add_argument("--clean-csv", default="data/interim/traverse_city_daytime_clean_v1.csv")
    full.add_argument("--contract-report", default="artifacts/reports/data_contract_report.json")
    full.add_argument("--quality-report", default="artifacts/data_quality_report.json")
    full.add_argument("--clean-summary", default="artifacts/reports/data_cleaning_summary.json")
    full.add_argument("--parquet-output", default="data/features/tabular/traverse_city_daytime_clean_v1.parquet")
    full.add_argument(
        "--parquet-index-mapping",
        default="data/features/mappings/traverse_city_daytime_image_index.parquet",
    )
    full.add_argument("--parquet-report", default="artifacts/reports/parquet_build_report.json")
    full.add_argument("--parquet-engine", default="auto", choices=["auto", "pyarrow", "fastparquet"])
    full.add_argument("--parquet-compression", default="snappy")
    full.add_argument("--parquet-dry-run", action="store_true")
    full.add_argument("--strict-parquet", action="store_true", help="Fail pipeline if parquet engine is missing")
    full.add_argument("--eda-summary", default="artifacts/reports/eda_summary.json")
    full.add_argument("--baseline-report", default="artifacts/reports/baseline_results.json")
    full.add_argument("--fig-dir", default="artifacts/figures")
    full.add_argument("--pipeline-summary", default="artifacts/reports/pipeline_summary.json")
    full.add_argument("--horizon-hours", type=int, default=24)
    full.add_argument("--meteo-lookback", type=int, default=48)
    full.add_argument("--image-lookback", type=int, default=16)
    full.add_argument("--threshold", type=float, default=0.5)
    full.add_argument("--precision-threshold", type=float, default=0.7)
    full.add_argument("--nrows", type=int, default=None)

    full.add_argument("--include-multimodal", action="store_true")
    full.add_argument("--multimodal-output-dir", default="artifacts/reports")
    full.add_argument("--multimodal-checkpoint", default="models/pytorch/michigancast_multimodal_best.pth")
    full.add_argument("--multimodal-epochs", type=int, default=8)
    full.add_argument("--multimodal-device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "full":
        summary = _run_full_pipeline(args)
        print(f"[cli] pipeline_steps={summary['steps']}")
        print(f"[cli] summary={args.pipeline_summary}")


if __name__ == "__main__":
    main()
