from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from src.data.contracts import MISSING_MARKERS

DEFAULT_INPUT = "data/processed/tabular/traverse_city_daytime_meteo_preprocessed.csv"
DEFAULT_IMAGE_DIR = "data/processed/images/lake_michigan_64_png"
DEFAULT_OUTPUT = "artifacts/data_quality_report.json"

METEO_COLUMNS = [
    "Temp_F",
    "RH_pct",
    "Dewpt_F",
    "Wind_Spd_mph",
    "Wind_Direction_deg",
    "Peak_Wind_Gust_mph",
    "Low_Cloud_Ht_ft",
    "Med_Cloud_Ht_ft",
    "High_Cloud_Ht_ft",
    "Visibility_mi",
    "Atm_Press_hPa",
    "Sea_Lev_Press_hPa",
    "Altimeter_hPa",
    "Precip_in",
]


@dataclass(frozen=True)
class ValidationConfig:
    input_csv: str = DEFAULT_INPUT
    image_dir: str = DEFAULT_IMAGE_DIR
    output_json: str = DEFAULT_OUTPUT
    nrows: int | None = None
    max_examples: int = 20


def _sample_indices(mask: pd.Series, max_examples: int) -> List[int]:
    return mask[mask].index.to_series().head(max_examples).astype(int).tolist()


def _load_validation_dataframe(input_csv: str, nrows: int | None = None) -> pd.DataFrame:
    usecols = ["Date_UTC", "Time_UTC", "File_name_for_1D_lake", "File_name_for_2D_lake", *METEO_COLUMNS]
    return pd.read_csv(input_csv, usecols=usecols, nrows=nrows, low_memory=False)


def check_missing_code_markers(df: pd.DataFrame, *, max_examples: int = 20) -> Dict[str, object]:
    markers = MISSING_MARKERS | {"m", "M", "NC"}
    per_column: Dict[str, object] = {}
    total_marker_cells = 0

    for col in METEO_COLUMNS:
        if col not in df.columns:
            continue
        series = df[col]
        text = series.astype(str).str.strip()
        marker_mask = text.isin(markers)
        marker_count = int(marker_mask.sum())
        total_marker_cells += marker_count
        per_column[col] = {
            "marker_count": marker_count,
            "marker_ratio": float(marker_count / len(df)) if len(df) else 0.0,
            "sample_row_indices": _sample_indices(marker_mask, max_examples),
        }

    return {
        "total_marker_cells": total_marker_cells,
        "columns": per_column,
    }


def check_time_continuity(df: pd.DataFrame, *, max_examples: int = 20) -> Dict[str, object]:
    required = {"Date_UTC", "Time_UTC"}
    missing = required - set(df.columns)
    if missing:
        return {"error": f"Missing time columns: {sorted(missing)}"}

    local = df.copy()
    local["timestamp_utc"] = pd.to_datetime(
        local["Date_UTC"].astype(str).str.strip() + " " + local["Time_UTC"].astype(str).str.strip(),
        format="%Y-%m-%d %H:%M",
        errors="coerce",
    )
    null_ts_mask = local["timestamp_utc"].isna()
    if null_ts_mask.any():
        return {
            "error": "Failed to parse some timestamps",
            "invalid_timestamp_rows": _sample_indices(null_ts_mask, max_examples),
        }

    local = local.sort_values("timestamp_utc").reset_index(drop=True)
    local["delta_hours"] = local["timestamp_utc"].diff().dt.total_seconds() / 3600.0

    # For daytime-only data, common expected steps are 1h (intra-day) and 17h (overnight).
    allowed_short_gaps = {1.0, 17.0}
    gap_mask = (~local["delta_hours"].isna()) & (~local["delta_hours"].isin(allowed_short_gaps))

    gap_examples = []
    for idx in local.index[gap_mask].tolist()[:max_examples]:
        gap_examples.append(
            {
                "row_index": int(idx),
                "prev_timestamp": str(local.at[idx - 1, "timestamp_utc"]),
                "curr_timestamp": str(local.at[idx, "timestamp_utc"]),
                "delta_hours": float(local.at[idx, "delta_hours"]),
            }
        )

    return {
        "row_count": int(len(local)),
        "timestamp_start": str(local["timestamp_utc"].min()),
        "timestamp_end": str(local["timestamp_utc"].max()),
        "nonstandard_gap_count": int(gap_mask.sum()),
        "allowed_short_gaps_hours": sorted(allowed_short_gaps),
        "sample_nonstandard_gaps": gap_examples,
    }


def inspect_image_inventory(image_dir: str, *, max_examples: int = 20) -> Dict[str, object]:
    path = Path(image_dir)
    if not path.exists():
        return {"error": f"Image directory not found: {image_dir}"}

    png_files = sorted(path.glob("*.png"))
    numeric_ids = []
    non_numeric_files = []
    zero_byte_files = []

    for file in png_files:
        if file.stat().st_size == 0:
            zero_byte_files.append(file.name)
        stem = file.stem
        if stem.isdigit():
            numeric_ids.append(int(stem))
        else:
            non_numeric_files.append(file.name)

    numeric_set = set(numeric_ids)
    missing_between_min_max = []
    if numeric_ids:
        for image_id in range(min(numeric_ids), max(numeric_ids) + 1):
            if image_id not in numeric_set:
                missing_between_min_max.append(image_id)
                if len(missing_between_min_max) >= max_examples:
                    break

    return {
        "image_dir": str(path),
        "png_count": len(png_files),
        "numeric_id_count": len(numeric_ids),
        "min_numeric_id": min(numeric_ids) if numeric_ids else None,
        "max_numeric_id": max(numeric_ids) if numeric_ids else None,
        "non_numeric_file_count": len(non_numeric_files),
        "non_numeric_file_examples": non_numeric_files[:max_examples],
        "zero_byte_file_count": len(zero_byte_files),
        "zero_byte_file_examples": zero_byte_files[:max_examples],
        "missing_ids_between_min_max_count": len(missing_between_min_max),
        "missing_ids_between_min_max_examples": missing_between_min_max[:max_examples],
        "numeric_ids": numeric_ids,
    }


def check_image_table_alignment(
    df: pd.DataFrame,
    image_inventory: Dict[str, object],
    *,
    max_examples: int = 20,
) -> Dict[str, object]:
    if "numeric_ids" not in image_inventory:
        return {"error": "Image inventory does not include numeric IDs."}

    row_count = len(df)
    expected_ids = set(range(row_count))
    available_ids = set(image_inventory["numeric_ids"])
    missing_ids = sorted(expected_ids - available_ids)
    extra_ids = sorted(available_ids - expected_ids)

    missing_filename_mask = pd.Series(False, index=df.index)
    if "File_name_for_1D_lake" in df.columns:
        missing_filename_mask = (
            df["File_name_for_1D_lake"].isna()
            | df["File_name_for_1D_lake"].astype(str).str.strip().isin(MISSING_MARKERS)
        )

    return {
        "table_row_count": int(row_count),
        "expected_image_id_range": [0, max(row_count - 1, 0)],
        "available_numeric_image_count": int(len(available_ids)),
        "missing_image_ids_count": int(len(missing_ids)),
        "missing_image_id_examples": missing_ids[:max_examples],
        "extra_image_ids_count": int(len(extra_ids)),
        "extra_image_id_examples": extra_ids[:max_examples],
        "missing_file_name_for_1d_lake_count": int(missing_filename_mask.sum()),
        "missing_file_name_row_examples": _sample_indices(missing_filename_mask, max_examples),
    }


def build_data_quality_report(config: ValidationConfig) -> Dict[str, object]:
    df = _load_validation_dataframe(config.input_csv, nrows=config.nrows)
    marker_report = check_missing_code_markers(df, max_examples=config.max_examples)
    continuity_report = check_time_continuity(df, max_examples=config.max_examples)
    image_inventory = inspect_image_inventory(config.image_dir, max_examples=config.max_examples)
    alignment_report = check_image_table_alignment(df, image_inventory, max_examples=config.max_examples)

    status = "pass"
    issues = []
    if marker_report.get("total_marker_cells", 0) > 0:
        status = "warn"
        issues.append("found_missing_markers_in_meteorological_columns")
    if continuity_report.get("nonstandard_gap_count", 0) > 0:
        status = "warn"
        issues.append("found_nonstandard_time_gaps")
    if alignment_report.get("missing_image_ids_count", 0) > 0 or alignment_report.get("extra_image_ids_count", 0) > 0:
        status = "warn"
        issues.append("image_table_alignment_mismatch")
    if "error" in continuity_report or "error" in image_inventory or "error" in alignment_report:
        status = "fail"
        issues.append("validation_error")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "issues": issues,
        "input_csv": config.input_csv,
        "image_dir": config.image_dir,
        "validated_rows": int(len(df)),
        "missing_code_check": marker_report,
        "time_continuity_check": continuity_report,
        "image_inventory_check": {k: v for k, v in image_inventory.items() if k != "numeric_ids"},
        "image_table_alignment_check": alignment_report,
    }
    return report


def write_report(report: Dict[str, object], output_json: str) -> None:
    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MichiganCast data quality validation checks")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input tabular CSV")
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR, help="Processed image directory")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output JSON report path")
    parser.add_argument("--nrows", type=int, default=None, help="Optional row limit")
    parser.add_argument("--max-examples", type=int, default=20, help="Max row/file examples to keep")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    config = ValidationConfig(
        input_csv=args.input,
        image_dir=args.image_dir,
        output_json=args.output,
        nrows=args.nrows,
        max_examples=args.max_examples,
    )
    report = build_data_quality_report(config)
    write_report(report, config.output_json)
    print(f"[validate] status={report['status']} rows={report['validated_rows']} output={config.output_json}")
    print(f"[validate] issues={report['issues']}")


if __name__ == "__main__":
    main()
