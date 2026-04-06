from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.data.contracts import (
    DEFAULT_NUMERIC_RULES,
    MISSING_MARKERS,
    validate_dataframe_against_contract,
)
from src.data.validate import METEO_COLUMNS
from src.features.labeling import attach_utc_timestamp

DEFAULT_INPUT = "data/processed/tabular/traverse_city_daytime_meteo_preprocessed.csv"
DEFAULT_OUTPUT = "data/interim/traverse_city_daytime_clean_v1.csv"
DEFAULT_SUMMARY = "artifacts/reports/data_cleaning_summary.json"


@dataclass(frozen=True)
class CleaningConfig:
    input_csv: str = DEFAULT_INPUT
    output_csv: str = DEFAULT_OUTPUT
    summary_json: str = DEFAULT_SUMMARY
    require_image_filename: bool = False
    nrows: int | None = None


def _load_raw_subset(input_csv: str, nrows: int | None = None) -> pd.DataFrame:
    usecols = ["Date_UTC", "Time_UTC", "File_name_for_1D_lake", *METEO_COLUMNS]
    return pd.read_csv(input_csv, usecols=usecols, nrows=nrows, low_memory=False)


def _normalize_markers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    obj_cols = out.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        cleaned = out[col].astype(str).str.strip()
        out[col] = cleaned.where(~cleaned.isin(MISSING_MARKERS), other=pd.NA)
    return out


def _to_numeric(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _sanitize_physical_ranges(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, int]]:
    out = df.copy()
    replaced_counts: Dict[str, int] = {}
    for col, rule in DEFAULT_NUMERIC_RULES.items():
        if col not in out.columns:
            continue
        invalid = pd.Series(False, index=out.index)
        if rule.min_value is not None:
            invalid = invalid | (out[col] < rule.min_value)
        if rule.max_value is not None:
            invalid = invalid | (out[col] > rule.max_value)
        replaced_counts[col] = int(invalid.sum())
        out.loc[invalid, col] = pd.NA
    return out, replaced_counts


def _daytime_filter(df: pd.DataFrame, *, time_col: str = "Time_UTC") -> pd.DataFrame:
    out = df.copy()
    out[time_col] = out[time_col].astype(str).str.strip()
    return out[(out[time_col] >= "14:00") & (out[time_col] <= "21:00")].copy()


def clean_dataframe(raw: pd.DataFrame, *, require_image_filename: bool = False) -> tuple[pd.DataFrame, Dict[str, object]]:
    summary: Dict[str, object] = {"steps": []}
    df = raw.copy()
    summary["input_rows"] = int(len(df))

    df = _normalize_markers(df)
    summary["steps"].append("normalize_missing_markers")

    df = _to_numeric(df, METEO_COLUMNS)
    summary["steps"].append("coerce_numeric_columns")

    df, replaced = _sanitize_physical_ranges(df)
    summary["replaced_out_of_range_values"] = replaced
    summary["steps"].append("replace_out_of_range_values_with_null")

    df = _daytime_filter(df)
    summary["rows_after_daytime_filter"] = int(len(df))
    summary["steps"].append("apply_daytime_filter_14_to_21_utc")

    df = attach_utc_timestamp(df, date_col="Date_UTC", time_col="Time_UTC", timestamp_col="timestamp_utc")
    summary["steps"].append("attach_timestamp_and_sort")

    before_dedup = len(df)
    df = df.drop_duplicates(subset=["timestamp_utc"], keep="first").reset_index(drop=True)
    summary["dropped_duplicate_timestamps"] = int(before_dedup - len(df))
    summary["steps"].append("drop_duplicate_timestamps")

    required_subset = ["Date_UTC", "Time_UTC", "timestamp_utc", *METEO_COLUMNS]
    before_na_drop = len(df)
    df = df.dropna(subset=required_subset).reset_index(drop=True)
    summary["dropped_rows_with_required_nulls"] = int(before_na_drop - len(df))
    summary["steps"].append("drop_required_null_rows")

    if require_image_filename and "File_name_for_1D_lake" in df.columns:
        before_image_drop = len(df)
        has_image = df["File_name_for_1D_lake"].notna() & (df["File_name_for_1D_lake"].astype(str).str.len() > 0)
        df = df[has_image].reset_index(drop=True)
        summary["dropped_rows_missing_image_filename"] = int(before_image_drop - len(df))
        summary["steps"].append("drop_rows_without_image_filename")

    df["source_row_id"] = df.index.astype(int)
    summary["output_rows"] = int(len(df))
    summary["output_columns"] = list(df.columns)
    return df, summary


def run_cleaning_pipeline(config: CleaningConfig) -> Dict[str, object]:
    raw = _load_raw_subset(config.input_csv, nrows=config.nrows)
    clean_df, summary = clean_dataframe(raw, require_image_filename=config.require_image_filename)

    contract_report = validate_dataframe_against_contract(clean_df)
    summary["contract_status"] = contract_report["status"]
    summary["contract_failed_checks"] = contract_report["failed_checks"]

    out_csv = Path(config.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(out_csv, index=False)

    summary["config"] = asdict(config)
    summary["output_csv"] = str(out_csv)
    out_json = Path(config.summary_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MichiganCast reproducible data cleaning pipeline")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input tabular CSV")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output cleaned CSV")
    parser.add_argument("--summary", default=DEFAULT_SUMMARY, help="Output cleaning summary JSON")
    parser.add_argument(
        "--require-image-filename",
        action="store_true",
        help="Drop rows where File_name_for_1D_lake is empty",
    )
    parser.add_argument("--nrows", type=int, default=None, help="Optional row limit for debug runs")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    config = CleaningConfig(
        input_csv=args.input,
        output_csv=args.output,
        summary_json=args.summary,
        require_image_filename=args.require_image_filename,
        nrows=args.nrows,
    )
    summary = run_cleaning_pipeline(config)
    print(
        f"[clean] rows_in={summary['input_rows']} rows_out={summary['output_rows']} "
        f"contract_status={summary['contract_status']} output={summary['output_csv']}"
    )


if __name__ == "__main__":
    main()
