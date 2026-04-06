from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd

from src.features.labeling import (
    DEFAULT_DATE_COL,
    DEFAULT_TIME_COL,
    DEFAULT_TIMESTAMP_COL,
    attach_utc_timestamp,
)


@dataclass(frozen=True)
class TimeSplitConfig:
    """Year-based split boundaries."""

    train_years: tuple[int, int] = (2006, 2012)
    val_years: tuple[int, int] = (2013, 2014)
    test_years: tuple[int, int] = (2015, 2015)
    timestamp_col: str = DEFAULT_TIMESTAMP_COL


def _year_mask(series: pd.Series, year_range: tuple[int, int]) -> pd.Series:
    start, end = year_range
    if start > end:
        raise ValueError(f"Invalid year range: {year_range}")
    years = series.dt.year
    return (years >= start) & (years <= end)


def _describe_frame(name: str, df: pd.DataFrame, timestamp_col: str) -> str:
    if df.empty:
        return f"{name:<5} rows=0"
    start = df[timestamp_col].min()
    end = df[timestamp_col].max()
    return (
        f"{name:<5} rows={len(df):>7} "
        f"start={start} end={end} years={int(start.year)}-{int(end.year)}"
    )


def print_split_summary(splits: Dict[str, pd.DataFrame], timestamp_col: str) -> None:
    """Print row count and time span for each split."""
    print("Time-based split summary:")
    for name in ("train", "val", "test"):
        print(_describe_frame(name, splits[name], timestamp_col))


def split_dataframe_by_year(
    df: pd.DataFrame,
    *,
    config: TimeSplitConfig | None = None,
    date_col: str = DEFAULT_DATE_COL,
    time_col: str = DEFAULT_TIME_COL,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Split raw rows by timestamp year range.

    This is a strict time split (no random shuffle).
    """
    cfg = config or TimeSplitConfig()
    data = (
        df.copy()
        if cfg.timestamp_col in df.columns
        else attach_utc_timestamp(df, date_col=date_col, time_col=time_col, timestamp_col=cfg.timestamp_col)
    )
    data = data.sort_values(cfg.timestamp_col).reset_index(drop=True)

    train = data[_year_mask(data[cfg.timestamp_col], cfg.train_years)].copy()
    val = data[_year_mask(data[cfg.timestamp_col], cfg.val_years)].copy()
    test = data[_year_mask(data[cfg.timestamp_col], cfg.test_years)].copy()

    overlap = (set(train.index) & set(val.index)) | (set(train.index) & set(test.index)) | (set(val.index) & set(test.index))
    if overlap:
        raise ValueError("Split overlap detected, year ranges must be disjoint.")

    splits = {"train": train, "val": val, "test": test}
    if verbose:
        print_split_summary(splits, cfg.timestamp_col)
    return splits


def split_samples_by_label_year(
    sample_index: pd.DataFrame,
    *,
    config: TimeSplitConfig | None = None,
    label_time_col: str = "y_time",
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Split pre-built forecast samples by label timestamp year.

    Using y_time for split avoids future-label leakage across train/val/test.
    """
    cfg = config or TimeSplitConfig()
    if label_time_col not in sample_index.columns:
        raise ValueError(f"Missing label time column: {label_time_col}")

    data = sample_index.copy()
    data[label_time_col] = pd.to_datetime(data[label_time_col], errors="raise")

    train = data[_year_mask(data[label_time_col], cfg.train_years)].copy()
    val = data[_year_mask(data[label_time_col], cfg.val_years)].copy()
    test = data[_year_mask(data[label_time_col], cfg.test_years)].copy()

    splits = {"train": train, "val": val, "test": test}
    if verbose:
        print_split_summary(splits, label_time_col)
    return splits


def _parse_year_range(text: str) -> tuple[int, int]:
    parts = text.split(":")
    if len(parts) != 2:
        raise ValueError(f"Expected YEAR_START:YEAR_END, got: {text}")
    return int(parts[0]), int(parts[1])


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Time-based data splitting for MichiganCast")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output-dir", default="", help="Optional directory for split CSV outputs")
    parser.add_argument("--train-years", default="2006:2012", help="Train year range, e.g. 2006:2012")
    parser.add_argument("--val-years", default="2013:2014", help="Validation year range, e.g. 2013:2014")
    parser.add_argument("--test-years", default="2015:2015", help="Test year range, e.g. 2015:2015")
    parser.add_argument("--sample-index", action="store_true", help="Set when input is sample index with y_time column")
    parser.add_argument("--date-col", default=DEFAULT_DATE_COL)
    parser.add_argument("--time-col", default=DEFAULT_TIME_COL)
    parser.add_argument("--timestamp-col", default=DEFAULT_TIMESTAMP_COL)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    cfg = TimeSplitConfig(
        train_years=_parse_year_range(args.train_years),
        val_years=_parse_year_range(args.val_years),
        test_years=_parse_year_range(args.test_years),
        timestamp_col=args.timestamp_col,
    )

    data = pd.read_csv(args.input)
    if args.sample_index:
        splits = split_samples_by_label_year(data, config=cfg, label_time_col="y_time", verbose=True)
    else:
        splits = split_dataframe_by_year(
            data,
            config=cfg,
            date_col=args.date_col,
            time_col=args.time_col,
            verbose=True,
        )

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, split_df in splits.items():
            out_path = out_dir / f"{name}.csv"
            split_df.to_csv(out_path, index=False)
            print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
