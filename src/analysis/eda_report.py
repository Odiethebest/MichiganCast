from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data.clean import CleaningConfig, run_cleaning_pipeline
from src.features.labeling import attach_utc_timestamp

DEFAULT_INPUT = "data/interim/traverse_city_daytime_clean_v1.csv"
DEFAULT_SUMMARY_JSON = "artifacts/reports/eda_summary.json"
DEFAULT_FIG_DIR = "artifacts/figures"


def _ensure_input_exists(input_csv: str, auto_clean: bool) -> None:
    path = Path(input_csv)
    if path.exists():
        return
    if not auto_clean:
        raise FileNotFoundError(f"Input not found: {input_csv}")
    cfg = CleaningConfig(output_csv=input_csv)
    run_cleaning_pipeline(cfg)


def _compute_class_summary(df: pd.DataFrame) -> Dict[str, object]:
    y = (df["Precip_in"] > 0).astype(int)
    pos_count = int(y.sum())
    total = int(len(y))
    pos_rate = float(pos_count / total) if total else 0.0
    neg_count = total - pos_count
    imbalance_ratio = float(neg_count / max(pos_count, 1))

    if pos_rate < 0.1:
        conclusion = (
            f"Severe class imbalance: positive precipitation rate is {pos_rate:.2%}. "
            "Rare-event-focused metrics and threshold tuning are required."
        )
    elif pos_rate < 0.2:
        conclusion = (
            f"Moderate-to-high class imbalance: positive precipitation rate is {pos_rate:.2%}. "
            "PR-AUC and recall-oriented evaluation should be prioritized."
        )
    else:
        conclusion = f"Class balance is acceptable for binary classification (positive rate={pos_rate:.2%})."

    return {
        "total_samples": total,
        "positive_samples": pos_count,
        "negative_samples": neg_count,
        "positive_rate": pos_rate,
        "negative_to_positive_ratio": imbalance_ratio,
        "class_imbalance_conclusion": conclusion,
    }


def _compute_monthly_precip_pattern(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["is_rain"] = (out["Precip_in"] > 0).astype(int)
    out["month"] = out["timestamp_utc"].dt.month
    monthly = (
        out.groupby("month", as_index=False)
        .agg(
            samples=("is_rain", "count"),
            positive_rate=("is_rain", "mean"),
            avg_precip_in=("Precip_in", "mean"),
        )
        .sort_values("month")
        .reset_index(drop=True)
    )
    return monthly


def _compute_yearly_drift(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["is_rain"] = (out["Precip_in"] > 0).astype(int)
    out["year"] = out["timestamp_utc"].dt.year
    yearly = (
        out.groupby("year", as_index=False)
        .agg(
            samples=("is_rain", "count"),
            positive_rate=("is_rain", "mean"),
            avg_temp_f=("Temp_F", "mean"),
        )
        .sort_values("year")
        .reset_index(drop=True)
    )
    return yearly


def _save_fig_class_distribution(df: pd.DataFrame, fig_dir: Path) -> str:
    fig_path = fig_dir / "eda_class_distribution.png"
    values = (df["Precip_in"] > 0).astype(int)
    counts = values.value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    plt.bar(counts.index.astype(str), counts.values, color=["#8dd3c7", "#fb8072"])
    plt.title("Rain Event Class Distribution")
    plt.xlabel("Class (0=No Rain, 1=Rain)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()
    return str(fig_path)


def _save_fig_monthly_pattern(monthly: pd.DataFrame, fig_dir: Path) -> str:
    fig_path = fig_dir / "eda_monthly_positive_rate.png"
    plt.figure(figsize=(8, 4))
    sns.lineplot(data=monthly, x="month", y="positive_rate", marker="o")
    plt.title("Monthly Rain Positive Rate")
    plt.xlabel("Month")
    plt.ylabel("Positive Rate")
    plt.ylim(0, max(0.01, monthly["positive_rate"].max() * 1.15))
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()
    return str(fig_path)


def _save_fig_correlation(df: pd.DataFrame, fig_dir: Path) -> str:
    fig_path = fig_dir / "eda_numeric_correlation_heatmap.png"
    numeric_cols = [
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
    avail = [c for c in numeric_cols if c in df.columns]
    corr = df[avail].corr(numeric_only=True)
    plt.figure(figsize=(12, 9))
    sns.heatmap(corr, cmap="YlGnBu", center=0.0, square=False)
    plt.title("Numeric Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()
    return str(fig_path)


def _save_fig_precip_hist(df: pd.DataFrame, fig_dir: Path) -> str:
    fig_path = fig_dir / "eda_precipitation_histogram.png"
    plt.figure(figsize=(7, 4))
    sns.histplot(df["Precip_in"], bins=60, kde=False, color="#80b1d3")
    plt.title("Precipitation Distribution")
    plt.xlabel("Precip_in")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()
    return str(fig_path)


def _top_correlations(df: pd.DataFrame, top_k: int = 10) -> list[dict]:
    numeric = df.select_dtypes(include=["number"])
    corr = numeric.corr(numeric_only=True).stack().reset_index()
    corr.columns = ["feature_a", "feature_b", "corr"]
    corr = corr[corr["feature_a"] != corr["feature_b"]]
    corr["abs_corr"] = corr["corr"].abs()
    corr = corr.sort_values("abs_corr", ascending=False)

    seen = set()
    rows = []
    for _, row in corr.iterrows():
        key = tuple(sorted((row["feature_a"], row["feature_b"])))
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "feature_a": row["feature_a"],
                "feature_b": row["feature_b"],
                "corr": float(row["corr"]),
            }
        )
        if len(rows) >= top_k:
            break
    return rows


def build_eda_report(
    input_csv: str,
    summary_json: str,
    fig_dir: str,
    *,
    auto_clean: bool = True,
    nrows: int | None = None,
) -> Dict[str, object]:
    _ensure_input_exists(input_csv, auto_clean=auto_clean)
    df = pd.read_csv(input_csv, nrows=nrows, low_memory=False)
    df = attach_utc_timestamp(df)

    monthly = _compute_monthly_precip_pattern(df)
    yearly = _compute_yearly_drift(df)
    class_summary = _compute_class_summary(df)

    fig_path = Path(fig_dir)
    fig_path.mkdir(parents=True, exist_ok=True)
    figures = {
        "class_distribution": _save_fig_class_distribution(df, fig_path),
        "monthly_positive_rate": _save_fig_monthly_pattern(monthly, fig_path),
        "correlation_heatmap": _save_fig_correlation(df, fig_path),
        "precip_histogram": _save_fig_precip_hist(df, fig_path),
    }

    report = {
        "input_csv": input_csv,
        "rows_analyzed": int(len(df)),
        "timestamp_start": str(df["timestamp_utc"].min()),
        "timestamp_end": str(df["timestamp_utc"].max()),
        "class_summary": class_summary,
        "monthly_precip_pattern": monthly.to_dict(orient="records"),
        "yearly_drift": yearly.to_dict(orient="records"),
        "top_numeric_correlations": _top_correlations(df, top_k=10),
        "figure_paths": figures,
    }

    out = Path(summary_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate MichiganCast EDA summary report and plots")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input cleaned CSV")
    parser.add_argument("--summary", default=DEFAULT_SUMMARY_JSON, help="Output summary JSON")
    parser.add_argument("--fig-dir", default=DEFAULT_FIG_DIR, help="Figure output directory")
    parser.add_argument("--nrows", type=int, default=None, help="Optional row limit for quick runs")
    parser.add_argument(
        "--no-auto-clean",
        action="store_true",
        help="Do not run cleaning pipeline automatically when input is missing",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    report = build_eda_report(
        args.input,
        args.summary,
        args.fig_dir,
        auto_clean=not args.no_auto_clean,
        nrows=args.nrows,
    )
    class_summary = report["class_summary"]
    print(
        f"[eda] rows={report['rows_analyzed']} "
        f"positive_rate={class_summary['positive_rate']:.4f} "
        f"summary={args.summary}"
    )
    print(f"[eda] imbalance={class_summary['class_imbalance_conclusion']}")


if __name__ == "__main__":
    main()
