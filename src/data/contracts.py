from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

MISSING_MARKERS = {"", "m", "M", "NC", "na", "NA", "nan", "NaN", "None", "null", "NULL"}

REQUIRED_COLUMNS = [
    "Date_UTC",
    "Time_UTC",
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
class NumericRule:
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    max_missing_ratio: float = 0.0


DEFAULT_NUMERIC_RULES: Dict[str, NumericRule] = {
    "Temp_F": NumericRule(min_value=-80.0, max_value=130.0, max_missing_ratio=0.02),
    "RH_pct": NumericRule(min_value=0.0, max_value=100.0, max_missing_ratio=0.02),
    "Dewpt_F": NumericRule(min_value=-100.0, max_value=100.0, max_missing_ratio=0.02),
    "Wind_Spd_mph": NumericRule(min_value=0.0, max_value=150.0, max_missing_ratio=0.02),
    "Wind_Direction_deg": NumericRule(min_value=0.0, max_value=360.0, max_missing_ratio=0.05),
    "Peak_Wind_Gust_mph": NumericRule(min_value=0.0, max_value=200.0, max_missing_ratio=0.05),
    "Low_Cloud_Ht_ft": NumericRule(min_value=0.0, max_value=70000.0, max_missing_ratio=0.1),
    "Med_Cloud_Ht_ft": NumericRule(min_value=0.0, max_value=70000.0, max_missing_ratio=0.1),
    "High_Cloud_Ht_ft": NumericRule(min_value=0.0, max_value=70000.0, max_missing_ratio=0.1),
    "Visibility_mi": NumericRule(min_value=0.0, max_value=80.0, max_missing_ratio=0.05),
    "Atm_Press_hPa": NumericRule(min_value=800.0, max_value=1100.0, max_missing_ratio=0.02),
    "Sea_Lev_Press_hPa": NumericRule(min_value=800.0, max_value=1100.0, max_missing_ratio=0.02),
    "Altimeter_hPa": NumericRule(min_value=800.0, max_value=1100.0, max_missing_ratio=0.02),
    "Precip_in": NumericRule(min_value=0.0, max_value=20.0, max_missing_ratio=0.02),
}


def _marker_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_string_dtype(series) or series.dtype == object:
        normalized = series.astype(str).str.strip()
        return normalized.isin(MISSING_MARKERS) | series.isna()
    return series.isna()


def _sample_indices(mask: pd.Series, max_examples: int) -> List[int]:
    return mask[mask].index.to_series().head(max_examples).astype(int).tolist()


def _normalize_for_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_string_dtype(series) or series.dtype == object:
        cleaned = series.astype(str).str.strip()
        cleaned = cleaned.where(~cleaned.isin(MISSING_MARKERS), other=pd.NA)
        return pd.to_numeric(cleaned, errors="coerce")
    return pd.to_numeric(series, errors="coerce")


def validate_dataframe_against_contract(
    df: pd.DataFrame,
    *,
    required_columns: Iterable[str] = REQUIRED_COLUMNS,
    numeric_rules: Dict[str, NumericRule] = DEFAULT_NUMERIC_RULES,
    max_row_examples: int = 20,
) -> Dict[str, object]:
    """Validate schema and value constraints with row-level diagnostics."""
    required = list(required_columns)
    missing_required = [c for c in required if c not in df.columns]

    report: Dict[str, object] = {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "missing_required_columns": missing_required,
        "column_checks": {},
        "failed_checks": [],
        "status": "pass",
    }

    if missing_required:
        report["failed_checks"].append(
            {
                "check": "required_columns_present",
                "detail": f"Missing required columns: {missing_required}",
            }
        )

    for col, rule in numeric_rules.items():
        if col not in df.columns:
            continue

        col_report: Dict[str, object] = {"rule": rule.__dict__.copy()}
        raw = df[col]
        marker = _marker_mask(raw)
        numeric = _normalize_for_numeric(raw)

        missing_ratio = float(marker.mean()) if len(marker) else 0.0
        non_numeric_mask = (~marker) & numeric.isna()
        below_min = pd.Series(False, index=df.index)
        above_max = pd.Series(False, index=df.index)
        if rule.min_value is not None:
            below_min = numeric < rule.min_value
        if rule.max_value is not None:
            above_max = numeric > rule.max_value

        col_report["missing_count"] = int(marker.sum())
        col_report["missing_ratio"] = missing_ratio
        col_report["missing_row_indices"] = _sample_indices(marker, max_row_examples)

        col_report["non_numeric_count"] = int(non_numeric_mask.sum())
        col_report["non_numeric_row_indices"] = _sample_indices(non_numeric_mask, max_row_examples)

        col_report["below_min_count"] = int(below_min.sum())
        col_report["below_min_row_indices"] = _sample_indices(below_min, max_row_examples)

        col_report["above_max_count"] = int(above_max.sum())
        col_report["above_max_row_indices"] = _sample_indices(above_max, max_row_examples)

        violations = []
        if missing_ratio > rule.max_missing_ratio:
            violations.append(
                f"missing_ratio {missing_ratio:.4f} > allowed {rule.max_missing_ratio:.4f}"
            )
        if int(non_numeric_mask.sum()) > 0:
            violations.append("contains non-numeric values")
        if int(below_min.sum()) > 0:
            violations.append(f"contains values below min_value={rule.min_value}")
        if int(above_max.sum()) > 0:
            violations.append(f"contains values above max_value={rule.max_value}")

        col_report["violations"] = violations
        if violations:
            report["failed_checks"].append(
                {"check": f"numeric_rule::{col}", "detail": "; ".join(violations)}
            )

        report["column_checks"][col] = col_report

    if report["failed_checks"]:
        report["status"] = "fail"
    return report


def validate_dataset_file(
    input_path: str | Path,
    *,
    output_path: str | Path = "artifacts/reports/data_contract_report.json",
    nrows: Optional[int] = None,
    max_row_examples: int = 20,
) -> Dict[str, object]:
    """Read dataset, validate against contract, and write JSON report."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path, nrows=nrows, low_memory=False)
    report = validate_dataframe_against_contract(df, max_row_examples=max_row_examples)
    report["input_path"] = str(path)
    report["validated_nrows"] = int(len(df))

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate dataset against MichiganCast data contract")
    parser.add_argument(
        "--input",
        default="data/processed/tabular/traverse_city_daytime_meteo_preprocessed.csv",
        help="Input CSV path",
    )
    parser.add_argument(
        "--output",
        default="artifacts/reports/data_contract_report.json",
        help="Output JSON report path",
    )
    parser.add_argument("--nrows", type=int, default=None, help="Optional row limit for quick checks")
    parser.add_argument(
        "--max-row-examples",
        type=int,
        default=20,
        help="Max row indices to keep per failed check",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    report = validate_dataset_file(
        args.input,
        output_path=args.output,
        nrows=args.nrows,
        max_row_examples=args.max_row_examples,
    )
    print(f"[contracts] status={report['status']} rows={report['validated_nrows']} output={args.output}")
    print(f"[contracts] failed_checks={len(report['failed_checks'])}")


if __name__ == "__main__":
    main()
