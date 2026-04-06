from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd


@dataclass(frozen=True)
class ParquetBuildConfig:
    input_csv: str = "data/interim/traverse_city_daytime_clean_v1.csv"
    output_parquet: str = "data/features/tabular/traverse_city_daytime_clean_v1.parquet"
    index_mapping_parquet: str = "data/features/mappings/traverse_city_daytime_image_index.parquet"
    report_json: str = "artifacts/reports/parquet_build_report.json"
    engine: str = "auto"
    compression: str = "snappy"
    nrows: int | None = None
    dry_run: bool = False


def _ensure_parent(path: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _parse_image_id(value: object) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    stem = Path(text).stem
    return int(stem) if stem.isdigit() else None


def _build_index_mapping(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["dataset_row_idx"] = range(len(df))
    if "source_row_id" in df.columns:
        out["source_row_id"] = pd.to_numeric(df["source_row_id"], errors="coerce").fillna(-1).astype(int)
    else:
        out["source_row_id"] = out["dataset_row_idx"]

    if "File_name_for_1D_lake" in df.columns:
        out["image_filename"] = df["File_name_for_1D_lake"].astype(str)
        parsed = df["File_name_for_1D_lake"].map(_parse_image_id)
        parsed_series = pd.to_numeric(pd.Series(parsed, dtype="float64"), errors="coerce")
        out["image_id"] = parsed_series.fillna(out["source_row_id"].astype(float)).astype(int)
    else:
        out["image_filename"] = ""
        out["image_id"] = out["source_row_id"]

    if "timestamp_utc" in df.columns:
        out["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce")
    elif "Date_UTC" in df.columns and "Time_UTC" in df.columns:
        out["timestamp_utc"] = pd.to_datetime(
            df["Date_UTC"].astype(str).str.strip() + " " + df["Time_UTC"].astype(str).str.strip(),
            errors="coerce",
        )
    else:
        out["timestamp_utc"] = pd.NaT
    return out


def _check_parquet_support(engine: str) -> None:
    if engine in {"auto", "pyarrow"}:
        try:
            import pyarrow  # noqa: F401
            return
        except ImportError:
            if engine == "pyarrow":
                raise RuntimeError("pyarrow is required for engine='pyarrow'.")
    if engine in {"auto", "fastparquet"}:
        try:
            import fastparquet  # noqa: F401
            return
        except ImportError:
            if engine == "fastparquet":
                raise RuntimeError("fastparquet is required for engine='fastparquet'.")
    raise RuntimeError(
        "No parquet engine available. Install one of: pyarrow, fastparquet. "
        "Example: conda install -n pytorch_env pyarrow"
    )


def _resolve_engine(engine: str) -> str:
    if engine == "auto":
        try:
            import pyarrow  # noqa: F401
            return "pyarrow"
        except ImportError:
            try:
                import fastparquet  # noqa: F401
                return "fastparquet"
            except ImportError:
                raise RuntimeError(
                    "No parquet engine found. Install pyarrow or fastparquet in pytorch_env."
                )
    return engine


def _benchmark_reader(fn, repeats: int = 3) -> float:
    fn()  # warm-up read
    begin = time.perf_counter()
    for _ in range(repeats):
        fn()
    duration = time.perf_counter() - begin
    return duration / repeats


def build_parquet_assets(cfg: ParquetBuildConfig) -> Dict[str, object]:
    input_path = Path(cfg.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {cfg.input_csv}")

    start_read = time.perf_counter()
    df = pd.read_csv(input_path, nrows=cfg.nrows, low_memory=False)
    csv_read_seconds = time.perf_counter() - start_read
    memory_mb = float(df.memory_usage(deep=True).sum() / (1024 ** 2))

    report: Dict[str, object] = {
        "input_csv": cfg.input_csv,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "memory_usage_mb": memory_mb,
        "csv_read_seconds": csv_read_seconds,
        "dry_run": cfg.dry_run,
    }

    if cfg.dry_run:
        out_report = _ensure_parent(cfg.report_json)
        out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return report

    _check_parquet_support(cfg.engine)
    resolved_engine = _resolve_engine(cfg.engine)

    out_parquet = _ensure_parent(cfg.output_parquet)
    start_write = time.perf_counter()
    df.to_parquet(out_parquet, index=False, engine=resolved_engine, compression=cfg.compression)
    parquet_write_seconds = time.perf_counter() - start_write

    mapping_df = _build_index_mapping(df)
    mapping_out = _ensure_parent(cfg.index_mapping_parquet)
    mapping_df.to_parquet(mapping_out, index=False, engine=resolved_engine, compression=cfg.compression)

    parquet_read_seconds = _benchmark_reader(
        lambda: pd.read_parquet(out_parquet, engine=resolved_engine),
        repeats=3,
    )
    csv_bench_seconds = _benchmark_reader(
        lambda: pd.read_csv(input_path, nrows=cfg.nrows, low_memory=False),
        repeats=3,
    )

    projected_cols = [
        col for col in ["timestamp_utc", "Temp_F", "RH_pct", "Sea_Lev_Press_hPa", "Precip_in"] if col in df.columns
    ]
    projected_csv_seconds = _benchmark_reader(
        lambda: pd.read_csv(input_path, usecols=projected_cols, nrows=cfg.nrows, low_memory=False),
        repeats=3,
    )
    projected_parquet_seconds = _benchmark_reader(
        lambda: pd.read_parquet(out_parquet, columns=projected_cols, engine=resolved_engine),
        repeats=3,
    )

    report.update(
        {
            "parquet_engine": resolved_engine,
            "compression": cfg.compression,
            "output_parquet": str(out_parquet),
            "index_mapping_parquet": str(mapping_out),
            "parquet_size_bytes": int(out_parquet.stat().st_size),
            "index_mapping_size_bytes": int(mapping_out.stat().st_size),
            "parquet_write_seconds": parquet_write_seconds,
            "parquet_read_seconds": parquet_read_seconds,
            "csv_read_benchmark_seconds": csv_bench_seconds,
            "csv_to_parquet_read_speedup": float(csv_bench_seconds / max(parquet_read_seconds, 1e-9)),
            "projected_columns": projected_cols,
            "projected_csv_read_seconds": projected_csv_seconds,
            "projected_parquet_read_seconds": projected_parquet_seconds,
            "projected_csv_to_parquet_speedup": float(
                projected_csv_seconds / max(projected_parquet_seconds, 1e-9)
            ),
            "mapping_columns": list(mapping_df.columns),
        }
    )

    out_report = _ensure_parent(cfg.report_json)
    out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Parquet table and row-image index mapping (T32)")
    parser.add_argument("--input", default="data/interim/traverse_city_daytime_clean_v1.csv")
    parser.add_argument("--output", default="data/features/tabular/traverse_city_daytime_clean_v1.parquet")
    parser.add_argument(
        "--index-mapping",
        default="data/features/mappings/traverse_city_daytime_image_index.parquet",
    )
    parser.add_argument("--report", default="artifacts/reports/parquet_build_report.json")
    parser.add_argument("--engine", default="auto", choices=["auto", "pyarrow", "fastparquet"])
    parser.add_argument("--compression", default="snappy")
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    cfg = ParquetBuildConfig(
        input_csv=args.input,
        output_parquet=args.output,
        index_mapping_parquet=args.index_mapping,
        report_json=args.report,
        engine=args.engine,
        compression=args.compression,
        nrows=args.nrows,
        dry_run=args.dry_run,
    )
    report = build_parquet_assets(cfg)
    print(f"[parquet] rows={report['rows']} cols={report['columns']} dry_run={report['dry_run']}")
    if not report["dry_run"]:
        print(f"[parquet] output={report['output_parquet']}")
        print(f"[parquet] index_mapping={report['index_mapping_parquet']}")
        print(f"[parquet] csv_to_parquet_read_speedup={report['csv_to_parquet_read_speedup']:.3f}")
    print(f"[parquet] report={cfg.report_json}")


if __name__ == "__main__":
    main()
