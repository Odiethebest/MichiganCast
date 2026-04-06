from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd

DEFAULT_TIMESTAMP_COL = "timestamp_utc"
DEFAULT_DATE_COL = "Date_UTC"
DEFAULT_TIME_COL = "Time_UTC"
DEFAULT_PRECIP_COL = "Precip_in"
DEFAULT_TARGET_COL = "target_rain"

DEFAULT_DROP_COLUMNS = (
    "Date_UTC",
    "Time_UTC",
    "Date_CST",
    "Time_CST",
    "File_name_for_1D_lake",
    "File_name_for_2D_lake",
    "Lake_data_1D",
    "Lake_data_2D",
    "precipitation_category",
    "LES_Precipitation",
)


@dataclass(frozen=True)
class ForecastSamplingConfig:
    """Sampling config for building future-forecast training examples."""

    horizons_hours: tuple[int, ...] = (6, 24)
    meteo_lookback_steps: int = 48
    image_lookback_steps: int = 16
    rain_threshold: float = 0.0
    precip_col: str = DEFAULT_PRECIP_COL
    target_col: str = DEFAULT_TARGET_COL
    date_col: str = DEFAULT_DATE_COL
    time_col: str = DEFAULT_TIME_COL
    timestamp_col: str = DEFAULT_TIMESTAMP_COL
    forbidden_same_time_features: tuple[str, ...] = (DEFAULT_PRECIP_COL,)

    @property
    def max_lookback_steps(self) -> int:
        return max(self.meteo_lookback_steps, self.image_lookback_steps)


def attach_utc_timestamp(
    df: pd.DataFrame,
    *,
    date_col: str = DEFAULT_DATE_COL,
    time_col: str = DEFAULT_TIME_COL,
    timestamp_col: str = DEFAULT_TIMESTAMP_COL,
) -> pd.DataFrame:
    """Create a canonical timestamp column and return time-sorted dataframe."""
    required = {date_col, time_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required time columns: {sorted(missing)}")

    out = df.copy()
    out[timestamp_col] = pd.to_datetime(
        out[date_col].astype(str).str.strip() + " " + out[time_col].astype(str).str.strip(),
        format="%Y-%m-%d %H:%M",
        errors="raise",
    )
    out = out.sort_values(timestamp_col).reset_index(drop=True)

    duplicated = out[timestamp_col].duplicated()
    if duplicated.any():
        duplicate_values = out.loc[duplicated, timestamp_col].astype(str).head(5).tolist()
        raise ValueError(
            "Duplicate timestamps found. Expected one row per timestamp. "
            f"Sample duplicates: {duplicate_values}"
        )
    return out


def build_meteorological_feature_columns(
    all_columns: Sequence[str],
    *,
    drop_columns: Iterable[str] = DEFAULT_DROP_COLUMNS,
    forbid_same_time_columns: Iterable[str] = (DEFAULT_PRECIP_COL,),
) -> list[str]:
    """
    Build meteorological feature list with leakage-safe defaults.

    `Precip_in` is excluded by default to avoid accidental same-time leakage.
    """
    drop = set(drop_columns)
    forbidden = set(forbid_same_time_columns)
    selected = [col for col in all_columns if col not in drop and col not in forbidden]
    if not selected:
        raise ValueError("No usable meteorological features left after filtering.")
    return selected


def validate_temporal_order(sample_index: pd.DataFrame) -> None:
    """
    Enforce the hard constraint required by T01:
    X_time must be strictly earlier than y_time.
    """
    required_cols = {
        "anchor_time",
        "y_time",
        "anchor_idx",
        "y_idx",
        "meteo_end_idx",
        "image_end_idx",
    }
    missing = required_cols - set(sample_index.columns)
    if missing:
        raise ValueError(f"Sample index missing required columns: {sorted(missing)}")

    invalid = sample_index[
        (sample_index["anchor_time"] >= sample_index["y_time"])
        | (sample_index["anchor_idx"] >= sample_index["y_idx"])
        | (sample_index["meteo_end_idx"] >= sample_index["y_idx"])
        | (sample_index["image_end_idx"] >= sample_index["y_idx"])
    ]
    if not invalid.empty:
        raise ValueError(
            "Temporal leakage detected: some samples do not satisfy X_time < y_time. "
            f"Invalid rows: {len(invalid)}"
        )


def validate_no_same_time_leakage(
    sample_index: pd.DataFrame,
    *,
    forbidden_features: Iterable[str] = (DEFAULT_PRECIP_COL,),
) -> None:
    """
    Auto-check for same-timestep leakage.

    The index-based checks guarantee no input window reaches y_idx.
    We also persist the forbidden feature policy in DataFrame metadata.
    """
    validate_temporal_order(sample_index)
    sample_index.attrs["forbidden_same_time_features"] = list(forbidden_features)


def build_forecast_sample_index(
    df: pd.DataFrame,
    *,
    config: ForecastSamplingConfig | None = None,
) -> pd.DataFrame:
    """
    Build sample index for future forecasting.

    For each anchor time t, predict precipitation at t + horizon_hours.
    """
    cfg = config or ForecastSamplingConfig()
    if cfg.precip_col not in df.columns:
        raise ValueError(f"Missing precipitation column: {cfg.precip_col}")
    if cfg.meteo_lookback_steps <= 0 or cfg.image_lookback_steps <= 0:
        raise ValueError("Lookback steps must be positive.")
    if not cfg.horizons_hours or any(h <= 0 for h in cfg.horizons_hours):
        raise ValueError("All forecast horizons must be positive integers.")

    data = (
        df.copy()
        if cfg.timestamp_col in df.columns
        else attach_utc_timestamp(
            df,
            date_col=cfg.date_col,
            time_col=cfg.time_col,
            timestamp_col=cfg.timestamp_col,
        )
    )
    data = data.sort_values(cfg.timestamp_col).reset_index(drop=True)
    time_to_idx = {ts: i for i, ts in enumerate(data[cfg.timestamp_col])}

    records: list[dict] = []
    sample_id = 0
    for anchor_idx in range(cfg.max_lookback_steps - 1, len(data)):
        anchor_time = data.at[anchor_idx, cfg.timestamp_col]
        meteo_start_idx = anchor_idx - cfg.meteo_lookback_steps + 1
        image_start_idx = anchor_idx - cfg.image_lookback_steps + 1

        for horizon_hours in cfg.horizons_hours:
            y_time = anchor_time + pd.Timedelta(hours=int(horizon_hours))
            y_idx = time_to_idx.get(y_time)
            if y_idx is None or y_idx <= anchor_idx:
                continue

            y_precip = float(data.at[y_idx, cfg.precip_col])
            records.append(
                {
                    "sample_id": sample_id,
                    "horizon_hours": int(horizon_hours),
                    "anchor_idx": anchor_idx,
                    "anchor_time": anchor_time,
                    "meteo_start_idx": meteo_start_idx,
                    "meteo_end_idx": anchor_idx,
                    "image_start_idx": image_start_idx,
                    "image_end_idx": anchor_idx,
                    "y_idx": y_idx,
                    "y_time": y_time,
                    "y_precip": y_precip,
                    cfg.target_col: int(y_precip > cfg.rain_threshold),
                }
            )
            sample_id += 1

    sample_index = pd.DataFrame.from_records(records)
    if sample_index.empty:
        raise ValueError(
            "No valid forecast samples created. Check timestamp continuity, "
            "lookback windows, and forecast horizons."
        )

    validate_no_same_time_leakage(
        sample_index,
        forbidden_features=cfg.forbidden_same_time_features,
    )
    return sample_index


def summarize_sample_index(
    sample_index: pd.DataFrame,
    *,
    target_col: str = DEFAULT_TARGET_COL,
) -> pd.DataFrame:
    """Create a compact per-horizon summary table."""
    required = {"horizon_hours", "anchor_time", "y_time", target_col}
    missing = required - set(sample_index.columns)
    if missing:
        raise ValueError(f"Missing summary columns: {sorted(missing)}")

    summary = (
        sample_index.groupby("horizon_hours", as_index=False)
        .agg(
            samples=("sample_id", "count"),
            anchor_start=("anchor_time", "min"),
            anchor_end=("anchor_time", "max"),
            label_start=("y_time", "min"),
            label_end=("y_time", "max"),
            positive_rate=(target_col, "mean"),
        )
        .sort_values("horizon_hours")
        .reset_index(drop=True)
    )
    return summary
