from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.split import TimeSplitConfig, split_samples_by_label_year
from src.features.labeling import (
    ForecastSamplingConfig,
    attach_utc_timestamp,
    build_forecast_sample_index,
    build_meteorological_feature_columns,
)


@dataclass(frozen=True)
class DatasetBuildConfig:
    image_dir: str
    horizon_hours: int = 24
    meteo_lookback_steps: int = 48
    image_lookback_steps: int = 16
    image_size: int = 64
    normalize_images: bool = True
    image_id_col: str = "source_row_id"
    target_col: str = "target_rain"
    max_samples_per_split: int | None = None
    train_years: Tuple[int, int] = (2006, 2012)
    val_years: Tuple[int, int] = (2013, 2014)
    test_years: Tuple[int, int] = (2015, 2015)


class MultimodalForecastDataset(Dataset):
    """Return aligned image+meteo sequences for binary rain forecasting."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        sample_index: pd.DataFrame,
        *,
        feature_columns: Sequence[str],
        image_dir: str | Path,
        image_id_col: str = "source_row_id",
        image_size: int = 64,
        normalize_images: bool = True,
        target_col: str = "target_rain",
        drop_missing_images: bool = True,
    ) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.sample_index = sample_index.reset_index(drop=True)
        self.feature_columns = list(feature_columns)
        self.image_dir = Path(image_dir)
        self.image_id_col = image_id_col
        self.image_size = image_size
        self.normalize_images = normalize_images
        self.target_col = target_col

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        required_sample_cols = {
            "meteo_start_idx",
            "meteo_end_idx",
            "image_start_idx",
            "image_end_idx",
            target_col,
            "y_time",
        }
        missing = required_sample_cols - set(self.sample_index.columns)
        if missing:
            raise ValueError(f"Sample index missing required columns: {sorted(missing)}")

        if image_id_col in self.df.columns:
            self.image_ids = self.df[image_id_col].astype(int).to_numpy()
        else:
            self.image_ids = np.arange(len(self.df), dtype=np.int64)

        self.feature_array = self.df[self.feature_columns].to_numpy(dtype=np.float32)
        self.available_image_ids = self._scan_available_image_ids()
        self.filtered_out_missing_images = 0
        if drop_missing_images:
            self._filter_samples_with_missing_images()

    def _scan_available_image_ids(self) -> set[int]:
        ids = set()
        for file in self.image_dir.glob("*.png"):
            if file.stem.isdigit():
                ids.add(int(file.stem))
        return ids

    def _filter_samples_with_missing_images(self) -> None:
        keep_mask = []
        for _, row in self.sample_index.iterrows():
            start = int(row["image_start_idx"])
            end = int(row["image_end_idx"])
            needed_ids = self.image_ids[start : end + 1]
            valid = True
            for image_id in needed_ids:
                if int(image_id) not in self.available_image_ids:
                    valid = False
                    break
            keep_mask.append(valid)

        keep_mask_series = pd.Series(keep_mask, dtype=bool)
        dropped = int((~keep_mask_series).sum())
        self.filtered_out_missing_images = dropped
        self.sample_index = self.sample_index.loc[keep_mask_series].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.sample_index)

    def _load_image_tensor(self, image_id: int) -> torch.Tensor:
        path = self.image_dir / f"{image_id}.png"
        with Image.open(path) as img:
            gray = img.convert("L")
            if self.image_size > 0:
                gray = gray.resize((self.image_size, self.image_size), Image.BILINEAR)
            arr = np.asarray(gray, dtype=np.float32) / 255.0
        if self.normalize_images:
            arr = (arr - 0.5) / 0.5
        return torch.from_numpy(arr).unsqueeze(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.sample_index.iloc[idx]
        meteo_start = int(row["meteo_start_idx"])
        meteo_end = int(row["meteo_end_idx"])
        image_start = int(row["image_start_idx"])
        image_end = int(row["image_end_idx"])

        meteo_seq = torch.from_numpy(self.feature_array[meteo_start : meteo_end + 1])
        image_ids = self.image_ids[image_start : image_end + 1]
        image_seq = torch.stack([self._load_image_tensor(int(image_id)) for image_id in image_ids], dim=0)

        return {
            "image_sequence": image_seq,
            "meteo_sequence": meteo_seq,
            "target": torch.tensor(float(row[self.target_col]), dtype=torch.float32),
        }


def _cap_samples(df: pd.DataFrame, max_samples: int | None) -> pd.DataFrame:
    if max_samples is None or len(df) <= max_samples:
        return df
    return df.iloc[:max_samples].reset_index(drop=True)


def build_multimodal_datasets(
    dataframe: pd.DataFrame,
    config: DatasetBuildConfig,
) -> tuple[Dict[str, MultimodalForecastDataset], list[str], Dict[str, object]]:
    """
    Build train/val/test datasets from tabular dataframe + image folder.
    """
    df = attach_utc_timestamp(dataframe)
    feature_columns = build_meteorological_feature_columns(df.columns)

    sample_cfg = ForecastSamplingConfig(
        horizons_hours=(config.horizon_hours,),
        meteo_lookback_steps=config.meteo_lookback_steps,
        image_lookback_steps=config.image_lookback_steps,
        target_col=config.target_col,
    )
    sample_index = build_forecast_sample_index(df, config=sample_cfg)

    split_cfg = TimeSplitConfig(
        train_years=config.train_years,
        val_years=config.val_years,
        test_years=config.test_years,
    )
    splits = split_samples_by_label_year(sample_index, config=split_cfg, label_time_col="y_time", verbose=True)
    split_before_cap = {k: len(v) for k, v in splits.items()}
    splits = {k: _cap_samples(v, config.max_samples_per_split) for k, v in splits.items()}

    datasets: Dict[str, MultimodalForecastDataset] = {}
    dropped_missing: Dict[str, int] = {}
    for split_name, split_index in splits.items():
        ds = MultimodalForecastDataset(
            df,
            split_index,
            feature_columns=feature_columns,
            image_dir=config.image_dir,
            image_id_col=config.image_id_col,
            image_size=config.image_size,
            normalize_images=config.normalize_images,
            target_col=config.target_col,
            drop_missing_images=True,
        )
        datasets[split_name] = ds
        dropped_missing[split_name] = ds.filtered_out_missing_images

    metadata = {
        "feature_columns": feature_columns,
        "split_size_before_cap": split_before_cap,
        "split_size_after_cap": {k: len(v) for k, v in splits.items()},
        "dataset_size_after_image_filter": {k: len(v) for k, v in datasets.items()},
        "dropped_missing_images": dropped_missing,
        "horizon_hours": config.horizon_hours,
        "meteo_lookback_steps": config.meteo_lookback_steps,
        "image_lookback_steps": config.image_lookback_steps,
    }
    return datasets, feature_columns, metadata
