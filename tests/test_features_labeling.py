import unittest

import pandas as pd

from src.features.labeling import (
    ForecastSamplingConfig,
    build_forecast_sample_index,
    build_meteorological_feature_columns,
)


def _build_synthetic_weather(rows: int = 80) -> pd.DataFrame:
    timestamps = pd.date_range("2013-01-01 14:00:00", periods=rows, freq="h")
    data = {
        "Date_UTC": [ts.strftime("%Y-%m-%d") for ts in timestamps],
        "Time_UTC": [ts.strftime("%H:%M") for ts in timestamps],
        "Temp_F": [30.0 + (i % 10) for i in range(rows)],
        "RH_pct": [70.0 + (i % 20) for i in range(rows)],
        "Dewpt_F": [25.0 + (i % 8) for i in range(rows)],
        "Wind_Spd_mph": [5.0 + (i % 6) for i in range(rows)],
        "Wind_Direction_deg": [float((i * 10) % 360) for i in range(rows)],
        "Peak_Wind_Gust_mph": [8.0 + (i % 7) for i in range(rows)],
        "Low_Cloud_Ht_ft": [1000.0 + (i % 5) * 200 for i in range(rows)],
        "Med_Cloud_Ht_ft": [3000.0 + (i % 5) * 300 for i in range(rows)],
        "High_Cloud_Ht_ft": [7000.0 + (i % 5) * 500 for i in range(rows)],
        "Visibility_mi": [10.0 for _ in range(rows)],
        "Atm_Press_hPa": [1005.0 + (i % 4) for i in range(rows)],
        "Sea_Lev_Press_hPa": [1008.0 + (i % 4) for i in range(rows)],
        "Altimeter_hPa": [1007.0 + (i % 4) for i in range(rows)],
        "Precip_in": [0.2 if (i % 9 == 0) else 0.0 for i in range(rows)],
    }
    return pd.DataFrame(data)


class TestFeatureLabeling(unittest.TestCase):
    def test_forecast_index_temporal_order(self) -> None:
        df = _build_synthetic_weather()
        cfg = ForecastSamplingConfig(
            horizons_hours=(6,),
            meteo_lookback_steps=12,
            image_lookback_steps=4,
            target_col="target_rain",
        )
        sample_index = build_forecast_sample_index(df, config=cfg)
        self.assertGreater(len(sample_index), 0)
        self.assertTrue((sample_index["anchor_time"] < sample_index["y_time"]).all())
        self.assertTrue((sample_index["meteo_end_idx"] < sample_index["y_idx"]).all())
        self.assertTrue((sample_index["image_end_idx"] < sample_index["y_idx"]).all())

    def test_precip_column_excluded_from_features(self) -> None:
        df = _build_synthetic_weather()
        feature_cols = build_meteorological_feature_columns(df.columns)
        self.assertNotIn("Precip_in", feature_cols)


if __name__ == "__main__":
    unittest.main()
