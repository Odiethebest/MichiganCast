import unittest

import pandas as pd

from src.data.contracts import REQUIRED_COLUMNS, validate_dataframe_against_contract


def _valid_row() -> dict:
    return {
        "Date_UTC": "2014-01-01",
        "Time_UTC": "14:00",
        "Temp_F": 32.0,
        "RH_pct": 80.0,
        "Dewpt_F": 28.0,
        "Wind_Spd_mph": 8.0,
        "Wind_Direction_deg": 180.0,
        "Peak_Wind_Gust_mph": 12.0,
        "Low_Cloud_Ht_ft": 2000.0,
        "Med_Cloud_Ht_ft": 4000.0,
        "High_Cloud_Ht_ft": 8000.0,
        "Visibility_mi": 10.0,
        "Atm_Press_hPa": 1010.0,
        "Sea_Lev_Press_hPa": 1012.0,
        "Altimeter_hPa": 1011.0,
        "Precip_in": 0.0,
    }


class TestDataContracts(unittest.TestCase):
    def test_contract_passes_on_valid_data(self) -> None:
        df = pd.DataFrame([_valid_row(), _valid_row()])
        report = validate_dataframe_against_contract(df)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["missing_required_columns"], [])
        self.assertEqual(report["failed_checks"], [])

    def test_contract_fails_on_out_of_range_value(self) -> None:
        row = _valid_row()
        row["Altimeter_hPa"] = 700.0
        df = pd.DataFrame([row])
        report = validate_dataframe_against_contract(df)
        self.assertEqual(report["status"], "fail")
        failed_checks = {item["check"] for item in report["failed_checks"]}
        self.assertIn("numeric_rule::Altimeter_hPa", failed_checks)

    def test_required_columns_covered(self) -> None:
        self.assertEqual(set(REQUIRED_COLUMNS), set(_valid_row().keys()))


if __name__ == "__main__":
    unittest.main()
