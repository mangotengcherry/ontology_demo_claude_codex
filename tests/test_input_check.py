"""Tests for the input preflight validator (src/input_check.py)."""
import os
import tempfile
import unittest

import pandas as pd

from src.input_check import check_inputs


ROOT_FEATURE = "num|PRC1|erd|pcf|RF|ETCHER"
METRO_FEATURE = "num|MET1|CD1|AVG"


def _raw():
    return pd.DataFrame([
        {"root_lot_id": "L1", "wafer_id": "W1", "tkout_time": "t", "target": 1.0,
         ROOT_FEATURE: 0.1, METRO_FEATURE: 10.0},
        {"root_lot_id": "L2", "wafer_id": "W2", "tkout_time": "t", "target": 9.0,
         ROOT_FEATURE: 2.0, METRO_FEATURE: 20.0},
    ])


def _relation():
    return pd.DataFrame([{"prc_step": "PRC1", "metro_step": "MET1", "metro_item": "CD1",
                          "subitem_id": "AVG", "metro_grade": "A"}])


class InputCheckTests(unittest.TestCase):
    def _write(self, d, raw=True, rel=True, allwide=True, bad=None, allwide_bad_names=False):
        if raw:
            _raw().to_csv(os.path.join(d, "raw_data.csv"), index=False)
        if rel:
            _relation().to_csv(os.path.join(d, "prc_metro_relation.csv"), index=False)
        if allwide:
            cols = {"root_lot_id": ["L1", "L2"], "wafer_id": ["W1", "W2"]}
            if allwide_bad_names:
                cols["WRONG_NAME_1"] = [0.1, 0.2]
            else:
                cols[ROOT_FEATURE] = [0.1, 0.2]
                cols[METRO_FEATURE] = [0.3, 1.6]
            pd.DataFrame(cols).to_csv(os.path.join(d, "all_wafer_shap_value.csv"), index=False)
        if bad is not None:
            bad.to_csv(os.path.join(d, "bad_wafers.csv"), index=False)

    def test_ok_when_contract_satisfied(self):
        with tempfile.TemporaryDirectory() as d:
            self._write(d, bad=pd.DataFrame([{"root_lot_id": "L2", "wafer_id": "W2"}]))
            r = check_inputs(d, require_shap=True)
            self.assertTrue(r["ok"], r["errors"])
            self.assertTrue(any("리스트 사용" in m for m in r["info"]))

    def test_missing_raw_is_error(self):
        with tempfile.TemporaryDirectory() as d:
            self._write(d, raw=False)
            r = check_inputs(d)
            self.assertFalse(r["ok"])
            self.assertTrue(any("raw_data.csv" in e for e in r["errors"]))

    def test_wide_shap_name_mismatch_is_error(self):
        with tempfile.TemporaryDirectory() as d:
            self._write(d, allwide_bad_names=True)
            r = check_inputs(d, require_shap=True)
            self.assertFalse(r["ok"])
            self.assertTrue(any("일치하지 않" in e for e in r["errors"]))

    def test_missing_shap_ok_for_mode1(self):
        with tempfile.TemporaryDirectory() as d:
            self._write(d, allwide=False)
            self.assertTrue(check_inputs(d, require_shap=False)["ok"])
            self.assertFalse(check_inputs(d, require_shap=True)["ok"])

    def test_bad_wafer_not_in_raw_is_error(self):
        with tempfile.TemporaryDirectory() as d:
            self._write(d, bad=pd.DataFrame([{"root_lot_id": "ZZ", "wafer_id": "W9"}]))
            r = check_inputs(d)
            self.assertFalse(r["ok"])
            self.assertTrue(any("없는 wafer" in e for e in r["errors"]))


if __name__ == "__main__":
    unittest.main()
