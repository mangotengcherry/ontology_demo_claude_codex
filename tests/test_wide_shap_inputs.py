"""Tests for the wide-form per-wafer SHAP input contract (bad/all wafer SHAP)."""
import os
import tempfile
import unittest

import pandas as pd

from src import shap_inputs
from src.real_dataset_adapter import build_standard_dataset, input_files_exist


ROOT_FEATURE = "num|PRC1|erd|pcf|RF_factor_avg|ETCHER"
METRO_FEATURE = "num|MET1|CD1|AVG"
PPID_FEATURE = "cat|ppid|PRC1"
CHAMBER_FEATURE = "cat|eqp_ch|PRC1"


def _raw():
    return pd.DataFrame(
        [
            {"root_lot_id": "L1", "wafer_id": "W1", "tkout_time": "t", "target": 1.0,
             PPID_FEATURE: "P_A", CHAMBER_FEATURE: "CH_A", ROOT_FEATURE: 0.1, METRO_FEATURE: 10.0},
            {"root_lot_id": "L1", "wafer_id": "W2", "tkout_time": "t", "target": 2.0,
             PPID_FEATURE: "P_A", CHAMBER_FEATURE: "CH_A", ROOT_FEATURE: 0.15, METRO_FEATURE: 11.0},
            {"root_lot_id": "L2", "wafer_id": "W3", "tkout_time": "t", "target": 3.0,
             PPID_FEATURE: "P_B", CHAMBER_FEATURE: "CH_B", ROOT_FEATURE: 0.2, METRO_FEATURE: 12.0},
            {"root_lot_id": "L2", "wafer_id": "W4", "tkout_time": "t", "target": 9.0,
             PPID_FEATURE: "P_B", CHAMBER_FEATURE: "CH_B", ROOT_FEATURE: 0.8, METRO_FEATURE: 20.0},
        ]
    )


def _relation():
    return pd.DataFrame(
        [{"prc_step": "PRC1", "metro_step": "MET1", "metro_item": "CD1",
          "subitem_id": "AVG", "metro_grade": "A"}]
    )


def _all_wide(ids="pair", extra_meta=False):
    rows = [
        {ROOT_FEATURE: 0.10, METRO_FEATURE: 0.30, PPID_FEATURE: 0.05},
        {ROOT_FEATURE: 0.15, METRO_FEATURE: 0.40, PPID_FEATURE: 0.05},
        {ROOT_FEATURE: 0.20, METRO_FEATURE: 0.50, PPID_FEATURE: 0.06},
        {ROOT_FEATURE: 0.80, METRO_FEATURE: 1.60, PPID_FEATURE: 0.20},
    ]
    keys = [("L1", "W1"), ("L1", "W2"), ("L2", "W3"), ("L2", "W4")]
    for r, (lot, waf) in zip(rows, keys):
        if ids == "pair":
            r["root_lot_id"], r["wafer_id"] = lot, waf
        else:
            r["root_lot_wafer_id"] = f"{lot}|{waf}"
        if extra_meta:
            r["base_value"] = 5.0
            r["prediction"] = 7.0
            r["target"] = 1.0
    return pd.DataFrame(rows)


def _bad_wide(ids="pair", extra_meta=False):
    return _all_wide(ids=ids, extra_meta=extra_meta).iloc[[3]].reset_index(drop=True)


class WideShapAdapterTests(unittest.TestCase):
    def _setup(self, input_dir, *, bad=True, allw=True, legacy=False, ids="pair", extra_meta=False):
        _raw().to_csv(os.path.join(input_dir, "raw_data.csv"), index=False)
        _relation().to_csv(os.path.join(input_dir, "prc_metro_relation.csv"), index=False)
        if allw:
            _all_wide(ids=ids, extra_meta=extra_meta).to_csv(
                os.path.join(input_dir, "all_wafer_shap_value.csv"), index=False)
        if bad:
            _bad_wide(ids=ids, extra_meta=extra_meta).to_csv(
                os.path.join(input_dir, "bad_wafer_shap_value.csv"), index=False)
        if legacy:
            pd.DataFrame([
                {"feature": METRO_FEATURE, "shap_value": 99.0},   # deliberately wrong: wide must win
                {"feature": ROOT_FEATURE, "shap_value": 99.0},
            ]).to_csv(os.path.join(input_dir, "x_feature_shap_value.csv"), index=False)

    def test_wide_bad_and_all_build_cohort_mean_and_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            inp = os.path.join(tmp, "input"); out = os.path.join(tmp, "data")
            os.makedirs(inp)
            self._setup(inp)
            self.assertTrue(input_files_exist(inp))

            art = build_standard_dataset(inp, out, bad_quantile=0.75)

            sv = pd.read_csv(os.path.join(out, "shap_values.csv"))
            self.assertEqual(set(sv["shap_scope"]), {"bad_wafer_mean"})
            self.assertEqual(set(sv["feature_id"]), {ROOT_FEATURE, METRO_FEATURE, PPID_FEATURE})
            # mean|SHAP| importance from the bad cohort (W4 row): METRO=1.6, ROOT=0.8
            abs_by_feat = sv.set_index("feature_id")["abs_shap_value"].to_dict()
            self.assertAlmostEqual(abs_by_feat[METRO_FEATURE], 1.6, places=5)
            self.assertAlmostEqual(abs_by_feat[ROOT_FEATURE], 0.8, places=5)

            cmp_path = os.path.join(out, "shap_cohort_comparison.csv")
            self.assertTrue(os.path.exists(cmp_path))
            comp = pd.read_csv(cmp_path)
            for col in ["feature_id", "bad_mean_abs_shap", "good_mean_abs_shap",
                        "abs_shap_gap_bad_minus_good", "causal_role"]:
                self.assertIn(col, comp.columns)
            top = comp.iloc[0]
            self.assertEqual(top["feature_id"], METRO_FEATURE)       # largest bad-vs-good gap
            self.assertEqual(top["causal_role"], "mediator_candidate")
            # good cohort mean|SHAP| for METRO = mean(0.3,0.4,0.5) = 0.4
            self.assertAlmostEqual(
                comp.set_index("feature_id").loc[METRO_FEATURE, "good_mean_abs_shap"], 0.4, places=5)
            self.assertIsNotNone(art.shap_cohort_comparison)

    def test_combined_root_lot_wafer_id_form(self):
        with tempfile.TemporaryDirectory() as tmp:
            inp = os.path.join(tmp, "input"); out = os.path.join(tmp, "data")
            os.makedirs(inp)
            self._setup(inp, ids="combined")
            build_standard_dataset(inp, out, bad_quantile=0.75)
            comp = pd.read_csv(os.path.join(out, "shap_cohort_comparison.csv"))
            self.assertEqual(comp.iloc[0]["feature_id"], METRO_FEATURE)
            self.assertAlmostEqual(
                comp.set_index("feature_id").loc[METRO_FEATURE, "bad_mean_abs_shap"], 1.6, places=5)

    def test_all_only_derives_bad_cohort_from_bad_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            inp = os.path.join(tmp, "input"); out = os.path.join(tmp, "data")
            os.makedirs(inp)
            self._setup(inp, bad=False, allw=True)
            self.assertTrue(input_files_exist(inp))
            build_standard_dataset(inp, out, bad_quantile=0.75)
            sv = pd.read_csv(os.path.join(out, "shap_values.csv")).set_index("feature_id")
            # bad cohort = W4 row only, derived from target bad_flag
            self.assertAlmostEqual(sv.loc[METRO_FEATURE, "abs_shap_value"], 1.6, places=5)

    def test_wide_takes_precedence_over_legacy(self):
        with tempfile.TemporaryDirectory() as tmp:
            inp = os.path.join(tmp, "input"); out = os.path.join(tmp, "data")
            os.makedirs(inp)
            self._setup(inp, legacy=True)
            build_standard_dataset(inp, out, bad_quantile=0.75)
            sv = pd.read_csv(os.path.join(out, "shap_values.csv")).set_index("feature_id")
            self.assertAlmostEqual(sv.loc[METRO_FEATURE, "abs_shap_value"], 1.6, places=5)  # not 99

    def test_meta_columns_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            inp = os.path.join(tmp, "input"); out = os.path.join(tmp, "data")
            os.makedirs(inp)
            self._setup(inp, extra_meta=True)
            build_standard_dataset(inp, out, bad_quantile=0.75)
            sv = pd.read_csv(os.path.join(out, "shap_values.csv"))
            for meta in ["base_value", "prediction", "target"]:
                self.assertNotIn(meta, set(sv["feature_id"]))


class ShapInputsUnitTests(unittest.TestCase):
    def test_detect_feature_columns_prefers_raw_intersection(self):
        df = pd.DataFrame({"root_lot_id": ["a"], "wafer_id": ["b"],
                           ROOT_FEATURE: [0.1], "base_value": [1.0], "other_num": [2.0]})
        feats = shap_inputs.detect_feature_columns(df, [ROOT_FEATURE, METRO_FEATURE])
        self.assertEqual(feats, [ROOT_FEATURE])

    def test_detect_feature_columns_fallback_drops_meta(self):
        df = pd.DataFrame({"root_lot_id": ["a"], "wafer_id": ["b"],
                           "f1": [0.1], "f2": [0.2], "prediction": [3.0]})
        feats = shap_inputs.detect_feature_columns(df, raw_feature_cols=None)
        self.assertEqual(set(feats), {"f1", "f2"})

    def test_cohort_mean_decouples_signed_and_abs(self):
        # signed mean cancels to ~0 but mean|SHAP| stays large (bidirectional feature).
        wide = pd.DataFrame({"f": [1.0, -1.0, 1.0, -1.0]})
        out = shap_inputs.cohort_mean_shap(wide, ["f"]).set_index("feature")
        self.assertAlmostEqual(out.loc["f", "shap_value"], 0.0, places=6)
        self.assertAlmostEqual(out.loc["f", "mean_abs_shap"], 1.0, places=6)

    def test_cohort_comparison_gap_sign(self):
        wide = pd.DataFrame({
            "root_lot_id": ["L1", "L1", "L2"], "wafer_id": ["W1", "W2", "W3"],
            "f": [0.1, 0.2, 2.0],
        })
        bad_keys = {"L2|W3"}
        comp = shap_inputs.cohort_comparison(wide, ["f"], bad_keys).set_index("feature_id")
        self.assertAlmostEqual(comp.loc["f", "bad_mean_abs_shap"], 2.0, places=6)
        self.assertGreater(comp.loc["f", "abs_shap_gap_bad_minus_good"], 0)


if __name__ == "__main__":
    unittest.main()
