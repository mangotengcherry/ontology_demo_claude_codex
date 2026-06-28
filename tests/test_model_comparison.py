"""Regression tests for the performance arm (scripts/model_comparison_demo.py).

Kept fast: small wafer count + few CatBoost iterations. Runs in an isolated temp
cwd so generated input/ and data/ never touch the repo tree.
"""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import model_comparison_demo as mc  # noqa: E402


class ModelComparisonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._cwd = os.getcwd()
        cls._tmp = tempfile.mkdtemp(prefix="mc_test_")
        os.chdir(cls._tmp)
        cls.ds = mc.load_dataset(
            input_dir=os.path.join(cls._tmp, "input"),
            bad_quantile=0.80, n_wafers=160, seed=7,
        )

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._cwd)

    def test_roles_assigned(self):
        roles = set(self.ds.roles.values())
        self.assertIn("leakage_or_post_outcome", roles)
        self.assertIn("mediator_candidate", roles)
        self.assertIn("root_cause_candidate", roles)

    def test_leakage_dropped_by_role(self):
        all_cols, _ = mc.select_columns(self.ds, drop_leakage=False)
        kept, _ = mc.select_columns(self.ds, drop_leakage=True)
        leaked = [c for c in all_cols if self.ds.roles.get(c) == mc.LEAKAGE_ROLE]
        self.assertTrue(leaked, "virtual data should contain a leakage proxy")
        for c in leaked:
            self.assertNotIn(c, kept)

    def test_measured_chain_present_and_sane(self):
        med = self.ds.mediation
        self.assertFalse(med.empty, "mediation chain should be measured on virtual data")
        top = med.sort_values("indirect", key=lambda s: s.abs(), ascending=False).iloc[0]
        self.assertTrue(bool(top["sign_consistent"]))
        self.assertGreater(abs(top["a"]), 0.3)

    def test_monotone_only_a_grade(self):
        cols, _ = mc.select_columns(self.ds, drop_leakage=True)
        mono = mc.monotone_map(self.ds, cols, self.ds.X[cols], self.ds.y)
        for feat in mono:
            self.assertEqual(str(self.ds.grades.get(feat, "")).upper(), "A")
            self.assertIn(mono[feat], (-1, 1))

    def test_leakage_inflates_flat_r2(self):
        """Flat-with-leakage test R2 should exceed the leakage-free ceiling."""
        from sklearn.model_selection import train_test_split

        idx = np.arange(len(self.ds.X))
        tr, te = train_test_split(idx, test_size=0.25, random_state=7)
        full_cols, full_cats = mc.select_columns(self.ds, drop_leakage=False)
        nl_cols, nl_cats = mc.select_columns(self.ds, drop_leakage=True)
        _, p_full, _ = mc.fit_predict(
            self.ds.X.iloc[tr][full_cols], self.ds.y.iloc[tr], self.ds.X.iloc[te][full_cols],
            full_cats, mc.FLAT_DEPTH, 80, 7)
        _, p_nl, _ = mc.fit_predict(
            self.ds.X.iloc[tr][nl_cols], self.ds.y.iloc[tr], self.ds.X.iloc[te][nl_cols],
            nl_cats, mc.FLAT_DEPTH, 80, 7)
        r2_full = mc._r2(self.ds.y.iloc[te], p_full)
        r2_nl = mc._r2(self.ds.y.iloc[te], p_nl)
        self.assertGreater(r2_full, r2_nl)


if __name__ == "__main__":
    unittest.main()
