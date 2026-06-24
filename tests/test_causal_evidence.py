import unittest

import numpy as np
import pandas as pd

from src.causal_evidence import (
    association,
    measure_edges,
    mediation,
    stratified_stability,
)


def _frame(**cols) -> pd.DataFrame:
    return pd.DataFrame(cols)


class AssociationTests(unittest.TestCase):
    def test_standardized_slope_equals_pearson(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0, 1, 200)
        y = 0.7 * x + rng.normal(0, 0.5, 200)
        stat = association(_frame(x=x, y=y), "x", "y")
        self.assertAlmostEqual(stat["slope_std"], stat["pearson_r"], places=9)
        self.assertLess(stat["p_value"], 0.01)


class MediationTests(unittest.TestCase):
    def _full_mediation_frame(self, n=400, seed=0):
        rng = np.random.default_rng(seed)
        root = rng.normal(0, 1, n)
        mediator = 0.9 * root + rng.normal(0, 0.3, n)
        target = 1.0 * mediator + rng.normal(0, 0.3, n)  # root affects target ONLY via mediator
        return _frame(root=root, mediator=mediator, defect_rate=target)

    def test_indirect_equals_total_minus_direct(self):
        m = mediation(self._full_mediation_frame(), "root", "mediator")
        self.assertAlmostEqual(m["indirect"], m["c_total"] - m["c_direct"], places=6)

    def test_recovers_known_full_mediation(self):
        m = mediation(self._full_mediation_frame(), "root", "mediator")
        self.assertTrue(m["sign_consistent"])
        self.assertGreater(m["prop_mediated"], 0.85)
        self.assertLess(m["prop_mediated"], 1.15)
        self.assertLess(m["indirect_p"], 0.01)
        self.assertFalse(m["unstable"])

    def test_small_n_returns_none(self):
        small = self._full_mediation_frame(n=10)
        self.assertIsNone(mediation(small, "root", "mediator", min_n=30))


class MeasureEdgesTests(unittest.TestCase):
    def _inputs(self, root_step, med_step, n=120, seed=3):
        rng = np.random.default_rng(seed)
        root = rng.normal(0, 1, n)
        mediator = 0.9 * root + rng.normal(0, 0.3, n)
        target = 1.0 * mediator + rng.normal(0, 0.3, n)
        ids = pd.DataFrame({
            "root_lot_id": ["L"] * n,
            "wafer_id": [f"W{i}" for i in range(n)],
        })
        feature_matrix = ids.assign(**{"num|R": root, "num|M": mediator})
        target_df = ids.assign(defect_rate=target, bad_flag=(target > np.median(target)).astype(int))
        feature_dictionary = pd.DataFrame([
            {"feature_id": "num|R", "causal_role": "root_cause_candidate", "process_step": root_step},
            {"feature_id": "num|M", "causal_role": "mediator_candidate", "process_step": med_step},
        ])
        return feature_matrix, target_df, feature_dictionary

    def test_recovers_seeded_chain(self):
        fm, td, fd = self._inputs(root_step="CVD", med_step="CVD")
        med = measure_edges(fm, td, fd)
        self.assertEqual(len(med), 1)
        row = med.iloc[0]
        self.assertEqual((row["root"], row["mediator"]), ("num|R", "num|M"))
        self.assertTrue(bool(row["seeded"]))
        self.assertGreater(abs(row["indirect"]), 0.2)

    def test_discovers_chain_without_name_match(self):
        # Names disagree (root step != mediator step) but the data correlation is strong:
        # the old string-equality edge would never form; the measured engine still recovers it.
        fm, td, fd = self._inputs(root_step="CVD", med_step="MET_CVD")
        med = measure_edges(fm, td, fd)
        self.assertEqual(len(med), 1)
        self.assertFalse(bool(med.iloc[0]["seeded"]))


class StratificationTests(unittest.TestCase):
    def test_flags_simpson_reversal(self):
        rng = np.random.default_rng(7)
        n = 200
        # Within each chamber the root->mediator slope is NEGATIVE, but the chamber
        # means line up so the POOLED association is positive (Simpson's paradox).
        root_a = rng.normal(0, 1, n)
        med_a = -1.0 * root_a + rng.normal(0, 0.3, n)
        tgt_a = 1.0 * med_a + rng.normal(0, 0.3, n)
        root_b = rng.normal(6, 1, n)
        med_b = 10 - 1.0 * root_b + rng.normal(0, 0.3, n)
        tgt_b = 1.0 * med_b + rng.normal(0, 0.3, n)
        frame = pd.DataFrame({
            "root": np.concatenate([root_a, root_b]),
            "mediator": np.concatenate([med_a, med_b]),
            "defect_rate": np.concatenate([tgt_a, tgt_b]),
            "chamber": ["A"] * n + ["B"] * n,
        })
        pooled = mediation(frame, "root", "mediator")
        self.assertGreater(pooled["indirect"], 0)  # pooled looks positive...
        stable = stratified_stability(frame, "root", "mediator", pooled["indirect"], "chamber")
        self.assertFalse(stable)  # ...but it flips inside every chamber


if __name__ == "__main__":
    unittest.main()
