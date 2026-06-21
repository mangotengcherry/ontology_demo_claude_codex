import os
import tempfile
import unittest

import pandas as pd

from src.real_dataset_adapter import build_standard_dataset


ROOT_FEATURE = "num|PRC1|erd|pcf|RF_factor_avg|ETCHER"
METRO_FEATURE = "num|MET1|CD1|AVG"
VM_FEATURE = "num|VM1|RF_TIME_1|VALUE"
PPID_FEATURE = "cat|ppid|PRC1"
EQP_FEATURE = "cat|eqp|PRC1"
CHAMBER_FEATURE = "cat|eqp_ch|PRC1"


class RealDatasetAdapterTests(unittest.TestCase):
    def test_builds_standard_dataset_from_bad_wafer_mean_shap_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = os.path.join(tmp, "input")
            output_dir = os.path.join(tmp, "data")
            os.makedirs(input_dir)

            raw = pd.DataFrame(
                [
                    {
                        "root_lot_id": "L1",
                        "wafer_id": "W1",
                        "tkout_time": "2026-05-01 08:00:00",
                        "target": 1.0,
                        PPID_FEATURE: "P_A",
                        EQP_FEATURE: "EQ_A",
                        CHAMBER_FEATURE: "CH_A",
                        ROOT_FEATURE: 0.1,
                        METRO_FEATURE: 10.0,
                        VM_FEATURE: 2.0,
                    },
                    {
                        "root_lot_id": "L1",
                        "wafer_id": "W2",
                        "tkout_time": "2026-05-01 08:10:00",
                        "target": 2.0,
                        PPID_FEATURE: "P_A",
                        EQP_FEATURE: "EQ_A",
                        CHAMBER_FEATURE: "CH_A",
                        ROOT_FEATURE: 0.2,
                        METRO_FEATURE: 11.0,
                        VM_FEATURE: 2.5,
                    },
                    {
                        "root_lot_id": "L2",
                        "wafer_id": "W3",
                        "tkout_time": "2026-05-01 08:20:00",
                        "target": 3.0,
                        PPID_FEATURE: "P_B",
                        EQP_FEATURE: "EQ_B",
                        CHAMBER_FEATURE: "CH_B",
                        ROOT_FEATURE: 0.3,
                        METRO_FEATURE: 12.0,
                        VM_FEATURE: 3.0,
                    },
                    {
                        "root_lot_id": "L2",
                        "wafer_id": "W4",
                        "tkout_time": "2026-05-01 08:30:00",
                        "target": 9.0,
                        PPID_FEATURE: "P_B",
                        EQP_FEATURE: "EQ_B",
                        CHAMBER_FEATURE: "CH_B",
                        ROOT_FEATURE: 2.0,
                        METRO_FEATURE: 20.0,
                        VM_FEATURE: 5.0,
                    },
                ]
            )
            raw.to_csv(os.path.join(input_dir, "raw_data.csv"), index=False)
            pd.DataFrame(
                [
                    {"feature": METRO_FEATURE, "shap_value": 0.8},
                    {"feature": ROOT_FEATURE, "shap_value": 0.3},
                    {"feature": PPID_FEATURE, "shap_value": 0.1},
                ]
            ).to_csv(os.path.join(input_dir, "x_feature_shap_value.csv"), index=False)
            pd.DataFrame(
                [
                    {
                        "prc_step": "PRC1",
                        "metro_step": "MET1",
                        "metro_item": "CD1",
                        "subitem_id": "AVG",
                        "metro_grade": "A",
                    }
                ]
            ).to_csv(os.path.join(input_dir, "prc_metro_relation.csv"), index=False)

            artifacts = build_standard_dataset(input_dir, output_dir, bad_quantile=0.75)

            target = pd.read_csv(os.path.join(output_dir, "target.csv"))
            self.assertEqual(target["bad_flag"].sum(), 1)
            self.assertIn("defect_rate", target.columns)

            shap_values = pd.read_csv(os.path.join(output_dir, "shap_values.csv"))
            self.assertEqual(set(shap_values["feature_id"]), {METRO_FEATURE, ROOT_FEATURE, PPID_FEATURE})
            self.assertEqual(set(shap_values["shap_scope"]), {"bad_wafer_mean"})
            self.assertEqual(
                shap_values.set_index("feature_id").loc[METRO_FEATURE, "bad_good_separation_simple"],
                artifacts.shap_values.set_index("feature_id").loc[METRO_FEATURE, "bad_good_separation_simple"],
            )

            feature_dict = pd.read_csv(os.path.join(output_dir, "feature_dictionary.csv"))
            role_by_feature = feature_dict.set_index("feature_id")["causal_role"].to_dict()
            self.assertEqual(role_by_feature[METRO_FEATURE], "mediator_candidate")
            self.assertEqual(role_by_feature[ROOT_FEATURE], "root_cause_candidate")
            self.assertEqual(role_by_feature[PPID_FEATURE], "confounder")

            edges = pd.read_csv(os.path.join(output_dir, "causal_edges.csv"), comment="#")
            edge_pairs = set(zip(edges["source_feature"], edges["target_feature"], edges["relation"]))
            self.assertIn((ROOT_FEATURE, METRO_FEATURE, "physicallyAffects"), edge_pairs)
            self.assertIn((METRO_FEATURE, "defect_rate", "mediates"), edge_pairs)

            history = pd.read_csv(os.path.join(output_dir, "process_history.csv"))
            w4_history = history[history["wafer_id"] == "W4"].iloc[0]
            self.assertEqual(w4_history["ppid"], "P_B")
            self.assertEqual(w4_history["chamber_id"], "CH_B")


if __name__ == "__main__":
    unittest.main()
