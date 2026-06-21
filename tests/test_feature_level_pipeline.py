import os
import tempfile
import unittest

import pandas as pd

from src.hypothesis_engine import build_hypothesis_cards, naive_vs_ontology_summary
from src.loaders import load_all_data
from src.ontology_mapping import map_shap_to_feature_dictionary
from src.pipeline import run_real_dataset_pipeline
from src.real_dataset_adapter import build_standard_dataset
from src.reporting import write_markdown_report
from src.shap_interpreter import aggregate_shap_by_context


ROOT_FEATURE = "num|PRC1|erd|pcf|RF_factor_avg|ETCHER"
METRO_FEATURE = "num|MET1|CD1|AVG"


class FeatureLevelPipelineTests(unittest.TestCase):
    def _write_input_files(self, input_dir):
        pd.DataFrame(
            [
                {
                    "root_lot_id": "L1",
                    "wafer_id": "W1",
                    "tkout_time": "2026-05-01 08:00:00",
                    "target": 1.0,
                    "cat|ppid|PRC1": "P_A",
                    "cat|eqp_ch|PRC1": "CH_A",
                    ROOT_FEATURE: 0.1,
                    METRO_FEATURE: 10.0,
                },
                {
                    "root_lot_id": "L2",
                    "wafer_id": "W2",
                    "tkout_time": "2026-05-01 08:10:00",
                    "target": 9.0,
                    "cat|ppid|PRC1": "P_B",
                    "cat|eqp_ch|PRC1": "CH_B",
                    ROOT_FEATURE: 2.0,
                    METRO_FEATURE: 20.0,
                },
            ]
        ).to_csv(os.path.join(input_dir, "raw_data.csv"), index=False)
        pd.DataFrame(
            [
                {"feature": METRO_FEATURE, "shap_value": 1.2},
                {"feature": ROOT_FEATURE, "shap_value": 0.5},
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

    def test_feature_level_shap_builds_cards_summary_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = os.path.join(tmp, "input")
            data_dir = os.path.join(tmp, "data")
            output_dir = os.path.join(tmp, "outputs")
            os.makedirs(input_dir)
            os.makedirs(output_dir)

            self._write_input_files(input_dir)

            build_standard_dataset(input_dir, data_dir, bad_quantile=0.5)
            data = load_all_data(data_dir)
            mapped = map_shap_to_feature_dictionary(data["shap_values"], data["feature_dictionary"])

            cards = build_hypothesis_cards(
                mapped,
                data["target"],
                data["process_history"],
                data["causal_edges"],
                data["feature_dictionary"],
                top_n=3,
            )
            self.assertFalse(cards.empty)
            self.assertIn(ROOT_FEATURE, cards.iloc[0]["root_cause_candidate_features"])
            self.assertIn(METRO_FEATURE, cards.iloc[0]["mediator_candidate_features"])
            self.assertEqual(cards.iloc[0]["bad_wafer_group_size"], 1)

            summary = naive_vs_ontology_summary(
                mapped,
                data["target"],
                data["causal_edges"],
                data["feature_dictionary"],
                data["ground_truth"],
            )
            self.assertEqual(summary["naive_top_nonleak"], METRO_FEATURE)
            self.assertEqual(summary["ontology_root"], ROOT_FEATURE)

            ontology_summary = aggregate_shap_by_context(
                mapped,
                ["causal_role"],
                wafer_ids=data["target"].loc[data["target"]["bad_flag"] == 1, "wafer_id"].tolist(),
            )
            report_path = os.path.join(output_dir, "report.md")
            write_markdown_report(data, mapped, cards, ontology_summary, report_path)

            with open(report_path, encoding="utf-8") as f:
                report = f.read()
            self.assertIn("Bad wafer cohort SHAP", report)
            self.assertIn(ROOT_FEATURE, report)
            self.assertIn(METRO_FEATURE, report)

    def test_run_real_dataset_pipeline_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = os.path.join(tmp, "input")
            data_dir = os.path.join(tmp, "data")
            output_dir = os.path.join(tmp, "outputs")
            os.makedirs(input_dir)
            self._write_input_files(input_dir)

            result = run_real_dataset_pipeline(input_dir, data_dir, output_dir, bad_quantile=0.5)

            self.assertEqual(result["mode"], "real")
            self.assertTrue(os.path.exists(os.path.join(output_dir, "hypothesis_cards.csv")))
            self.assertTrue(os.path.exists(os.path.join(output_dir, "ontology_level_shap_summary.csv")))
            self.assertTrue(os.path.exists(os.path.join(output_dir, "report.md")))
            cards = pd.read_csv(os.path.join(output_dir, "hypothesis_cards.csv"))
            self.assertIn(ROOT_FEATURE, cards.iloc[0]["root_cause_candidate_features"])


if __name__ == "__main__":
    unittest.main()
