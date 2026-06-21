import os
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

from src.virtual_data_generator import generate_virtual_input_dataset


class VirtualModeTests(unittest.TestCase):
    def test_generate_virtual_input_dataset_writes_real_input_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            generate_virtual_input_dataset(tmp, n_wafers=40)

            raw = pd.read_csv(os.path.join(tmp, "raw_data.csv"))
            shap = pd.read_csv(os.path.join(tmp, "x_feature_shap_value.csv"))
            relation = pd.read_csv(os.path.join(tmp, "prc_metro_relation.csv"))

            self.assertIn("root_lot_id", raw.columns)
            self.assertIn("wafer_id", raw.columns)
            self.assertIn("target", raw.columns)
            self.assertIn("num|CVD|erd|pcf|PRESSURE_SLOPE_P95|CVD", raw.columns)
            self.assertEqual(set(shap.columns), {"feature", "shap_value"})
            self.assertIn("num|MET_CVD|THK_EDGE|AVG", set(shap["feature"]))
            self.assertEqual(
                list(relation.columns),
                ["prc_step", "metro_step", "metro_item", "subitem_id", "metro_grade"],
            )

    def test_run_demo_virtual_mode_generates_inputs_and_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = os.path.join(tmp, "input")
            data_dir = os.path.join(tmp, "data")
            output_dir = os.path.join(tmp, "outputs")

            result = subprocess.run(
                [
                    sys.executable,
                    "run_demo.py",
                    "--mode",
                    "virtual",
                    "--input-dir",
                    input_dir,
                    "--data-dir",
                    data_dir,
                    "--output-dir",
                    output_dir,
                ],
                cwd=os.getcwd(),
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(os.path.exists(os.path.join(input_dir, "raw_data.csv")))
            self.assertTrue(os.path.exists(os.path.join(output_dir, "hypothesis_cards.csv")))
            self.assertTrue(os.path.exists(os.path.join(output_dir, "report.md")))
            self.assertIn("ONTOLOGY-SHAP VIRTUAL DATASET RUN", result.stdout)


if __name__ == "__main__":
    unittest.main()
