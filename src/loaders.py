"""Load and lightly validate the standard ontology-SHAP CSV artifacts."""
from __future__ import annotations

import os
from typing import Dict

import pandas as pd

REQUIRED_COLUMNS: Dict[str, list] = {
    "feature_matrix": ["root_lot_id", "wafer_id"],
    "target": ["root_lot_id", "wafer_id", "defect_rate", "bad_flag"],
    "shap_values": ["root_lot_id", "wafer_id", "target_id", "feature_id",
                    "feature_value", "shap_value", "abs_shap_value", "shap_direction"],
    "feature_dictionary": ["feature_id", "causal_role", "leakage_risk", "mechanism_group"],
    "causal_edges": ["source_feature", "target_feature", "relation", "confidence"],
    "process_history": ["wafer_id", "process_step", "chamber_id", "ppid"],
    "engineer_feedback": ["hypothesis_id", "engineer_judgment", "action_status"],
    "ground_truth": ["true_root_feature", "true_chain_text", "true_chamber"],
}


def load_all_data(data_dir: str = "data") -> Dict[str, pd.DataFrame]:
    """Load every CSV into a dict keyed by logical name; validate required columns."""
    data: Dict[str, pd.DataFrame] = {}
    for name, required in REQUIRED_COLUMNS.items():
        path = os.path.join(data_dir, f"{name}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {path}. Run `python3 run_demo.py` first to build standard CSV artifacts."
            )
        # causal_edges.csv has a leading '#' ontology-market comment line.
        df = pd.read_csv(path, comment="#") if name == "causal_edges" else pd.read_csv(path)
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        data[name] = df

    # Optional convenience artifact (model parity); not strictly required.
    pred_path = os.path.join(data_dir, "model_predictions.csv")
    if os.path.exists(pred_path):
        data["model_predictions"] = pd.read_csv(pred_path)

    # Optional measured-mediation evidence (built by the real-dataset adapter).
    med_path = os.path.join(data_dir, "mediation.csv")
    if os.path.exists(med_path):
        data["mediation"] = pd.read_csv(med_path)

    # Optional good-vs-bad SHAP cohort comparison (from all_wafer_shap_value.csv).
    cmp_path = os.path.join(data_dir, "shap_cohort_comparison.csv")
    if os.path.exists(cmp_path):
        data["shap_cohort_comparison"] = pd.read_csv(cmp_path)
    return data
