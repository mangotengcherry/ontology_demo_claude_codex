"""Reusable execution pipeline for ontology-SHAP datasets."""
from __future__ import annotations

import os
from typing import Dict

import pandas as pd

from src.loaders import load_all_data
from src.ontology_mapping import map_shap_to_feature_dictionary
from src.real_dataset_adapter import build_standard_dataset
from src.reporting import write_markdown_report
from src.shap_interpreter import aggregate_shap_by_context
from src.hypothesis_engine import build_hypothesis_cards, naive_vs_ontology_summary


def run_analysis(data_dir: str = "data", output_dir: str = "outputs") -> Dict:
    """Load standard artifacts, build cards, summaries, and a Markdown report."""
    data = load_all_data(data_dir)
    mapped = map_shap_to_feature_dictionary(data["shap_values"], data["feature_dictionary"])
    cards = build_hypothesis_cards(
        mapped,
        data["target"],
        data["process_history"],
        data["causal_edges"],
        data["feature_dictionary"],
        top_n=5,
        mediation_df=data.get("mediation"),
    )
    bad_wafers = data["target"].loc[data["target"]["bad_flag"] == 1, "wafer_id"].tolist()
    ontology_summary = _ontology_summary(mapped, bad_wafers)
    summary = naive_vs_ontology_summary(
        mapped,
        data["target"],
        data["causal_edges"],
        data["feature_dictionary"],
        data["ground_truth"],
    )

    os.makedirs(output_dir, exist_ok=True)
    cards.to_csv(os.path.join(output_dir, "hypothesis_cards.csv"), index=False)
    ontology_summary.to_csv(os.path.join(output_dir, "ontology_level_shap_summary.csv"), index=False)
    write_markdown_report(data, mapped, cards, ontology_summary, os.path.join(output_dir, "report.md"))
    return {
        "data": data,
        "mapped": mapped,
        "cards": cards,
        "ontology_summary": ontology_summary,
        "summary": summary,
        "output_dir": output_dir,
    }


def run_real_dataset_pipeline(
    input_dir: str = "input",
    data_dir: str = "data",
    output_dir: str = "outputs",
    bad_quantile: float = 0.80,
) -> Dict:
    """Convert real input CSVs, then run the standard ontology-SHAP analysis."""
    artifacts = build_standard_dataset(input_dir, data_dir, bad_quantile=bad_quantile)
    result = run_analysis(data_dir, output_dir)
    result.update({"mode": "real", "artifacts": artifacts})
    return result


def _ontology_summary(mapped: pd.DataFrame, bad_wafers: list[str]) -> pd.DataFrame:
    group_cols = ["causal_role", "process_step", "mechanism_group"]
    frames = []
    for group_col in group_cols:
        if group_col in mapped.columns:
            frames.append(
                aggregate_shap_by_context(mapped, [group_col], wafer_ids=bad_wafers).assign(group_kind=group_col)
            )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
