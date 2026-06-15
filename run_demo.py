"""End-to-end demo pipeline.

generate data + real SHAP  ->  build hypothesis cards  ->  export outputs  ->  README assets

Run:  python run_demo.py
Then: streamlit run app.py
"""
from __future__ import annotations

import os

import pandas as pd

from src.data_generator import generate_all
from src.loaders import load_all_data
from src.ontology_mapping import map_shap_to_feature_dictionary
from src.shap_interpreter import aggregate_shap_by_context
from src.hypothesis_engine import build_hypothesis_cards, naive_vs_ontology_summary

DATA_DIR = "data"
OUT_DIR = "outputs"


def main() -> None:
    # 1) Generate every CSV (incl. real CatBoost + TreeSHAP, or logged fallback).
    summary = generate_all(DATA_DIR)

    # 2) Load + map ontology metadata onto SHAP.
    data = load_all_data(DATA_DIR)
    mapped = map_shap_to_feature_dictionary(data["shap_values"], data["feature_dictionary"])

    # 3) Headline check: naive (mediator) vs ontology root vs ground truth.
    cmp = naive_vs_ontology_summary(
        mapped, data["target"], data["causal_edges"], data["feature_dictionary"], data["ground_truth"]
    )

    # 4) Build hypothesis cards + ontology-level SHAP aggregation; export.
    cards = build_hypothesis_cards(
        mapped, data["target"], data["process_history"], data["causal_edges"],
        data["feature_dictionary"], top_n=5,
    )
    bad_wafers = data["target"].loc[data["target"]["bad_flag"] == 1, "wafer_id"].tolist()
    role_agg = aggregate_shap_by_context(mapped, ["causal_role"], wafer_ids=bad_wafers)
    step_agg = aggregate_shap_by_context(mapped, ["process_step"], wafer_ids=bad_wafers)
    mech_agg = aggregate_shap_by_context(mapped, ["mechanism_group"], wafer_ids=bad_wafers)
    ontology_summary = pd.concat([role_agg.assign(group_kind="causal_role"),
                                  step_agg.assign(group_kind="process_step"),
                                  mech_agg.assign(group_kind="mechanism_group")], ignore_index=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    cards.to_csv(os.path.join(OUT_DIR, "demo_hypothesis_cards.csv"), index=False)
    ontology_summary.to_csv(os.path.join(OUT_DIR, "ontology_level_shap_summary.csv"), index=False)

    # 5) README result images.
    from scripts.make_readme_assets import main as make_assets
    make_assets(DATA_DIR)

    # --- Acceptance summary
    print("\n" + "=" * 78)
    print("ACCEPTANCE CHECK")
    print("=" * 78)
    print(f"SHAP mode                         : {summary['shap_mode']}")
    print(f"naive top-SHAP (excl. leakage)    : {cmp['naive_top_nonleak']} "
          f"[{cmp['naive_top_nonleak_role']}]  (= 증상/mediator)")
    print(f"ontology-traced root cause        : {cmp['ontology_root']}")
    print(f"ground-truth root                 : {cmp['ground_truth_root']}")
    print(f"MATCH (ontology == ground truth)  : {'✓' if cmp['match'] else '✗'}")
    target_card = cards["evidence_path_text"].str.startswith(
        "CVD 압력 불안정(pressure instability) → edge 두께 비균일(non-uniformity) → 불량률 상승"
    ).any()
    print(f"required CVD card present         : {'✓' if target_card else '✗'}")
    print(f"leakage excluded from cards       : {'✓' if cmp['naive_top_overall_role'] == 'leakage_or_post_outcome' else '?'}"
          f"  (raw #1 = {cmp['naive_top_overall']})")
    print(f"outputs written                   : {OUT_DIR}/demo_hypothesis_cards.csv, "
          f"{OUT_DIR}/ontology_level_shap_summary.csv")
    print("=" * 78)
    print("다음 단계: streamlit run app.py")


if __name__ == "__main__":
    main()
