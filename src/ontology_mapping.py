"""Join ontology feature metadata onto raw SHAP rows (the first ontology upgrade)."""
from __future__ import annotations

import pandas as pd

# Columns brought over from the feature dictionary onto each SHAP row.
_META_COLS = [
    "display_name_ko", "source_type", "process_step", "module", "metric_type",
    "wafer_region", "wl_or_layer", "observed_stage", "temporal_relation_to_target",
    "causal_role", "leakage_risk", "controllability", "mechanism_group",
    "interpretation_template_ko",
]


def map_shap_to_feature_dictionary(
    shap_df: pd.DataFrame, feature_dict_df: pd.DataFrame
) -> pd.DataFrame:
    """Left-join feature metadata onto SHAP rows.

    Features without metadata are flagged `causal_role = 'unknown'`. A boolean
    `is_leakage` column marks target-derived / post-outcome features so the app and
    hypothesis engine can exclude them from causal interpretation while keeping them visible.
    """
    meta = feature_dict_df.set_index("feature_id")
    available = [c for c in _META_COLS if c in feature_dict_df.columns]
    mapped = shap_df.merge(
        feature_dict_df[["feature_id"] + available], on="feature_id", how="left"
    )

    # Missing metadata -> unknown role, unknown leakage risk.
    mapped["causal_role"] = mapped["causal_role"].fillna("unknown")
    if "leakage_risk" in mapped:
        mapped["leakage_risk"] = mapped["leakage_risk"].fillna("unknown")
    if "mechanism_group" in mapped:
        mapped["mechanism_group"] = mapped["mechanism_group"].fillna("unknown")

    mapped["has_metadata"] = mapped["feature_id"].isin(meta.index)
    mapped["is_leakage"] = mapped["causal_role"].eq("leakage_or_post_outcome")
    return mapped
