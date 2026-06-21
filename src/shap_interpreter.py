"""Select bad wafers and slice / aggregate the mapped SHAP table."""
from __future__ import annotations

from typing import List

import pandas as pd


def get_bad_wafers(target_df: pd.DataFrame, top_k: int = 20) -> pd.DataFrame:
    """Return the worst wafers by defect_rate (bad_flag first, then highest rate)."""
    df = target_df.copy()
    df = df.sort_values(["bad_flag", "defect_rate"], ascending=[False, False])
    return df.head(top_k).reset_index(drop=True)


def get_top_shap_features(
    mapped_shap_df: pd.DataFrame,
    wafer_id: str,
    root_lot_id: str | None = None,
    target_id: str = "defect_rate",
    top_n: int = 10,
) -> pd.DataFrame:
    """Top-N features by |SHAP| for one wafer (raw SHAP view, leakage still included)."""
    if (
        "shap_scope" in mapped_shap_df.columns
        and not mapped_shap_df.empty
        and mapped_shap_df["shap_scope"].fillna("").eq("bad_wafer_mean").all()
    ):
        return mapped_shap_df.sort_values("abs_shap_value", ascending=False).head(top_n).reset_index(drop=True)
    df = mapped_shap_df[
        (mapped_shap_df["wafer_id"] == wafer_id) & (mapped_shap_df["target_id"] == target_id)
    ]
    if root_lot_id is not None:
        df = df[df["root_lot_id"] == root_lot_id]
    return df.sort_values("abs_shap_value", ascending=False).head(top_n).reset_index(drop=True)


def aggregate_shap_by_context(
    mapped_shap_df: pd.DataFrame,
    group_cols: List[str],
    wafer_ids: List[str] | None = None,
) -> pd.DataFrame:
    """Aggregate mean |SHAP| (and signed mean) by ontology context columns.

    `group_cols` can be any of: process_step, source_type, causal_role,
    mechanism_group, wafer_region, wl_or_layer.
    """
    df = mapped_shap_df
    if wafer_ids is not None:
        df = df[df["wafer_id"].isin(wafer_ids)]
        if df.empty and "shap_scope" in mapped_shap_df.columns:
            cohort_mean = mapped_shap_df["shap_scope"].fillna("").eq("bad_wafer_mean")
            if cohort_mean.all():
                df = mapped_shap_df
    agg = (
        df.groupby(group_cols)
        .agg(
            mean_abs_shap=("abs_shap_value", "mean"),
            total_abs_shap=("abs_shap_value", "sum"),
            mean_signed_shap=("shap_value", "mean"),
            n_rows=("abs_shap_value", "size"),
        )
        .reset_index()
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    return agg
