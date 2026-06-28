"""Diagnose mediator credit absorption: measured root effect vs model SHAP share.

This is the numeric proof of *why* a plain SHAP ranking misses the upstream cause.
When two features are collinear (root -> mediator), a tree model tends to hand the
credit to the downstream mediator. So the model's SHAP can show:

    mediator SHAP high, root SHAP low

even though the DATA-measured mediation says the root drives the chain
(`indirect = a*b` large). That gap is "credit absorption" -- the exact reason the
real-data SHAP ranking looked no better than eyeballing top features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def credit_absorption(
    mediation_df: pd.DataFrame,
    shap_share: dict,
    min_indirect: float = 0.10,
) -> pd.DataFrame:
    """For each measured chain, compare the root's model-SHAP share to the mediator's.

    `shap_share` maps feature_id -> normalized mean|SHAP| (shares, not raw values).
    A chain is flagged `credit_absorbed` when its measured indirect effect is
    meaningful yet the mediator carries more model SHAP than its root -- i.e. the
    model parked the credit on the symptom, not the controllable cause.
    """
    rows = []
    for _, r in mediation_df.iterrows():
        root, med = r["root"], r["mediator"]
        rs = float(shap_share.get(root, 0.0))
        ms = float(shap_share.get(med, 0.0))
        ind = float(r["indirect"]) if np.isfinite(r["indirect"]) else float("nan")
        absorbed = bool(np.isfinite(ind) and abs(ind) >= min_indirect and ms > rs)
        rows.append({
            "root": root, "mediator": med, "indirect": ind,
            "root_shap_share": rs, "mediator_shap_share": ms,
            "shap_gap": ms - rs, "credit_absorbed": absorbed,
        })
    return pd.DataFrame(rows)
