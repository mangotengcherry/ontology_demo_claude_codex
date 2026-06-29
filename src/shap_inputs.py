"""Parse wide-form per-wafer SHAP inputs into the project's internal SHAP views.

The company export contract changed from a single pre-aggregated long table
(`x_feature_shap_value.csv` with `feature`, `shap_value`) to two *wide* tables:

    bad_wafer_shap_value.csv  -- one row per bad wafer,  one column per x feature
    all_wafer_shap_value.csv  -- one row per good+bad wafer, one column per x feature

Each cell is that feature's SHAP value for that wafer. This module turns those
wide tables into the two views the rest of the pipeline needs:

  1. bad-wafer cohort mean SHAP  (feeds the ontology-SHAP interpretation arm)
  2. good-vs-bad cohort comparison (the model-interpretation view: which features
     the model leans on MORE for bad wafers than for good wafers)

It auto-detects the id columns (`root_lot_id`+`wafer_id`, or a combined
`root_lot_wafer_id`) and ignores non-feature export columns such as a base /
expected value, a prediction, the target, or a timestamp. numpy/pandas only.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

BAD_SHAP_FILE = "bad_wafer_shap_value.csv"
ALL_SHAP_FILE = "all_wafer_shap_value.csv"
LEGACY_SHAP_FILE = "x_feature_shap_value.csv"

# Columns that can appear in a wide SHAP export but are NOT x features.
SHAP_META_COLUMNS = {
    "root_lot_id", "wafer_id", "root_lot_wafer_id", "lot_id",
    "tkout_time", "time", "date",
    "target", "y", "label", "bad_flag", "defect_rate",
    "base_value", "expected_value", "bias", "intercept",
    "prediction", "pred", "y_pred", "model_output", "score", "shap_sum",
}

_EPS = 1e-9


# --------------------------------------------------------------------------------------
# id handling
# --------------------------------------------------------------------------------------
def combined_key(root_lot_id: pd.Series, wafer_id: pd.Series) -> pd.Series:
    """Canonical wafer key: 'root_lot_id|wafer_id' (matches bad_wafers.csv contract)."""
    return root_lot_id.astype(str) + "|" + wafer_id.astype(str)


def _resolve_keys(df: pd.DataFrame) -> pd.Series:
    """Return a canonical 'root_lot_id|wafer_id' key Series for a wide SHAP frame."""
    if {"root_lot_id", "wafer_id"}.issubset(df.columns):
        return combined_key(df["root_lot_id"], df["wafer_id"])
    if "root_lot_wafer_id" in df.columns:
        return df["root_lot_wafer_id"].astype(str)
    # No id columns at all: fall back to positional row ids so cohort math still works.
    return pd.Series([f"__row_{i}__" for i in range(len(df))], index=df.index)


# --------------------------------------------------------------------------------------
# feature-column detection
# --------------------------------------------------------------------------------------
def detect_feature_columns(df: pd.DataFrame, raw_feature_cols: Optional[List[str]] = None) -> List[str]:
    """Choose the SHAP value columns from a wide frame.

    Prefers the intersection with the raw_data feature columns (most robust, keeps
    feature_ids aligned). Falls back to every numeric column that is not a known
    meta column when no names line up (defensive: never returns empty if signal exists).
    """
    cols = list(df.columns)
    if raw_feature_cols:
        raw_set = set(map(str, raw_feature_cols))
        matched = [c for c in cols if str(c) in raw_set]
        if matched:
            return matched
    out: List[str] = []
    for c in cols:
        if str(c) in SHAP_META_COLUMNS:
            continue
        if pd.to_numeric(df[c], errors="coerce").notna().mean() >= 0.5:
            out.append(c)
    return out


def read_wide_shap(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


# --------------------------------------------------------------------------------------
# aggregations
# --------------------------------------------------------------------------------------
def cohort_mean_shap(wide: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Collapse a wide per-wafer SHAP frame into per-feature cohort statistics.

    Returns columns: feature, shap_value (mean signed SHAP -> direction),
    mean_abs_shap (mean |SHAP| -> true global importance), n_wafers.

    Decoupling `shap_value` (signed mean) from `mean_abs_shap` is deliberate: a
    bidirectional feature would be under-ranked by |signed mean|, so importance
    uses mean(|SHAP|) while sign/direction uses the signed mean.
    """
    rows = []
    for c in feature_cols:
        vals = pd.to_numeric(wide[c], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        rows.append(
            {
                "feature": str(c),
                "shap_value": float(np.mean(vals)),
                "mean_abs_shap": float(np.mean(np.abs(vals))),
                "n_wafers": int(vals.size),
            }
        )
    out = pd.DataFrame(rows, columns=["feature", "shap_value", "mean_abs_shap", "n_wafers"])
    return out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def cohort_comparison(
    all_wide: pd.DataFrame,
    feature_cols: List[str],
    bad_keys: set,
) -> pd.DataFrame:
    """Per-feature good-vs-bad SHAP comparison from the all-wafer wide frame.

    `bad_keys` are canonical 'root_lot_id|wafer_id' keys (the bad cohort). Features
    are ranked by how much MORE the model attributes to them on bad wafers than on
    good wafers -- this is the model-interpretation view of the company's own model.
    """
    keys = _resolve_keys(all_wide)
    is_bad = keys.isin(bad_keys).to_numpy()
    n_bad = int(is_bad.sum())
    n_good = int((~is_bad).sum())

    rows = []
    for c in feature_cols:
        vals = pd.to_numeric(all_wide[c], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(vals)
        bad_mask = is_bad & finite
        good_mask = (~is_bad) & finite
        bad_vals = vals[bad_mask]
        good_vals = vals[good_mask]
        all_vals = vals[finite]
        if all_vals.size == 0:
            continue
        bad_abs = float(np.mean(np.abs(bad_vals))) if bad_vals.size else 0.0
        good_abs = float(np.mean(np.abs(good_vals))) if good_vals.size else 0.0
        all_abs = float(np.mean(np.abs(all_vals)))
        bad_signed = float(np.mean(bad_vals)) if bad_vals.size else 0.0
        good_signed = float(np.mean(good_vals)) if good_vals.size else 0.0
        pooled_std = float(np.std(all_vals) + _EPS)
        rows.append(
            {
                "feature_id": str(c),
                "n_bad": int(bad_vals.size),
                "n_good": int(good_vals.size),
                "bad_mean_abs_shap": round(bad_abs, 6),
                "good_mean_abs_shap": round(good_abs, 6),
                "all_mean_abs_shap": round(all_abs, 6),
                "bad_mean_signed_shap": round(bad_signed, 6),
                "good_mean_signed_shap": round(good_signed, 6),
                "abs_shap_gap_bad_minus_good": round(bad_abs - good_abs, 6),
                "signed_shap_separation_z": round((bad_signed - good_signed) / pooled_std, 6),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    total_bad = out["bad_mean_abs_shap"].sum() or 1.0
    total_good = out["good_mean_abs_shap"].sum() or 1.0
    out["bad_shap_share_pct"] = (100.0 * out["bad_mean_abs_shap"] / total_bad).round(4)
    out["good_shap_share_pct"] = (100.0 * out["good_mean_abs_shap"] / total_good).round(4)
    out["n_bad_total"] = n_bad
    out["n_good_total"] = n_good
    return out.sort_values("abs_shap_gap_bad_minus_good", ascending=False).reset_index(drop=True)


def long_table(wide: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Melt a wide SHAP frame to long [root_lot_id, wafer_id, feature, shap_value].

    Used by the performance arm to roll up the provided production SHAP by role.
    """
    keys = _resolve_keys(wide)
    base = pd.DataFrame({"_key": keys.to_numpy()})
    if {"root_lot_id", "wafer_id"}.issubset(wide.columns):
        base["root_lot_id"] = wide["root_lot_id"].astype(str).to_numpy()
        base["wafer_id"] = wide["wafer_id"].astype(str).to_numpy()
    else:
        split = base["_key"].str.split("|", n=1, expand=True)
        base["root_lot_id"] = split[0]
        base["wafer_id"] = split[1] if split.shape[1] > 1 else split[0]
    frames = []
    for c in feature_cols:
        frames.append(
            pd.DataFrame(
                {
                    "root_lot_id": base["root_lot_id"].to_numpy(),
                    "wafer_id": base["wafer_id"].to_numpy(),
                    "feature": str(c),
                    "shap_value": pd.to_numeric(wide[c], errors="coerce").to_numpy(dtype=float),
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["root_lot_id", "wafer_id", "feature", "shap_value"]
    )


def bad_keys_from_wide(bad_wide: pd.DataFrame) -> set:
    """Canonical bad-cohort keys taken directly from the bad-wafer SHAP export."""
    return set(_resolve_keys(bad_wide).astype(str))


# --------------------------------------------------------------------------------------
# top-level resolver used by the adapter
# --------------------------------------------------------------------------------------
def resolve_cohort_mean(
    input_dir: str,
    raw_feature_cols: List[str],
    bad_keys: set,
    read_csv,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], str]:
    """Resolve the bad-cohort mean SHAP + optional cohort comparison from inputs.

    Precedence:
      1. bad_wafer_shap_value.csv (wide)              -> cohort mean from it
      2. all_wafer_shap_value.csv (wide), no bad file -> cohort mean from bad rows
      3. x_feature_shap_value.csv (legacy long)       -> used as-is

    `read_csv(filename) -> DataFrame | None` lets the caller control IO/existence.
    Returns (cohort_mean_df, comparison_df_or_None, source_tag).
    """
    bad_wide = read_csv(BAD_SHAP_FILE)
    all_wide = read_csv(ALL_SHAP_FILE)

    comparison = None
    if all_wide is not None:
        all_feats = detect_feature_columns(all_wide, raw_feature_cols)
        # When no explicit bad export, the bad cohort is the all-wafer rows flagged bad.
        comp_bad_keys = bad_keys_from_wide(bad_wide) if bad_wide is not None else set(bad_keys)
        comparison = cohort_comparison(all_wide, all_feats, comp_bad_keys)

    if bad_wide is not None:
        feats = detect_feature_columns(bad_wide, raw_feature_cols)
        return cohort_mean_shap(bad_wide, feats), comparison, "wide_bad_wafer"

    if all_wide is not None:
        feats = detect_feature_columns(all_wide, raw_feature_cols)
        keys = _resolve_keys(all_wide)
        bad_rows = all_wide[keys.isin(bad_keys).to_numpy()]
        if bad_rows.empty:
            bad_rows = all_wide  # no overlap -> fall back to all rows (better than empty)
        return cohort_mean_shap(bad_rows, feats), comparison, "wide_all_wafer_bad_subset"

    legacy = read_csv(LEGACY_SHAP_FILE)
    if legacy is not None:
        missing = [c for c in ["feature", "shap_value"] if c not in legacy.columns]
        if missing:
            raise ValueError(f"{LEGACY_SHAP_FILE} is missing required columns: {missing}")
        out = legacy[["feature", "shap_value"]].copy()
        out["feature"] = out["feature"].astype(str)
        out["shap_value"] = pd.to_numeric(out["shap_value"], errors="coerce")
        return out, comparison, "legacy_long"

    raise FileNotFoundError(
        f"No SHAP input found. Provide one of: {BAD_SHAP_FILE} / {ALL_SHAP_FILE} "
        f"(wide per-wafer) or {LEGACY_SHAP_FILE} (legacy long)."
    )


def shap_input_exists(exists) -> bool:
    """True when any supported SHAP input file is present. `exists(name) -> bool`."""
    return any(exists(name) for name in (BAD_SHAP_FILE, ALL_SHAP_FILE, LEGACY_SHAP_FILE))
