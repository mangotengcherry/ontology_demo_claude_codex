"""Generate virtual company-style input CSVs for evaluating the real pipeline."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT_FEATURE = "num|CVD|erd|pcf|PRESSURE_SLOPE_P95|CVD"
MEDIATOR_FEATURE = "num|MET_CVD|THK_EDGE|AVG"
VM_FEATURE = "num|CVD|RF_TIME_1|VALUE"
QUEUE_FEATURE = "num|ETCH|QUEUE_TIME|VALUE"
LEAKAGE_FEATURE = "num|FINAL|TARGET_PROXY|VALUE"
PPID_FEATURE = "cat|ppid|CVD"
EQP_FEATURE = "cat|eqp|CVD"
CHAMBER_FEATURE = "cat|eqp_ch|CVD"
SLOT_FEATURE = "cat|slot_n|CVD"


# mean|SHAP| weight per feature. mediator > root on purpose (credit absorption):
# a tree model parks credit on the downstream metrology symptom, so the ontology
# trace + measured mediation have something real to correct.
_SHAP_WEIGHTS = {
    LEAKAGE_FEATURE: 2.2,
    MEDIATOR_FEATURE: 1.35,
    ROOT_FEATURE: 0.85,
    VM_FEATURE: 0.32,
    QUEUE_FEATURE: 0.22,
    PPID_FEATURE: 0.18,
    CHAMBER_FEATURE: 0.15,
    SLOT_FEATURE: 0.06,
    EQP_FEATURE: 0.0,
}


def generate_virtual_input_dataset(input_dir: str = "input", n_wafers: int = 250, seed: int = 42) -> dict:
    """Write the virtual company-style real-data input contract.

    Files written:
      raw_data.csv               -- per-wafer features + target
      all_wafer_shap_value.csv   -- wide per-wafer SHAP (good + bad)   [new contract]
      bad_wafer_shap_value.csv   -- wide per-wafer SHAP (bad cohort)   [new contract]
      x_feature_shap_value.csv   -- legacy long bad-cohort mean SHAP   [back-compat]
      prc_metro_relation.csv     -- process/metro relation seed
      bad_wafers.csv             -- bad-wafer cohort

    The data intentionally makes the metrology mediator rank above the upstream
    ERD root in SHAP, so the ontology trace has something meaningful to correct.
    """
    os.makedirs(input_dir, exist_ok=True)
    raw = _build_raw_data(n_wafers=n_wafers, seed=seed)
    relation = _build_prc_metro_relation()
    bad_wafers = _build_bad_wafers(raw)

    bad_keys = set(bad_wafers["root_lot_id"].astype(str) + "|" + bad_wafers["wafer_id"].astype(str))
    all_shap_wide = _build_wide_shap(raw, seed=seed)
    raw_keys = raw["root_lot_id"].astype(str) + "|" + raw["wafer_id"].astype(str)
    bad_shap_wide = all_shap_wide[raw_keys.isin(bad_keys).to_numpy()].reset_index(drop=True)
    legacy_mean = _legacy_mean_from_wide(bad_shap_wide)

    raw.to_csv(os.path.join(input_dir, "raw_data.csv"), index=False)
    all_shap_wide.to_csv(os.path.join(input_dir, "all_wafer_shap_value.csv"), index=False)
    bad_shap_wide.to_csv(os.path.join(input_dir, "bad_wafer_shap_value.csv"), index=False)
    legacy_mean.to_csv(os.path.join(input_dir, "x_feature_shap_value.csv"), index=False)
    relation.to_csv(os.path.join(input_dir, "prc_metro_relation.csv"), index=False)
    bad_wafers.to_csv(os.path.join(input_dir, "bad_wafers.csv"), index=False)
    return {
        "input_dir": input_dir,
        "n_wafers": int(len(raw)),
        "n_features": int(len([c for c in raw.columns if c not in {"root_lot_id", "wafer_id", "tkout_time", "target"}])),
        "n_bad_wafers": int(len(bad_wafers)),
    }


def _build_wide_shap(raw: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Synthesize wide per-wafer SHAP consistent with the data-generating chain.

    SHAP for each feature ~ weight * z(numeric proxy of the feature) + small noise,
    signed so that higher values push defect_rate up. Bad wafers (high on the
    root->mediator chain) therefore carry larger SHAP on those features, which is
    exactly the good-vs-bad separation the cohort comparison surfaces.
    """
    rng = np.random.default_rng(seed + 7)
    out = raw[["root_lot_id", "wafer_id"]].copy()
    for feat, weight in _SHAP_WEIGHTS.items():
        if feat not in raw.columns:
            continue
        proxy = _numeric_proxy(raw[feat])
        shap = weight * _z(proxy) + rng.normal(0.0, max(weight * 0.15, 0.02), len(raw))
        out[feat] = np.round(shap, 6)
    return out


def _numeric_proxy(series: pd.Series) -> np.ndarray:
    """Numeric stand-in for a feature column so it can carry a synthetic SHAP signal."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() >= 0.5:
        return numeric.fillna(numeric.median() if numeric.notna().any() else 0.0).to_numpy(float)
    codes = series.astype(str).astype("category").cat.codes.to_numpy(float)
    return codes


def _legacy_mean_from_wide(bad_shap_wide: pd.DataFrame) -> pd.DataFrame:
    """Collapse the bad wide SHAP to the legacy long (feature, shap_value) table."""
    feature_cols = [c for c in bad_shap_wide.columns if c not in {"root_lot_id", "wafer_id"}]
    rows = [{"feature": c, "shap_value": round(float(bad_shap_wide[c].mean()), 6)} for c in feature_cols]
    return pd.DataFrame(rows).sort_values("shap_value", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def _build_raw_data(n_wafers: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_lots = max(1, int(np.ceil(n_wafers / 5)))
    rows = []
    for i in range(n_wafers):
        lot = f"VLOT{1000 + i // 5}"
        rows.append(
            {
                "root_lot_id": lot,
                "wafer_id": f"{lot}_W{(i % 5) + 1:02d}",
                "tkout_time": f"2026-05-{(i % 28) + 1:02d} 08:00:00",
            }
        )
    df = pd.DataFrame(rows)

    exposed = rng.random(n_wafers) < 0.30
    df[PPID_FEATURE] = np.where(exposed, "PPID_77A", "PPID_10B")
    df[EQP_FEATURE] = "CVD_TOOL_A"
    df[CHAMBER_FEATURE] = np.where(exposed, "CVD_CH_03", "CVD_CH_01")
    df[SLOT_FEATURE] = rng.integers(1, 26, size=n_wafers)

    root = rng.normal(0.0, 1.0, n_wafers) + exposed * 2.2
    mediator = 1.1 * root + rng.normal(0.0, 0.55, n_wafers)
    vm = 0.45 * root + rng.normal(0.0, 0.9, n_wafers)
    queue = rng.gamma(shape=2.0, scale=1.0, size=n_wafers)
    target = 5.0 + 2.8 * _z(mediator) + 0.7 * _z(queue) + rng.normal(0.0, 0.9, n_wafers)
    target = np.clip(target, 0.05, None)

    df[ROOT_FEATURE] = np.round(root, 5)
    df[MEDIATOR_FEATURE] = np.round(mediator, 5)
    df[VM_FEATURE] = np.round(vm, 5)
    df[QUEUE_FEATURE] = np.round(queue, 5)
    df[LEAKAGE_FEATURE] = np.round(target + rng.normal(0.0, 0.2, n_wafers), 5)
    df["target"] = np.round(target, 5)

    feature_cols = [
        PPID_FEATURE,
        EQP_FEATURE,
        CHAMBER_FEATURE,
        SLOT_FEATURE,
        ROOT_FEATURE,
        MEDIATOR_FEATURE,
        VM_FEATURE,
        QUEUE_FEATURE,
        LEAKAGE_FEATURE,
    ]
    return df[["root_lot_id", "wafer_id", "tkout_time", "target"] + feature_cols]


def _build_prc_metro_relation() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "prc_step": "CVD",
                "metro_step": "MET_CVD",
                "metro_item": "THK_EDGE",
                "subitem_id": "AVG",
                "metro_grade": "A",
            }
        ]
    )


def _build_bad_wafers(raw: pd.DataFrame) -> pd.DataFrame:
    threshold = pd.to_numeric(raw["target"], errors="coerce").quantile(0.80)
    bad = raw.loc[pd.to_numeric(raw["target"], errors="coerce") >= threshold, ["root_lot_id", "wafer_id"]]
    return bad.reset_index(drop=True)


def _z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (values - values.mean()) / (values.std() + 1e-9)
