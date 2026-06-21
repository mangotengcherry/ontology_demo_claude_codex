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


def generate_virtual_input_dataset(input_dir: str = "input", n_wafers: int = 250, seed: int = 42) -> dict:
    """Write raw_data, bad-wafer mean SHAP, and process-metro relation CSVs.

    The data intentionally makes the metrology mediator rank above the upstream
    ERD root in SHAP, so the ontology trace has something meaningful to correct.
    """
    os.makedirs(input_dir, exist_ok=True)
    raw = _build_raw_data(n_wafers=n_wafers, seed=seed)
    shap_values = _build_bad_wafer_mean_shap()
    relation = _build_prc_metro_relation()
    bad_wafers = _build_bad_wafers(raw)

    raw.to_csv(os.path.join(input_dir, "raw_data.csv"), index=False)
    shap_values.to_csv(os.path.join(input_dir, "x_feature_shap_value.csv"), index=False)
    relation.to_csv(os.path.join(input_dir, "prc_metro_relation.csv"), index=False)
    bad_wafers.to_csv(os.path.join(input_dir, "bad_wafers.csv"), index=False)
    return {
        "input_dir": input_dir,
        "n_wafers": int(len(raw)),
        "n_features": int(len([c for c in raw.columns if c not in {"root_lot_id", "wafer_id", "tkout_time", "target"}])),
        "n_bad_wafers": int(len(bad_wafers)),
    }


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


def _build_bad_wafer_mean_shap() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"feature": LEAKAGE_FEATURE, "shap_value": 2.2},
            {"feature": MEDIATOR_FEATURE, "shap_value": 1.35},
            {"feature": ROOT_FEATURE, "shap_value": 0.85},
            {"feature": VM_FEATURE, "shap_value": 0.32},
            {"feature": QUEUE_FEATURE, "shap_value": 0.22},
            {"feature": PPID_FEATURE, "shap_value": 0.18},
            {"feature": CHAMBER_FEATURE, "shap_value": 0.15},
        ]
    )


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
