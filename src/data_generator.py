"""Generate the synthetic semiconductor-yield dataset for the ontology-SHAP demo.

The dataset encodes ONE known ground-truth causal chain and deliberately sets up a
SHAP "trap": the mediator (edge-thickness non-uniformity) is *more* correlated with
the target than the true upstream root (CVD pressure instability), so naive top-SHAP
points at the symptom. The ontology layer later traces back to the real root.

Ground-truth chain:
    fdc_cvd_pressure_slope_p95  (root, controllable @ CVD_CH_03 / PPID_77A)
        --physicallyAffects-->  metro_thk_edge_wl128/160  (mediator)
        --mediates-->           defect_rate               (target)

Real SHAP is computed ONCE here (CatBoost + TreeSHAP) and written to CSV. If catboost
or shap are unavailable, a deterministic synthetic SHAP consistent with this pattern is
written instead (clearly logged). The Streamlit app only reads CSVs.

Run:  python -m src.data_generator   (or via run_demo.py)
"""
from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd

SEED = 42
N_LOTS = 50
WAFERS_PER_LOT = 5
N_WAFERS = N_LOTS * WAFERS_PER_LOT  # 250
TARGET_ID = "defect_rate"
BAD_CHAMBER = "CVD_CH_03"
BAD_PPID = "PPID_77A"
TRUE_ROOT_FEATURE = "fdc_cvd_pressure_slope_p95"

# Model feature columns (X) — order matters for CatBoost categorical indices.
CATEGORICAL_FEATURES = ["product_id", "route_id"]
NUMERIC_FEATURES = [
    "fdc_cvd_pressure_slope_p95",
    "fdc_cvd_rf_power_std",
    "fdc_etch_temp_range",
    "metro_thk_edge_wl128",
    "metro_thk_edge_wl160",
    "metro_cd_center_wl160",
    "queue_time_before_etch",
    "prev_defect_count_after_metro",
    "recipe_ppid_77a_flag",
    "final_eds_proxy_leakage_feature",
]
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


# --------------------------------------------------------------------------------------
# Synthetic feature + target generation
# --------------------------------------------------------------------------------------
def generate_feature_matrix_and_target() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the feature matrix, target, and process history with a coherent causal chain."""
    rng = np.random.default_rng(SEED)

    root_lot_ids = [f"LOT{1000 + i}" for i in range(N_LOTS)]
    rows = []
    for lot in root_lot_ids:
        for w in range(1, WAFERS_PER_LOT + 1):
            rows.append({"root_lot_id": lot, "wafer_id": f"{lot}_W{w:02d}"})
    df = pd.DataFrame(rows)

    # Confounders / grouping (these also drive which wafers go to the bad chamber).
    df["product_id"] = rng.choice(["PRODA", "PRODB"], size=N_WAFERS, p=[0.6, 0.4])
    df["route_id"] = rng.choice(["ROUTE_X", "ROUTE_Y"], size=N_WAFERS, p=[0.5, 0.5])

    # ~30% of wafers run on the troubled CVD chamber / recipe. PRODA is slightly more
    # exposed -> product_id becomes a (weak) confounder of the root cause.
    bad_chamber_prob = np.where(df["product_id"].values == "PRODA", 0.40, 0.22)
    on_bad_chamber = rng.random(N_WAFERS) < bad_chamber_prob
    df["chamber_id"] = np.where(on_bad_chamber, BAD_CHAMBER, "CVD_CH_01")
    df["ppid"] = np.where(on_bad_chamber, BAD_PPID, "PPID_10B")
    df["recipe_ppid_77a_flag"] = (df["ppid"].values == BAD_PPID).astype(int)

    # ---- ROOT: CVD pressure-slope instability, elevated & concentrated on bad chamber.
    pressure = rng.normal(0.0, 1.0, N_WAFERS) + on_bad_chamber * 2.2
    df["fdc_cvd_pressure_slope_p95"] = np.round(pressure, 4)

    # Secondary root: RF power std, partially shares the CVD instability.
    rf_power = 0.5 * pressure + rng.normal(0.0, 0.9, N_WAFERS) + on_bad_chamber * 0.6
    df["fdc_cvd_rf_power_std"] = np.round(rf_power, 4)

    # Unrelated-ish etch root candidate (noise; minor, off the true chain).
    df["fdc_etch_temp_range"] = np.round(rng.normal(0.0, 1.0, N_WAFERS), 4)

    # ---- MEDIATOR: edge-thickness non-uniformity = f(CVD instability) + own noise.
    # The mediator carries its own noise that is later shared with the target, which is
    # what makes corr(target, mediator) > corr(target, root): the structural trap.
    thk128 = 1.0 * pressure + rng.normal(0.0, 0.55, N_WAFERS)
    thk160 = 0.85 * pressure + 0.30 * rf_power + rng.normal(0.0, 0.55, N_WAFERS)
    df["metro_thk_edge_wl128"] = np.round(thk128, 4)
    df["metro_thk_edge_wl160"] = np.round(thk160, 4)

    # Center CD: weak mediator, mostly off-chain noise.
    df["metro_cd_center_wl160"] = np.round(0.2 * pressure + rng.normal(0.0, 1.0, N_WAFERS), 4)

    # Minor controllable root: queue time before etch.
    queue_time = rng.gamma(shape=2.0, scale=1.0, size=N_WAFERS)
    df["queue_time_before_etch"] = np.round(queue_time, 4)

    # ---- TARGET: defect_rate driven mainly by the mediator (edge thickness).
    latent = (
        3.0 * _z(thk128)
        + 2.2 * _z(thk160)
        + 0.5 * _z(queue_time)
        + 0.3 * _z(df["metro_cd_center_wl160"].values)
        + rng.normal(0.0, 1.0, N_WAFERS)
    )
    defect_rate = 6.0 + 2.0 * latent  # percent-like scale
    defect_rate = np.clip(defect_rate, 0.05, None)
    df_target = pd.DataFrame(
        {
            "root_lot_id": df["root_lot_id"],
            "wafer_id": df["wafer_id"],
            "defect_rate": np.round(defect_rate, 4),
        }
    )
    threshold = np.quantile(defect_rate, 0.80)
    df_target["bad_flag"] = (defect_rate >= threshold).astype(int)

    # ---- PROXY: prior defect count after metro (early-warning signal, correlated but
    # not causal — a downstream-of-mediator counter, observed before the final target).
    prev_defect = np.clip(
        np.round(0.8 * _z(thk128) + 0.5 * _z(latent) + rng.normal(0.0, 0.8, N_WAFERS) + 2.0),
        0,
        None,
    ).astype(int)
    df["prev_defect_count_after_metro"] = prev_defect

    # ---- LEAKAGE: derived from the target itself (post-outcome proxy). If kept in the
    # model it dominates SHAP; the ontology guardrail excludes it from causal reasoning.
    df["final_eds_proxy_leakage_feature"] = np.round(
        0.9 * defect_rate + rng.normal(0.0, 0.3, N_WAFERS), 4
    )

    # Reorder feature_matrix to the documented schema.
    feature_cols = [
        "root_lot_id", "wafer_id", "product_id", "route_id",
        "fdc_cvd_pressure_slope_p95", "fdc_cvd_rf_power_std", "fdc_etch_temp_range",
        "metro_thk_edge_wl128", "metro_thk_edge_wl160", "metro_cd_center_wl160",
        "queue_time_before_etch", "prev_defect_count_after_metro",
        "recipe_ppid_77a_flag", "final_eds_proxy_leakage_feature",
    ]
    feature_matrix = df[feature_cols].copy()

    process_history = _build_process_history(df, rng)
    return feature_matrix, df_target, process_history


def _z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / (x.std() + 1e-9)


def _build_process_history(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Two process steps (CVD, Etch) per wafer; bad wafers concentrate on CVD_CH_03/PPID_77A."""
    records: List[dict] = []
    steps = [
        ("CVD", "STEP_CVD", 10, "CVD_TOOL_A"),
        ("ETCH", "STEP_ETCH", 20, "ETCH_TOOL_B"),
    ]
    for _, r in df.iterrows():
        for process_step, step_id, step_order, tool in steps:
            if process_step == "CVD":
                chamber = r["chamber_id"]
                ppid = r["ppid"]
                recipe = "REC_CVD_01"
                days_since_pm = int(rng.integers(1, 40))
                # Troubled chamber tends to run longer past PM.
                if chamber == BAD_CHAMBER:
                    days_since_pm = int(rng.integers(25, 60))
            else:
                chamber = "ETCH_CH_01"
                ppid = "PPID_ETCH"
                recipe = "REC_ETCH_01"
                days_since_pm = int(rng.integers(1, 30))
            records.append(
                {
                    "root_lot_id": r["root_lot_id"],
                    "wafer_id": r["wafer_id"],
                    "step_id": f"{r['wafer_id']}_{step_id}",
                    "process_step": process_step,
                    "step_order": step_order,
                    "tool_id": tool,
                    "chamber_id": chamber,
                    "recipe_id": recipe,
                    "ppid": ppid,
                    "start_time": f"2026-05-{step_order:02d}T08:00:00",
                    "end_time": f"2026-05-{step_order:02d}T09:30:00",
                    "days_since_pm": days_since_pm,
                }
            )
    return pd.DataFrame(records)


# --------------------------------------------------------------------------------------
# Real SHAP (CatBoost + TreeSHAP) with deterministic fallback
# --------------------------------------------------------------------------------------
def compute_shap_values(
    feature_matrix: pd.DataFrame, target: pd.DataFrame
) -> tuple[pd.DataFrame, str, dict]:
    """Train CatBoost on X->defect_rate, compute TreeSHAP. Returns (long shap_df, mode, metrics)."""
    X = feature_matrix[MODEL_FEATURES].copy()
    y = target["defect_rate"].values
    try:
        shap_long, metrics = _real_shap(X, y, feature_matrix, target)
        return shap_long, "real_catboost_treeshap", metrics
    except Exception as exc:  # pragma: no cover - exercised only when libs missing
        print(f"[data_generator] CatBoost/SHAP unavailable ({exc!r}); "
              f"using deterministic synthetic SHAP fallback.")
        shap_long, metrics = _synthetic_shap(X, y, feature_matrix, target)
        return shap_long, "synthetic_fallback", metrics


def _real_shap(X, y, feature_matrix, target):
    from catboost import CatBoostRegressor, Pool
    import shap
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    cat_idx = [X.columns.get_loc(c) for c in CATEGORICAL_FEATURES]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=SEED)

    model = CatBoostRegressor(
        iterations=400, depth=6, learning_rate=0.05, loss_function="RMSE",
        random_seed=SEED, verbose=False,
    )
    model.fit(Pool(X_tr, y_tr, cat_features=cat_idx))

    def _metrics(Xs, ys):
        pred = model.predict(Pool(Xs, cat_features=cat_idx))
        return r2_score(ys, pred), float(np.sqrt(mean_squared_error(ys, pred)))

    r2_tr, rmse_tr = _metrics(X_tr, y_tr)
    r2_te, rmse_te = _metrics(X_te, y_te)
    metrics = {
        "model": "CatBoostRegressor", "train_r2": r2_tr, "test_r2": r2_te,
        "train_rmse": rmse_tr, "test_rmse": rmse_te, "y_pred": model.predict(Pool(X, cat_features=cat_idx)),
    }

    explainer = shap.TreeExplainer(model)
    shap_matrix = explainer.shap_values(Pool(X, cat_features=cat_idx))
    shap_long = _to_long_shap(shap_matrix, X, feature_matrix, target)
    return shap_long, metrics


def _synthetic_shap(X, y, feature_matrix, target):
    """Deterministic SHAP consistent with the ground-truth pattern (no ML libs needed)."""
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

    # Build a numeric design matrix (one-hot the two categoricals) for a transparent
    # linear surrogate, then use per-row centered contributions as "SHAP-like" values.
    Xn = X.copy()
    Xn = pd.get_dummies(Xn, columns=CATEGORICAL_FEATURES, drop_first=False)
    lr = LinearRegression().fit(Xn.values, y)
    pred = lr.predict(Xn.values)
    r2 = r2_score(y, pred)
    rmse = float(np.sqrt(mean_squared_error(y, pred)))

    contrib = (Xn.values - Xn.values.mean(axis=0)) * lr.coef_  # exact additive decomposition
    # Collapse one-hot columns back to the original categorical feature names.
    contrib_df = pd.DataFrame(contrib, columns=Xn.columns)
    collapsed = pd.DataFrame(index=contrib_df.index)
    for col in MODEL_FEATURES:
        member_cols = [c for c in Xn.columns if c == col or c.startswith(f"{col}_")]
        collapsed[col] = contrib_df[member_cols].sum(axis=1)
    shap_matrix = collapsed[MODEL_FEATURES].values
    metrics = {
        "model": "LinearRegression(synthetic SHAP surrogate)",
        "train_r2": r2, "test_r2": r2, "train_rmse": rmse, "test_rmse": rmse,
        "y_pred": pred,
    }
    shap_long = _to_long_shap(shap_matrix, X[MODEL_FEATURES], feature_matrix, target)
    return shap_long, metrics


def _to_long_shap(shap_matrix, X, feature_matrix, target) -> pd.DataFrame:
    """Convert an (n_wafers x n_features) SHAP matrix to the long CSV schema."""
    shap_df = pd.DataFrame(shap_matrix, columns=list(X.columns))
    long_rows = []
    for i in range(len(feature_matrix)):
        root_lot_id = feature_matrix.iloc[i]["root_lot_id"]
        wafer_id = feature_matrix.iloc[i]["wafer_id"]
        for feat in X.columns:
            sv = float(shap_df.iloc[i][feat])
            fv = feature_matrix.iloc[i][feat]
            long_rows.append(
                {
                    "root_lot_id": root_lot_id,
                    "wafer_id": wafer_id,
                    "target_id": TARGET_ID,
                    "feature_id": feat,
                    "feature_value": fv,
                    "shap_value": round(sv, 6),
                    "abs_shap_value": round(abs(sv), 6),
                    "shap_direction": "+" if sv >= 0 else "-",
                }
            )
    return pd.DataFrame(long_rows)


# --------------------------------------------------------------------------------------
# Static ontology artifacts (feature dictionary + causal-edge graph) and metadata CSVs
# --------------------------------------------------------------------------------------
def build_feature_dictionary() -> pd.DataFrame:
    """Per-feature metadata incl. the causal_role guardrail classification."""
    rows: List[Dict] = [
        dict(feature_id="fdc_cvd_pressure_slope_p95",
             display_name_ko="CVD 압력 기울기 P95(pressure slope)", source_type="FDC",
             process_step="CVD", module="CVD", metric_type="pressure_slope", aggregation="p95",
             wafer_region="chamber", wl_or_layer="-", observed_stage="in_situ_CVD",
             temporal_relation_to_target="before_target", causal_role="root_cause_candidate",
             leakage_risk="low", controllability="semi_controllable",
             mechanism_group="CVD_pressure_instability",
             interpretation_template_ko="CVD 챔버 압력 불안정(pressure instability)이 상류 원인 후보입니다."),
        dict(feature_id="fdc_cvd_rf_power_std",
             display_name_ko="CVD RF 파워 표준편차(RF power std)", source_type="FDC",
             process_step="CVD", module="CVD", metric_type="rf_power_std", aggregation="std",
             wafer_region="chamber", wl_or_layer="-", observed_stage="in_situ_CVD",
             temporal_relation_to_target="before_target", causal_role="root_cause_candidate",
             leakage_risk="low", controllability="semi_controllable",
             mechanism_group="CVD_RF_instability",
             interpretation_template_ko="RF 파워 변동(RF instability)이 압력 불안정과 동반되는 상류 신호입니다."),
        dict(feature_id="fdc_etch_temp_range",
             display_name_ko="Etch 온도 범위(temp range)", source_type="FDC",
             process_step="ETCH", module="ETCH", metric_type="temp_range", aggregation="range",
             wafer_region="chamber", wl_or_layer="-", observed_stage="in_situ_etch",
             temporal_relation_to_target="before_target", causal_role="root_cause_candidate",
             leakage_risk="low", controllability="controllable",
             mechanism_group="Etch_thermal",
             interpretation_template_ko="Etch 온도 범위는 부차적 공정 변동 후보입니다."),
        dict(feature_id="metro_thk_edge_wl128",
             display_name_ko="Edge 두께 비균일 wl128(edge thickness)", source_type="METROLOGY",
             process_step="CVD_POST_METRO", module="METRO", metric_type="thickness", aggregation="edge_mean",
             wafer_region="edge", wl_or_layer="wl128", observed_stage="post_CVD_metrology",
             temporal_relation_to_target="before_target", causal_role="mediator_candidate",
             leakage_risk="low", controllability="not_directly_controllable",
             mechanism_group="edge_thickness_nonuniformity",
             interpretation_template_ko="Edge 두께 비균일(non-uniformity)은 CVD 불안정의 결과인 중간 매개(mediator) 신호입니다."),
        dict(feature_id="metro_thk_edge_wl160",
             display_name_ko="Edge 두께 비균일 wl160(edge thickness)", source_type="METROLOGY",
             process_step="CVD_POST_METRO", module="METRO", metric_type="thickness", aggregation="edge_mean",
             wafer_region="edge", wl_or_layer="wl160", observed_stage="post_CVD_metrology",
             temporal_relation_to_target="before_target", causal_role="mediator_candidate",
             leakage_risk="low", controllability="not_directly_controllable",
             mechanism_group="edge_thickness_nonuniformity",
             interpretation_template_ko="Edge 두께 비균일(non-uniformity)은 CVD 불안정의 결과인 중간 매개(mediator) 신호입니다."),
        dict(feature_id="metro_cd_center_wl160",
             display_name_ko="Center CD wl160", source_type="METROLOGY",
             process_step="CVD_POST_METRO", module="METRO", metric_type="critical_dimension", aggregation="center_mean",
             wafer_region="center", wl_or_layer="wl160", observed_stage="post_CVD_metrology",
             temporal_relation_to_target="before_target", causal_role="mediator_candidate",
             leakage_risk="low", controllability="not_directly_controllable",
             mechanism_group="CD_center",
             interpretation_template_ko="Center CD는 약한 매개(mediator) 신호입니다."),
        dict(feature_id="queue_time_before_etch",
             display_name_ko="Etch 전 대기시간(queue time)", source_type="MES",
             process_step="QUEUE", module="LOGISTICS", metric_type="queue_time", aggregation="hours",
             wafer_region="lot", wl_or_layer="-", observed_stage="between_steps",
             temporal_relation_to_target="before_target", causal_role="root_cause_candidate",
             leakage_risk="low", controllability="controllable",
             mechanism_group="queue_time",
             interpretation_template_ko="Etch 전 대기시간(queue time)은 부차적 controllable 원인 후보입니다."),
        dict(feature_id="prev_defect_count_after_metro",
             display_name_ko="이전 결함 카운트(prior defect count)", source_type="DEFECT_INSPECTION",
             process_step="INSPECTION", module="DEFECT", metric_type="defect_count", aggregation="count",
             wafer_region="wafer", wl_or_layer="-", observed_stage="post_metro_inspection",
             temporal_relation_to_target="before_target", causal_role="proxy_indicator",
             leakage_risk="medium", controllability="not_controllable",
             mechanism_group="prior_defect_signal",
             interpretation_template_ko="이전 결함 카운트는 조기 경보(proxy)일 뿐 상류 원인이 아닙니다."),
        dict(feature_id="recipe_ppid_77a_flag",
             display_name_ko="PPID_77A 레시피 플래그(recipe flag)", source_type="MES",
             process_step="CVD", module="RECIPE", metric_type="flag", aggregation="binary",
             wafer_region="lot", wl_or_layer="-", observed_stage="recipe_assignment",
             temporal_relation_to_target="before_target", causal_role="confounder",
             leakage_risk="low", controllability="controllable",
             mechanism_group="recipe_grouping",
             interpretation_template_ko="PPID_77A는 원인과 결과 모두와 연관된 교란(confounder) 그룹 변수입니다."),
        dict(feature_id="product_id",
             display_name_ko="제품 ID(product)", source_type="MES",
             process_step="-", module="PRODUCT", metric_type="category", aggregation="-",
             wafer_region="lot", wl_or_layer="-", observed_stage="lot_attribute",
             temporal_relation_to_target="before_target", causal_role="confounder",
             leakage_risk="low", controllability="not_controllable",
             mechanism_group="product_grouping",
             interpretation_template_ko="제품 ID는 배경 교란(confounder) 변수입니다."),
        dict(feature_id="route_id",
             display_name_ko="라우트 ID(route)", source_type="MES",
             process_step="-", module="ROUTE", metric_type="category", aggregation="-",
             wafer_region="lot", wl_or_layer="-", observed_stage="lot_attribute",
             temporal_relation_to_target="before_target", causal_role="confounder",
             leakage_risk="low", controllability="not_controllable",
             mechanism_group="route_grouping",
             interpretation_template_ko="라우트 ID는 배경 교란(confounder) 변수입니다."),
        dict(feature_id="final_eds_proxy_leakage_feature",
             display_name_ko="최종 EDS 근접 지표(leakage proxy)", source_type="EDS",
             process_step="FINAL_TEST", module="EDS", metric_type="proxy", aggregation="-",
             wafer_region="wafer", wl_or_layer="-", observed_stage="after_target",
             temporal_relation_to_target="after_or_derived_from_target",
             causal_role="leakage_or_post_outcome",
             leakage_risk="high", controllability="not_controllable",
             mechanism_group="post_outcome_proxy",
             interpretation_template_ko="최종 EDS 지표는 target에서 파생된 누설(leakage)로, 인과 해석에서 제외합니다."),
    ]
    return pd.DataFrame(rows)


def build_causal_edges() -> pd.DataFrame:
    """Explicit ontology causal-relationship graph (DAG). Traversed by the hypothesis engine."""
    edges = [
        ("fdc_cvd_pressure_slope_p95", "metro_thk_edge_wl128", "physicallyAffects", 0.85),
        ("fdc_cvd_pressure_slope_p95", "metro_thk_edge_wl160", "physicallyAffects", 0.80),
        ("fdc_cvd_rf_power_std", "metro_thk_edge_wl160", "physicallyAffects", 0.60),
        ("metro_thk_edge_wl128", "defect_rate", "mediates", 0.80),
        ("metro_thk_edge_wl160", "defect_rate", "mediates", 0.75),
        ("metro_cd_center_wl160", "defect_rate", "mediates", 0.40),
        ("queue_time_before_etch", "defect_rate", "triggers", 0.40),
        ("prev_defect_count_after_metro", "defect_rate", "proxyOf", 0.70),
        ("recipe_ppid_77a_flag", "fdc_cvd_pressure_slope_p95", "propagatesTo", 0.50),
        ("final_eds_proxy_leakage_feature", "defect_rate", "proxyOf", 0.95),
    ]
    return pd.DataFrame(edges, columns=["source_feature", "target_feature", "relation", "confidence"])


def build_engineer_feedback() -> pd.DataFrame:
    """A few sample feedback rows (read-only in the app)."""
    return pd.DataFrame(
        [
            dict(hypothesis_id="HYP_001", engineer_judgment="타당(plausible)",
                 action_type="FDC_check", action_status="done",
                 outcome="CVD_CH_03 압력 센서 드리프트 확인", note="PM 주기 초과 구간과 일치"),
            dict(hypothesis_id="HYP_001", engineer_judgment="추가검증(needs_validation)",
                 action_type="recipe_review", action_status="in_progress",
                 outcome="PPID_77A 교란 여부 분리 필요", note="제품 PRODA 편중 의심"),
            dict(hypothesis_id="HYP_002", engineer_judgment="보류(hold)",
                 action_type="metro_recheck", action_status="planned",
                 outcome="-", note="edge 두께는 증상으로 판단"),
        ]
    )


def build_ground_truth() -> pd.DataFrame:
    return pd.DataFrame(
        [
            dict(
                true_root_feature=TRUE_ROOT_FEATURE,
                true_chain_text="fdc_cvd_pressure_slope_p95 -> metro_thk_edge_wl128 -> defect_rate",
                true_chamber=BAD_CHAMBER,
            )
        ]
    )


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def generate_all(data_dir: str = "data") -> dict:
    """Generate every CSV artifact. Returns a small summary dict (incl. SHAP mode + metrics)."""
    os.makedirs(data_dir, exist_ok=True)
    feature_matrix, target, process_history = generate_feature_matrix_and_target()
    shap_long, shap_mode, metrics = compute_shap_values(feature_matrix, target)

    feature_matrix.to_csv(os.path.join(data_dir, "feature_matrix.csv"), index=False)
    target.to_csv(os.path.join(data_dir, "target.csv"), index=False)
    shap_long.to_csv(os.path.join(data_dir, "shap_values.csv"), index=False)
    build_feature_dictionary().to_csv(os.path.join(data_dir, "feature_dictionary.csv"), index=False)
    process_history.to_csv(os.path.join(data_dir, "process_history.csv"), index=False)
    build_engineer_feedback().to_csv(os.path.join(data_dir, "engineer_feedback.csv"), index=False)
    build_ground_truth().to_csv(os.path.join(data_dir, "ground_truth.csv"), index=False)
    _write_causal_edges(os.path.join(data_dir, "causal_edges.csv"))

    # Persist model predictions for the README parity plot.
    pd.DataFrame(
        {"wafer_id": feature_matrix["wafer_id"], "y_true": target["defect_rate"], "y_pred": np.round(metrics["y_pred"], 4)}
    ).to_csv(os.path.join(data_dir, "model_predictions.csv"), index=False)

    summary = _report(feature_matrix, target, shap_long, shap_mode, metrics)
    return summary


def _write_causal_edges(path: str) -> None:
    """Write causal_edges.csv with the leading ontology-market header comment."""
    edges = build_causal_edges()
    header_comment = "# 이 파일을 편집해 도메인 인과관계를 추가/수정한다 (ontology market)\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(header_comment)
        edges.to_csv(f, index=False)


def _report(feature_matrix, target, shap_long, shap_mode, metrics) -> dict:
    """Print a console report and return a summary; sanity-checks the trap holds."""
    bad = target["bad_flag"] == 1
    # Mean |SHAP| per feature across bad wafers.
    bad_wafers = set(target.loc[bad, "wafer_id"])
    bad_shap = shap_long[shap_long["wafer_id"].isin(bad_wafers)]
    mean_abs = bad_shap.groupby("feature_id")["abs_shap_value"].mean().sort_values(ascending=False)

    feat_dict = build_feature_dictionary().set_index("feature_id")
    non_leak = [f for f in mean_abs.index if feat_dict.loc[f, "causal_role"] != "leakage_or_post_outcome"]
    naive_top_overall = mean_abs.index[0]
    naive_top_nonleak = next(f for f in mean_abs.index if f in non_leak)

    corr_med = np.corrcoef(feature_matrix["metro_thk_edge_wl128"], target["defect_rate"])[0, 1]
    corr_root = np.corrcoef(feature_matrix["fdc_cvd_pressure_slope_p95"], target["defect_rate"])[0, 1]

    print("=" * 78)
    print("ONTOLOGY-SHAP DEMO — data generation report")
    print("=" * 78)
    print(f"SHAP mode             : {shap_mode}")
    print(f"Model                 : {metrics['model']}")
    print(f"train R2 / test R2    : {metrics['train_r2']:.3f} / {metrics['test_r2']:.3f}")
    print(f"train RMSE / test RMSE: {metrics['train_rmse']:.3f} / {metrics['test_rmse']:.3f}")
    print(f"#wafers / #bad        : {len(feature_matrix)} / {int(bad.sum())}")
    print(f"corr(target, mediator edge_thk_wl128) = {corr_med:.3f}")
    print(f"corr(target, root  cvd_pressure)      = {corr_root:.3f}")
    print(f"  -> trap holds (mediator > root corr): {corr_med > corr_root}")
    print("-" * 78)
    print("Mean |SHAP| over bad wafers (top 6):")
    for f, v in mean_abs.head(6).items():
        role = feat_dict.loc[f, "causal_role"]
        print(f"   {v:8.4f}  {f:38s} [{role}]")
    print("-" * 78)
    print(f"naive top-SHAP overall            : {naive_top_overall} "
          f"[{feat_dict.loc[naive_top_overall, 'causal_role']}]")
    print(f"naive top-SHAP (excl. leakage)    : {naive_top_nonleak} "
          f"[{feat_dict.loc[naive_top_nonleak, 'causal_role']}]")
    print("=" * 78)

    return {
        "shap_mode": shap_mode,
        "metrics": {k: v for k, v in metrics.items() if k != "y_pred"},
        "n_wafers": int(len(feature_matrix)),
        "n_bad": int(bad.sum()),
        "naive_top_overall": naive_top_overall,
        "naive_top_nonleak": naive_top_nonleak,
        "corr_mediator": float(corr_med),
        "corr_root": float(corr_root),
        "trap_holds": bool(corr_med > corr_root),
    }


if __name__ == "__main__":
    generate_all("data")
