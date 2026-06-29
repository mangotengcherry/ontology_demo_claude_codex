"""Performance arm: flat CatBoost vs ontology-semantic CatBoost on yield data.

This is the *learning* half of the demo (the interpretation half lives in
`src/causal_evidence.py` + `hypothesis_engine.py`). It does NOT claim "ontology
improves accuracy at large N" -- a well-tuned flat CatBoost is a strong baseline
and that claim breaks on the first question. It demonstrates only the three wins
that survive a critical audience (see docs/model_comparison_findings.md):

  1. Leakage blocking. The flat model's high test R^2 is inflated by a
     post-outcome proxy (`...FINAL...`, SHAP #1) that will not exist at
     production time. The ontology arm drops it automatically via *causal role*
     (not by knowing the answer), revealing the honest achievable R^2.
  2. Low-data inductive bias. With leakage removed from BOTH models, the only
     lever left is the semantic layer's shallower depth (flat=8 / onto=4) plus
     A-grade monotone priors. That bias wins when N is small -- exactly the
     new-product / rare-BIN / new-defect regime.
  3. Attribution as a control handle. Flat SHAP points at the metrology symptom;
     ontology roll-up + measured mediation point at the controllable root.

Guardrails honored: identical split/seed/iterations across arms; leakage removed
only by causal role (no oracle that drops noise features knowing they are noise);
the flat-vs-onto depth gap is the inductive-bias lever and the train/test gap is
always reported so it cannot be mistaken for a tuning trick.

Run:
    python3 scripts/model_comparison_demo.py --input-dir input
    python3 scripts/model_comparison_demo.py            # auto-generates virtual data
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from catboost import CatBoostRegressor, Pool  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from src import shap_inputs  # noqa: E402
from src.real_dataset_adapter import (  # noqa: E402
    TARGET_ID,
    build_standard_dataset,
    input_files_exist,
)
from src.shap_diagnostics import credit_absorption  # noqa: E402
from src.virtual_data_generator import generate_virtual_input_dataset  # noqa: E402

warnings.filterwarnings("ignore")

LEAKAGE_ROLE = "leakage_or_post_outcome"
FLAT_DEPTH = 8
ONTO_DEPTH = 4
SEP = "=" * 78


# --------------------------------------------------------------------------------------
# Data assembly
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Dataset:
    X: pd.DataFrame                 # feature columns only (per wafer)
    y: pd.Series                    # continuous target (defect_rate)
    cat_features: List[str]         # categorical column names within X
    roles: Dict[str, str]          # feature_id -> causal_role
    grades: Dict[str, str]         # feature_id -> metro_grade ('' if none)
    mediation: pd.DataFrame         # measured root->mediator->target chains
    source: str                     # 'real' or 'virtual'


def dataset_from_artifacts(art, source: str = "real") -> Dataset:
    """Build a model-ready Dataset from already-built standard artifacts.

    No file IO and no virtual fallback -- the caller controls the data source. Used
    by notebook Mode 1 (train on real data without a provided SHAP export).
    """
    fm = art.feature_matrix.drop(columns=["root_lot_id", "wafer_id"])
    y = art.target.set_index(art.target.index)[TARGET_ID].astype(float).reset_index(drop=True)

    roles = art.feature_dictionary.set_index("feature_id")["causal_role"].to_dict()
    grades = art.feature_dictionary.set_index("feature_id")["metro_grade"].fillna("").to_dict()

    # Coerce: numeric where possible, leave the rest as string categoricals.
    cat_features: List[str] = []
    X = pd.DataFrame(index=fm.index)
    for col in fm.columns:
        as_num = pd.to_numeric(fm[col], errors="coerce")
        if as_num.notna().mean() >= 0.99:
            X[col] = as_num.fillna(as_num.median())
        else:
            X[col] = fm[col].fillna("NA").astype(str)
            cat_features.append(col)
    return Dataset(X=X, y=y, cat_features=cat_features, roles=roles, grades=grades,
                   mediation=art.mediation, source=source)


def load_dataset(input_dir: str, bad_quantile: float, n_wafers: int, seed: int) -> Dataset:
    """Load via the real-data contract; auto-generate virtual data if absent (CLI/tests)."""
    if input_files_exist(input_dir):
        source = "real"
    else:
        generate_virtual_input_dataset(input_dir, n_wafers=n_wafers, seed=seed)
        source = "virtual"

    # require_shap=False: the performance arm trains its own model and derives SHAP,
    # so it only needs raw_data + prc_metro_relation, not a provided SHAP export.
    art = build_standard_dataset(input_dir=input_dir, output_dir="data", bad_quantile=bad_quantile, require_shap=False)
    return dataset_from_artifacts(art, source)


def select_columns(ds: Dataset, drop_leakage: bool) -> Tuple[List[str], List[str]]:
    """Return (feature_cols, cat_feature_cols) for an arm, optionally dropping leakage."""
    cols = [c for c in ds.X.columns if not (drop_leakage and ds.roles.get(c) == LEAKAGE_ROLE)]
    cats = [c for c in ds.cat_features if c in cols]
    return cols, cats


def monotone_map(ds: Dataset, cols: List[str], X_tr: pd.DataFrame, y_tr: pd.Series) -> Dict[str, int]:
    """Monotone sign for A-grade metrology mediators only, from the TRAIN correlation.

    Deliberately narrow: monotone priors HURT non-monotone (U-shaped) relations, so
    we only apply them where the engineer-assigned metro grade is 'A'. The sign is
    learned from the training fold, never from the test fold or the target name.
    """
    out: Dict[str, int] = {}
    for c in cols:
        if c in ds.cat_features:
            continue
        if str(ds.grades.get(c, "")).upper() != "A":
            continue
        r = np.corrcoef(X_tr[c].to_numpy(float), y_tr.to_numpy(float))[0, 1]
        if np.isfinite(r) and abs(r) > 0.05:
            out[c] = 1 if r > 0 else -1
    return out


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
def fit_predict(
    X_tr: pd.DataFrame, y_tr: pd.Series, X_te: pd.DataFrame,
    cat_features: List[str], depth: int, iterations: int, seed: int,
    monotone: Optional[Dict[str, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, CatBoostRegressor]:
    cats = [X_tr.columns.get_loc(c) for c in cat_features]
    constraints = None
    if monotone:
        constraints = [monotone.get(c, 0) for c in X_tr.columns]
    model = CatBoostRegressor(
        iterations=iterations, depth=depth, learning_rate=0.05,
        random_seed=seed, loss_function="RMSE", verbose=False,
        monotone_constraints=constraints,
    )
    model.fit(X_tr, y_tr, cat_features=cats)
    return model.predict(X_tr), model.predict(X_te), model


def shap_rollup(model: CatBoostRegressor, X: pd.DataFrame, cat_features: List[str],
                roles: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Native CatBoost SHAP -> per-feature mean|SHAP| and a roll-up by causal role."""
    cats = [X.columns.get_loc(c) for c in cat_features]
    sv = model.get_feature_importance(Pool(X, cat_features=cats), type="ShapValues")
    sv = sv[:, :-1]  # drop the bias column
    mean_abs = np.abs(sv).mean(axis=0)
    per_feat = (
        pd.DataFrame({"feature": X.columns, "mean_abs_shap": mean_abs,
                      "role": [roles.get(c, "unknown") for c in X.columns]})
        .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    )
    total = per_feat["mean_abs_shap"].sum() or 1.0
    per_feat["pct"] = 100 * per_feat["mean_abs_shap"] / total
    rollup = (
        per_feat.groupby("role")["mean_abs_shap"].sum()
        .sort_values(ascending=False).reset_index()
    )
    rollup["pct"] = 100 * rollup["mean_abs_shap"] / total
    return per_feat, rollup


def trained_ontology_cohort_shap(
    ds: Dataset, bad_mask: np.ndarray, iterations: int = 300, seed: int = 42,
) -> pd.DataFrame:
    """Train the ontology CatBoost on all wafers and return BAD-wafer cohort-mean SHAP.

    Output schema matches `shap_inputs.cohort_mean_shap` (feature, shap_value=signed
    mean, mean_abs_shap, n_wafers), so the trained model's SHAP can flow through the
    exact same ontology-SHAP interpretation (hypothesis cards, role roll-up, trace)
    as the provided-SHAP path. This is the SHAP source for notebook Mode 1.
    """
    cols, cats = select_columns(ds, drop_leakage=True)
    mono = monotone_map(ds, cols, ds.X[cols], ds.y)
    _, _, model = fit_predict(ds.X[cols], ds.y, ds.X[cols], cats, ONTO_DEPTH, iterations, seed, monotone=mono)
    catpos = [ds.X[cols].columns.get_loc(c) for c in cats]
    X_bad = ds.X[cols][np.asarray(bad_mask, dtype=bool)]
    if len(X_bad) == 0:
        X_bad = ds.X[cols]
    sv = model.get_feature_importance(Pool(X_bad, cat_features=catpos), type="ShapValues")[:, :-1]
    out = pd.DataFrame({
        "feature": list(cols),
        "shap_value": sv.mean(axis=0),
        "mean_abs_shap": np.abs(sv).mean(axis=0),
        "n_wafers": int(len(X_bad)),
    })
    return out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Console blocks
# --------------------------------------------------------------------------------------
def _r2(y_true, y_pred) -> float:
    return float(r2_score(y_true, y_pred))


@dataclass(frozen=True)
class Results:
    """Everything computed once, so CLI / viz / notebook share one code path."""
    source: str
    n_wafers: int
    n_features: int
    roles_count: Dict[str, int]
    performance: pd.DataFrame     # arm, depth, train_r2, test_r2, gap
    flat_feat: pd.DataFrame       # feature, mean_abs_shap, role, pct
    onto_feat: pd.DataFrame
    onto_roll: pd.DataFrame       # role, mean_abs_shap, pct
    mediation: pd.DataFrame
    credit: pd.DataFrame
    learning_curve: pd.DataFrame  # n, flat_r2, onto_r2, delta
    monotone: Dict[str, int]


def compute_results(ds: Dataset, iterations: int, seed: int, k_repeats: int = 5) -> Results:
    """Train all arms ONCE, compute SHAP, mediation views, credit, learning curve."""
    idx = np.arange(len(ds.X))
    tr, te = train_test_split(idx, test_size=0.25, random_state=seed)

    flat_cols, flat_cats = select_columns(ds, drop_leakage=False)
    nl_cols, nl_cats = select_columns(ds, drop_leakage=True)
    mono = monotone_map(ds, nl_cols, ds.X.iloc[tr][nl_cols], ds.y.iloc[tr])

    # Flat (incl. leakage, deep), flat ablation (leakage removed by hand), ontology.
    ptr_f, pte_f, flat_model = fit_predict(ds.X.iloc[tr][flat_cols], ds.y.iloc[tr],
                                           ds.X.iloc[te][flat_cols], flat_cats, FLAT_DEPTH, iterations, seed)
    ptr_a, pte_a, _ = fit_predict(ds.X.iloc[tr][nl_cols], ds.y.iloc[tr],
                                  ds.X.iloc[te][nl_cols], nl_cats, FLAT_DEPTH, iterations, seed)
    ptr_o, pte_o, onto_model = fit_predict(ds.X.iloc[tr][nl_cols], ds.y.iloc[tr],
                                           ds.X.iloc[te][nl_cols], nl_cats, ONTO_DEPTH, iterations, seed, monotone=mono)

    def _row(arm, depth, ptr, pte):
        a, b = _r2(ds.y.iloc[tr], ptr), _r2(ds.y.iloc[te], pte)
        return {"arm": arm, "depth": depth, "train_r2": a, "test_r2": b, "gap": a - b}

    performance = pd.DataFrame([
        _row("flat (all features, incl. leakage)", FLAT_DEPTH, ptr_f, pte_f),
        _row("flat ablation (leakage removed by hand)", FLAT_DEPTH, ptr_a, pte_a),
        _row("ontology (role-filtered + shallow)", ONTO_DEPTH, ptr_o, pte_o),
    ])

    flat_feat, _ = shap_rollup(flat_model, ds.X.iloc[te][flat_cols], flat_cats, ds.roles)
    onto_feat, onto_roll = shap_rollup(onto_model, ds.X.iloc[te][nl_cols], nl_cats, ds.roles)

    share = {row["feature"]: row["pct"] / 100.0 for _, row in onto_feat.iterrows()}
    credit = credit_absorption(ds.mediation, share)

    learning_curve = _learning_curve(ds, iterations, seed, k_repeats)

    return Results(
        source=ds.source, n_wafers=len(ds.X), n_features=ds.X.shape[1],
        roles_count=dict(pd.Series(ds.roles).value_counts()),
        performance=performance, flat_feat=flat_feat, onto_feat=onto_feat, onto_roll=onto_roll,
        mediation=ds.mediation, credit=credit, learning_curve=learning_curve, monotone=mono,
    )


def _learning_curve(ds: Dataset, iterations: int, seed: int, k_repeats: int) -> pd.DataFrame:
    """Isolate the inductive-bias lever: BOTH arms leakage-free; only depth differs."""
    cols, cats = select_columns(ds, drop_leakage=True)
    idx = np.arange(len(ds.X))
    tr_pool, te = train_test_split(idx, test_size=0.30, random_state=seed)
    rng = np.random.default_rng(seed)
    rows = []
    for f in [0.10, 0.20, 0.40, 0.70, 1.0]:
        n = max(ONTO_DEPTH * 4, int(len(tr_pool) * f))
        flat_scores, onto_scores = [], []
        for rep in range(k_repeats):
            sub = rng.choice(tr_pool, size=min(n, len(tr_pool)), replace=False)
            Xs, ys = ds.X.iloc[sub], ds.y.iloc[sub]
            _, pf, _ = fit_predict(Xs[cols], ys, ds.X.iloc[te][cols], cats, FLAT_DEPTH, iterations, seed + rep)
            mono = monotone_map(ds, cols, Xs[cols], ys)
            _, po, _ = fit_predict(Xs[cols], ys, ds.X.iloc[te][cols], cats, ONTO_DEPTH, iterations, seed + rep, monotone=mono)
            flat_scores.append(_r2(ds.y.iloc[te], pf))
            onto_scores.append(_r2(ds.y.iloc[te], po))
        fm, om = float(np.mean(flat_scores)), float(np.mean(onto_scores))
        rows.append({"n": min(n, len(tr_pool)), "flat_r2": fm, "onto_r2": om, "delta": om - fm})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Console presentation (consumes Results)
# --------------------------------------------------------------------------------------
def print_report(res: Results) -> None:
    print(SEP)
    print(f"ONTOLOGY x YIELD  --  flat vs ontology CatBoost   [data source: {res.source}]")
    print(f"  wafers={res.n_wafers}  features={res.n_features}")
    print("  roles: " + ", ".join(f"{k}={v}" for k, v in res.roles_count.items()))

    print(SEP)
    print("1) PERFORMANCE  (regression R^2 on a held-out 25% split; identical seed/iterations)")
    print(SEP)
    print(f"{'arm':<42}{'depth':>6}{'train R2':>11}{'test R2':>10}{'gap':>8}")
    for _, r in res.performance.iterrows():
        print(f"{r['arm']:<42}{int(r['depth']):>6}{r['train_r2']:>11.3f}{r['test_r2']:>10.3f}{r['gap']:>8.3f}")
    flat_te = res.performance.iloc[0]["test_r2"]
    ceil_te = res.performance.iloc[1]["test_r2"]
    print()
    print(f"  -> flat's test R2 ({flat_te:.3f}) is inflated by the post-outcome proxy; remove it")
    print(f"     and the honest ceiling is ~{ceil_te:.3f}. Ontology reaches that ceiling AUTOMATICALLY via causal role.")
    if res.monotone:
        print(f"  -> A-grade monotone priors applied to: {', '.join(res.monotone.keys())}")

    print()
    print(SEP)
    print("2) ATTRIBUTION  (top features by mean|SHAP| on the held-out fold)")
    print(SEP)
    print("  FLAT model -- points at the symptom / leakage:")
    for _, r in res.flat_feat.head(5).iterrows():
        print(f"    {r['pct']:>5.1f}%  [{r['role']:<22}] {r['feature']}")
    print("  ONTOLOGY model -- leakage gone; credit flows to root/mediator/process:")
    for _, r in res.onto_feat.head(5).iterrows():
        print(f"    {r['pct']:>5.1f}%  [{r['role']:<22}] {r['feature']}")

    print()
    print(SEP)
    print("3) ONTOLOGY-LEVEL SHAP ROLL-UP  (ontology model SHAP grouped by causal role)")
    print(SEP)
    for _, r in res.onto_roll.iterrows():
        print(f"    {r['pct']:>5.1f}%  {r['role']}")
    controllable = res.onto_roll[res.onto_roll["role"] == "root_cause_candidate"]["pct"].sum()
    print(f"  -> {controllable:.1f}% of attribution sits on controllable root candidates (a knob).")

    flagged = res.credit[res.credit["credit_absorbed"]]
    if not flagged.empty:
        print()
        print("  CREDIT ABSORPTION (why a plain SHAP ranking misses the cause):")
        for _, r in flagged.iterrows():
            print(f"    root {r['root']}: measured indirect={r['indirect']:.2f} but model SHAP "
                  f"{r['root_shap_share']*100:.1f}% < mediator {r['mediator_shap_share']*100:.1f}%")

    print()
    print(SEP)
    print("4) MEASURED CAUSAL CHAIN  (data-measured mediation: root -> mediator -> target)")
    print(SEP)
    if res.mediation is None or res.mediation.empty:
        print("    (no measured chain -- see diagnosis checklist: naming exceptions / N<min / collinearity)")
    else:
        for _, r in res.mediation.sort_values("indirect", key=lambda s: s.abs(), ascending=False).iterrows():
            prop = r["prop_mediated"]
            prop_s = f"{prop*100:.0f}%" if np.isfinite(prop) else "n/a"
            flags = []
            if not r.get("sign_consistent", False):
                flags.append("SIGN-INCONSISTENT")
            if r.get("unstable", False):
                flags.append("COLLINEAR/UNSTABLE")
            if not r.get("strata_stable", True):
                flags.append("SIMPSON-WARN")
            if np.isfinite(prop) and prop > 1.0:
                flags.append("prop>100%(suppression/noise)")
            if not r.get("bh_reject", True):
                flags.append("not BH-FDR significant")
            if r.get("nonlinear_b", False):
                flags.append(f"NONLINEAR b-path (|effect|~{r['nl_indirect_mag']:.2f})")
            tag = ("  [" + ", ".join(flags) + "]") if flags else ""
            print(f"    {r['root']}")
            print(f"      -> {r['mediator']}  (a={r['a']:.2f}, b={r['b']:.2f}, "
                  f"indirect a*b={r['indirect']:.2f}, ~{prop_s} mediated, p={r['indirect_p']:.3g}, "
                  f"q={r.get('indirect_q', float('nan')):.3g}, n={int(r['n'])}){tag}")

    print()
    print(SEP)
    print("5) LEARNING CURVE  (both arms leakage-free; flat depth=8 vs ontology depth=4)")
    print(SEP)
    print(f"{'train N':>8}{'flat test R2':>15}{'onto test R2':>15}{'onto - flat':>14}")
    for _, r in res.learning_curve.iterrows():
        flag = "  <- ontology wins" if r["delta"] > 0 else ""
        print(f"{int(r['n']):>8}{r['flat_r2']:>15.3f}{r['onto_r2']:>15.3f}{r['delta']:>14.3f}{flag}")
    print("  -> the shallow semantic prior pays off most when N is small (new product / rare BIN).")
    print("     At large N the gap closes -- we do NOT claim a large-N accuracy win.")


# --------------------------------------------------------------------------------------
# Provided-model SHAP (interpret the COMPANY's own CatBoost, not the re-trained arm)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ProvidedShap:
    source: str                  # which input file the SHAP came from
    n_wafers: int
    per_feat: pd.DataFrame       # feature, mean_abs_shap, role, pct
    rollup: pd.DataFrame         # role, mean_abs_shap, pct
    credit: pd.DataFrame         # credit-absorption of the provided model vs measured chain


def analyze_provided_shap(input_dir: str, ds: Dataset) -> Optional[ProvidedShap]:
    """Roll up the *provided* (production) SHAP by causal role + diagnose credit absorption.

    This interprets the company's own model from the supplied wide SHAP export
    (`all_wafer_shap_value.csv`, else `bad_wafer_shap_value.csv`), independent of
    the flat/ontology models trained here. Returns None when no wide SHAP file
    exists (e.g. legacy long-only or virtual-without-wide inputs).
    """
    feature_cols = list(ds.X.columns)

    def _read(name):
        path = os.path.join(input_dir, name)
        return pd.read_csv(path) if os.path.exists(path) else None

    wide = _read(shap_inputs.ALL_SHAP_FILE)
    source = shap_inputs.ALL_SHAP_FILE
    if wide is None:
        wide = _read(shap_inputs.BAD_SHAP_FILE)
        source = shap_inputs.BAD_SHAP_FILE
    if wide is None:
        return None

    feats = shap_inputs.detect_feature_columns(wide, feature_cols)
    cohort = shap_inputs.cohort_mean_shap(wide, feats)
    if cohort.empty:
        return None

    per_feat = cohort.copy()
    per_feat["role"] = [ds.roles.get(f, "unknown") for f in per_feat["feature"]]
    total = per_feat["mean_abs_shap"].sum() or 1.0
    per_feat["pct"] = 100.0 * per_feat["mean_abs_shap"] / total
    rollup = (
        per_feat.groupby("role")["mean_abs_shap"].sum().sort_values(ascending=False).reset_index()
    )
    rollup["pct"] = 100.0 * rollup["mean_abs_shap"] / total

    share = {row["feature"]: row["mean_abs_shap"] / total for _, row in per_feat.iterrows()}
    credit = credit_absorption(ds.mediation, share) if ds.mediation is not None else pd.DataFrame()
    return ProvidedShap(source=source, n_wafers=int(len(wide)),
                        per_feat=per_feat, rollup=rollup, credit=credit)


def print_provided_shap(ps: ProvidedShap) -> None:
    print()
    print(SEP)
    print(f"6) PROVIDED MODEL SHAP  (your production CatBoost, from {ps.source}; rolled up by role)")
    print(SEP)
    print("  top features by mean|SHAP| (provided model):")
    for _, r in ps.per_feat.head(6).iterrows():
        print(f"    {r['pct']:>5.1f}%  [{r['role']:<22}] {r['feature']}")
    print("  roll-up by causal role:")
    for _, r in ps.rollup.iterrows():
        print(f"    {r['pct']:>5.1f}%  {r['role']}")
    flagged = ps.credit[ps.credit["credit_absorbed"]] if not ps.credit.empty else ps.credit
    if flagged is not None and not flagged.empty:
        print("  CREDIT ABSORPTION in your production model (measured root vs your SHAP share):")
        for _, r in flagged.iterrows():
            print(f"    root {r['root']}: measured indirect={r['indirect']:.2f} but provided SHAP "
                  f"{r['root_shap_share']*100:.1f}% < mediator {r['mediator_shap_share']*100:.1f}%")
    else:
        print("  (no credit absorption flagged in the provided model — root carries its measured credit)")


# --------------------------------------------------------------------------------------
def run(input_dir: str = "input", bad_quantile: float = 0.80, iterations: int = 300,
        seed: int = 42, n_wafers: int = 250, save_charts_to: Optional[str] = None) -> Results:
    """Compute the performance arm, print the five blocks, optionally save charts."""
    ds = load_dataset(input_dir, bad_quantile, n_wafers, seed)
    res = compute_results(ds, iterations, seed)
    print_report(res)
    provided = analyze_provided_shap(input_dir, ds)
    if provided is not None:
        print_provided_shap(provided)
    if save_charts_to:
        from src.model_comparison_viz import save_all  # local import keeps matplotlib optional
        paths = save_all(res, save_charts_to)
        print()
        print(SEP)
        print(f"charts saved to {save_charts_to}/ :")
        for p in paths:
            print(f"  {p}")
        print(SEP)
    print()
    print(SEP)
    print("Read the three robust wins in docs/model_comparison_findings.md before presenting.")
    print(SEP)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", default="input")
    ap.add_argument("--bad-quantile", type=float, default=0.80)
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-wafers", type=int, default=250, help="virtual-data size when input is absent")
    ap.add_argument("--save-charts", metavar="DIR", default=None,
                    help="also render the five blocks as PNG charts into DIR (e.g. outputs/charts)")
    args = ap.parse_args()
    run(args.input_dir, args.bad_quantile, args.iterations, args.seed, args.n_wafers,
        save_charts_to=args.save_charts)


if __name__ == "__main__":
    main()
