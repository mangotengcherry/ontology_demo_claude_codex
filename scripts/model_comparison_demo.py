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


def load_dataset(input_dir: str, bad_quantile: float, n_wafers: int, seed: int) -> Dataset:
    """Load via the real-data contract; auto-generate virtual data if absent."""
    if input_files_exist(input_dir):
        source = "real"
    else:
        generate_virtual_input_dataset(input_dir, n_wafers=n_wafers, seed=seed)
        source = "virtual"

    art = build_standard_dataset(input_dir=input_dir, output_dir="data", bad_quantile=bad_quantile)
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


# --------------------------------------------------------------------------------------
# Console blocks
# --------------------------------------------------------------------------------------
def _r2(y_true, y_pred) -> float:
    return float(r2_score(y_true, y_pred))


def block_performance(ds: Dataset, iterations: int, seed: int) -> Dict[str, object]:
    idx = np.arange(len(ds.X))
    tr, te = train_test_split(idx, test_size=0.25, random_state=seed)

    # Flat: everything (incl. leakage), deep.
    flat_cols, flat_cats = select_columns(ds, drop_leakage=False)
    p_tr, p_te, _ = fit_predict(ds.X.iloc[tr][flat_cols], ds.y.iloc[tr], ds.X.iloc[te][flat_cols],
                                flat_cats, FLAT_DEPTH, iterations, seed)
    flat = (_r2(ds.y.iloc[tr], p_tr), _r2(ds.y.iloc[te], p_te))

    # Flat ablation: same deep model but leakage removed by hand (proves the gap IS leakage).
    nl_cols, nl_cats = select_columns(ds, drop_leakage=True)
    p_tr2, p_te2, _ = fit_predict(ds.X.iloc[tr][nl_cols], ds.y.iloc[tr], ds.X.iloc[te][nl_cols],
                                  nl_cats, FLAT_DEPTH, iterations, seed)
    flat_nl = (_r2(ds.y.iloc[tr], p_tr2), _r2(ds.y.iloc[te], p_te2))

    # Ontology: leakage dropped by causal role, shallow, A-grade monotone priors.
    mono = monotone_map(ds, nl_cols, ds.X.iloc[tr][nl_cols], ds.y.iloc[tr])
    p_tr3, p_te3, onto_model = fit_predict(ds.X.iloc[tr][nl_cols], ds.y.iloc[tr], ds.X.iloc[te][nl_cols],
                                           nl_cats, ONTO_DEPTH, iterations, seed, monotone=mono)
    onto = (_r2(ds.y.iloc[tr], p_tr3), _r2(ds.y.iloc[te], p_te3))

    print(SEP)
    print("1) PERFORMANCE  (regression R^2 on a held-out 25% split; identical seed/iterations)")
    print(SEP)
    print(f"{'arm':<42}{'depth':>6}{'train R2':>11}{'test R2':>10}{'gap':>8}")
    print(f"{'flat (all features, incl. leakage)':<42}{FLAT_DEPTH:>6}{flat[0]:>11.3f}{flat[1]:>10.3f}{flat[0]-flat[1]:>8.3f}")
    print(f"{'flat ablation (leakage removed by hand)':<42}{FLAT_DEPTH:>6}{flat_nl[0]:>11.3f}{flat_nl[1]:>10.3f}{flat_nl[0]-flat_nl[1]:>8.3f}")
    print(f"{'ontology (role-filtered + shallow)':<42}{ONTO_DEPTH:>6}{onto[0]:>11.3f}{onto[1]:>10.3f}{onto[0]-onto[1]:>8.3f}")
    print()
    print(f"  -> flat's test R2 ({flat[1]:.3f}) is inflated by the post-outcome proxy; remove it")
    print(f"     and the honest ceiling is ~{flat_nl[1]:.3f}. Ontology reaches that ceiling AUTOMATICALLY")
    print(f"     via causal role, and the monotone A-grade prior is shown in ATTRIBUTION below.")
    if mono:
        print(f"  -> A-grade monotone priors applied to: {', '.join(mono.keys())}")
    return {"flat": flat, "flat_nl": flat_nl, "onto": onto, "tr": tr, "te": te,
            "nl_cols": nl_cols, "nl_cats": nl_cats, "mono": mono, "onto_model": onto_model,
            "flat_cols": flat_cols, "flat_cats": flat_cats}


def block_attribution(ds: Dataset, ctx: Dict[str, object], iterations: int, seed: int) -> None:
    tr, te = ctx["tr"], ctx["te"]
    # Flat model SHAP (leakage present) vs ontology model SHAP.
    _, _, flat_model = fit_predict(ds.X.iloc[tr][ctx["flat_cols"]], ds.y.iloc[tr],
                                   ds.X.iloc[te][ctx["flat_cols"]], ctx["flat_cats"],
                                   FLAT_DEPTH, iterations, seed)
    flat_feat, _ = shap_rollup(flat_model, ds.X.iloc[te][ctx["flat_cols"]], ctx["flat_cats"], ds.roles)
    onto_feat, onto_roll = shap_rollup(ctx["onto_model"], ds.X.iloc[te][ctx["nl_cols"]],
                                       ctx["nl_cats"], ds.roles)

    print()
    print(SEP)
    print("2) ATTRIBUTION  (top features by mean|SHAP| on the held-out fold)")
    print(SEP)
    print("  FLAT model -- points at the symptom / leakage:")
    for _, r in flat_feat.head(5).iterrows():
        print(f"    {r['pct']:>5.1f}%  [{r['role']:<22}] {r['feature']}")
    print("  ONTOLOGY model -- leakage gone; credit flows to root/mediator/process:")
    for _, r in onto_feat.head(5).iterrows():
        print(f"    {r['pct']:>5.1f}%  [{r['role']:<22}] {r['feature']}")

    print()
    print(SEP)
    print("3) ONTOLOGY-LEVEL SHAP ROLL-UP  (ontology model SHAP grouped by causal role)")
    print(SEP)
    for _, r in onto_roll.iterrows():
        print(f"    {r['pct']:>5.1f}%  {r['role']}")
    controllable = onto_roll[onto_roll["role"].isin(["root_cause_candidate"])]["pct"].sum()
    print(f"  -> {controllable:.1f}% of attribution sits on controllable root candidates (a knob),")
    print(f"     not on the metrology symptom alone.")

    # Credit-absorption diagnostic: measured root indirect effect vs model SHAP share.
    share = {row["feature"]: row["pct"] / 100.0 for _, row in onto_feat.iterrows()}
    ca = credit_absorption(ds.mediation, share)
    flagged = ca[ca["credit_absorbed"]]
    if not flagged.empty:
        print()
        print("  CREDIT ABSORPTION (why a plain SHAP ranking misses the cause):")
        for _, r in flagged.iterrows():
            print(f"    root {r['root']}")
            print(f"      measured indirect={r['indirect']:.2f} but model SHAP share "
                  f"{r['root_shap_share']*100:.1f}% < mediator {r['mediator_shap_share']*100:.1f}%"
                  f"  -> mediator absorbed the credit")


def block_chain(ds: Dataset) -> None:
    print()
    print(SEP)
    print("4) MEASURED CAUSAL CHAIN  (data-measured mediation: root -> mediator -> target)")
    print(SEP)
    med = ds.mediation
    if med is None or med.empty:
        print("    (no measured chain -- see diagnosis checklist: naming exceptions / N<min / collinearity)")
        return
    keep = med.sort_values("indirect", key=lambda s: s.abs(), ascending=False)
    for _, r in keep.iterrows():
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
            flags.append(f"NONLINEAR b-path (lin indirect underestimates; |effect|~{r['nl_indirect_mag']:.2f})")
        tag = ("  [" + ", ".join(flags) + "]") if flags else ""
        q = r.get("indirect_q", float("nan"))
        print(f"    {r['root']}")
        print(f"      -> {r['mediator']}  (a={r['a']:.2f}, b={r['b']:.2f}, "
              f"indirect a*b={r['indirect']:.2f}, ~{prop_s} mediated, p={r['indirect_p']:.3g}, "
              f"q={q:.3g}, n={int(r['n'])}){tag}")


def block_learning_curve(ds: Dataset, iterations: int, seed: int, k_repeats: int = 5) -> None:
    """Isolate the inductive-bias lever: BOTH arms leakage-free; only depth differs.

    Leakage win (#1) is already shown above; here we hold leakage out of both arms so
    the curve measures only the semantic layer's shallow-depth / monotone bias.
    """
    cols, cats = select_columns(ds, drop_leakage=True)
    idx = np.arange(len(ds.X))
    tr_pool, te = train_test_split(idx, test_size=0.30, random_state=seed)
    fracs = [0.10, 0.20, 0.40, 0.70, 1.0]
    rng = np.random.default_rng(seed)

    print()
    print(SEP)
    print("5) LEARNING CURVE  (both arms leakage-free; flat depth=8 vs ontology depth=4)")
    print(SEP)
    print(f"{'train N':>8}{'flat test R2':>15}{'onto test R2':>15}{'onto - flat':>14}")
    for f in fracs:
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
        fm, om = np.mean(flat_scores), np.mean(onto_scores)
        flag = "  <- ontology wins" if om > fm else ""
        print(f"{n:>8}{fm:>15.3f}{om:>15.3f}{om-fm:>14.3f}{flag}")
    print("  -> the shallow semantic prior pays off most when N is small (new product / rare BIN).")
    print("     At large N the gap closes -- we do NOT claim a large-N accuracy win.")


# --------------------------------------------------------------------------------------
def run(input_dir: str = "input", bad_quantile: float = 0.80, iterations: int = 300,
        seed: int = 42, n_wafers: int = 250) -> None:
    """Run all five performance-arm blocks. Reusable from run_demo.py."""
    ds = load_dataset(input_dir, bad_quantile, n_wafers, seed)
    print(SEP)
    print(f"ONTOLOGY x YIELD  --  flat vs ontology CatBoost   [data source: {ds.source}]")
    print(f"  wafers={len(ds.X)}  features={ds.X.shape[1]}  categorical={len(ds.cat_features)}")
    print(f"  roles: " + ", ".join(f"{k}={v}" for k, v in
          pd.Series(ds.roles).value_counts().items()))

    ctx = block_performance(ds, iterations, seed)
    block_attribution(ds, ctx, iterations, seed)
    block_chain(ds)
    block_learning_curve(ds, iterations, seed)
    print()
    print(SEP)
    print("Read the three robust wins in docs/model_comparison_findings.md before presenting.")
    print(SEP)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", default="input")
    ap.add_argument("--bad-quantile", type=float, default=0.80)
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-wafers", type=int, default=250, help="virtual-data size when input is absent")
    args = ap.parse_args()
    run(args.input_dir, args.bad_quantile, args.iterations, args.seed, args.n_wafers)


if __name__ == "__main__":
    main()
