"""Measure root -> mediator -> target causal evidence from per-wafer raw values.

This is the missing *measurement layer*. Instead of asserting a root->mediator
edge from feature naming, we MEASURE the association (standardized OLS slope ==
Pearson r) and run a product-of-coefficients mediation (indirect = a*b, with a
proportion-mediated and a Sobel significance test) directly on the per-wafer
`raw_data` feature values + `target`.

numpy/pandas only: closed-form OLS via the 2x2 correlation matrix, and
normal-approximation p-values through `math.erf`. No scipy / statsmodels, so the
dependency list stays frozen. Small-N, near-collinearity, and missing values are
guarded explicitly.
"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import pandas as pd

TARGET_COL = "defect_rate"
DEFAULT_MIN_N = 30          # below this we refuse to assert a measured chain
DISCOVERY_R = 0.30         # |pearson| to discover a non-name-matched root->mediator edge
COLLINEARITY_D = 0.05      # 1 - r_RM^2 floor; below this the mediation is unstable
MIN_STRATUM_N = 20
NONLINEAR_GAIN = 0.05      # extra partial-R^2 from a quadratic b-path to flag non-linearity
FDR_ALPHA = 0.05           # Benjamini-Hochberg level for the (root x mediator) sweep

MEDIATION_COLUMNS = [
    "root", "mediator", "target", "n",
    "a", "b", "c_total", "c_direct", "indirect", "prop_mediated",
    "a_p", "indirect_p", "sign_consistent", "unstable",
    "seeded", "strata_stable", "strata_tested",
    "nl_gain", "nl_indirect_mag", "nonlinear_b",
    "indirect_q", "bh_reject",
]


# --------------------------------------------------------------------------------------
# Small numeric helpers (no scipy)
# --------------------------------------------------------------------------------------
def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _two_sided_p(stat: float) -> float:
    if not np.isfinite(stat):
        return float("nan")
    return float(2.0 * (1.0 - _phi(abs(stat))))


def _numeric(frame: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Coerce the requested columns to numeric and drop incomplete rows (pairwise)."""
    out = frame[cols].apply(pd.to_numeric, errors="coerce")
    return out.dropna()


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: pd.Series, y: pd.Series) -> float:
    return _corr(x.rank().to_numpy(), y.rank().to_numpy())


def _slope_t(r: float, n: int) -> float:
    if abs(r) >= 1.0 or n <= 2:
        return float("inf") if r != 0 else 0.0
    return r * math.sqrt((n - 2) / (1 - r * r))


def bh_fdr(pvals: np.ndarray, alpha: float = FDR_ALPHA) -> tuple:
    """Benjamini-Hochberg FDR. Returns (q_values, reject) aligned to input order.

    NaN p-values pass through as NaN q (not counted in the m of BH). numpy only.
    """
    p = np.asarray(pvals, dtype=float)
    q = np.full(p.shape, np.nan)
    finite = np.where(np.isfinite(p))[0]
    m = len(finite)
    if m == 0:
        return q, np.zeros(p.shape, dtype=bool)
    order = finite[np.argsort(p[finite])]
    ranked = p[order]
    raw = ranked * m / np.arange(1, m + 1)
    monotone = np.minimum.accumulate(raw[::-1])[::-1]   # enforce non-decreasing q
    q[order] = np.clip(monotone, 0.0, 1.0)
    reject = np.zeros(p.shape, dtype=bool)
    reject[order] = q[order] <= alpha
    return q, reject


def _ols_r2(Y: np.ndarray, cols: List[np.ndarray]) -> float:
    """R^2 of an OLS of Y on an intercept + the given columns (numpy lstsq)."""
    X = np.column_stack([np.ones_like(Y)] + cols)
    beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((Y - Y.mean()) ** 2).sum())
    if ss_tot <= 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _nonlinear_b_paths(R: np.ndarray, M: np.ndarray, Y: np.ndarray) -> tuple:
    """Quantify a non-linear (quadratic) mediator->target b-path.

    base = Y ~ R ; linear = Y ~ R + M ; quad = Y ~ R + M + M^2.
    Returns (nl_gain, mediator_partial_total) where:
      nl_gain                = quad - linear  (how much the linear b under-shoots)
      mediator_partial_total = quad - base    (total mediator path R^2, incl. curve)
    A large nl_gain means a U-shaped / non-monotone relation the standard linear
    mediation would miss (indirect under-estimated). numpy only.
    """
    if len(Y) < 6 or M.std() < 1e-12:
        return 0.0, 0.0
    Mc = M - M.mean()
    base = _ols_r2(Y, [R])
    linear = _ols_r2(Y, [R, Mc])
    quad = _ols_r2(Y, [R, Mc, Mc * Mc])
    return max(quad - linear, 0.0), max(quad - base, 0.0)


# --------------------------------------------------------------------------------------
# Association + mediation
# --------------------------------------------------------------------------------------
def association(frame: pd.DataFrame, xcol: str, ycol: str) -> Optional[dict]:
    """Standardized simple-OLS slope (== Pearson r) of y on x, with a normal-approx p."""
    data = _numeric(frame, [xcol, ycol])
    n = len(data)
    if n < 3:
        return None
    r = _corr(data[xcol].to_numpy(), data[ycol].to_numpy())
    return {
        "x": xcol, "y": ycol, "n": n,
        "pearson_r": r,
        "spearman_r": _spearman(data[xcol], data[ycol]),
        "slope_std": r,                       # standardized simple-OLS slope == r
        "p_value": _two_sided_p(_slope_t(r, n)),
    }


def mediation(
    frame: pd.DataFrame,
    root: str,
    mediator: str,
    target: str = TARGET_COL,
    min_n: int = DEFAULT_MIN_N,
) -> Optional[dict]:
    """Product-of-coefficients mediation on standardized root/mediator/target.

    a       = corr(root, mediator)                       (root -> mediator)
    b, c'   = standardized OLS of target ~ mediator + root
    indirect= a*b   (exactly equals c_total - c_direct for standardized OLS)
    """
    data = _numeric(frame, [root, mediator, target])
    n = len(data)
    if n < min_n:
        return None

    R = data[root].to_numpy()
    M = data[mediator].to_numpy()
    Y = data[target].to_numpy()
    r_RM = _corr(R, M)
    r_YR = _corr(Y, R)
    r_YM = _corr(Y, M)

    d = 1.0 - r_RM * r_RM
    unstable = d < COLLINEARITY_D

    a = r_RM
    se_a = math.sqrt((1 - a * a) / (n - 2)) if abs(a) < 1 and n > 2 else float("nan")
    c_total = r_YR

    if unstable or n <= 3:
        b = c_direct = se_b = indirect = prop = indirect_p = float("nan")
    else:
        b = (r_YM - r_RM * r_YR) / d
        c_direct = (r_YR - r_RM * r_YM) / d
        r2 = b * r_YM + c_direct * r_YR
        resid = max(1.0 - r2, 0.0)
        se_b = math.sqrt(resid / ((n - 3) * d))
        indirect = a * b
        prop = indirect / c_total if abs(c_total) > 1e-6 else float("nan")
        se_ind = math.sqrt(b * b * se_a * se_a + a * a * se_b * se_b)
        indirect_p = _two_sided_p(indirect / se_ind) if se_ind > 0 else float("nan")

    sign_consistent = bool(
        np.isfinite(indirect) and np.isfinite(c_total)
        and indirect * c_total > 0
    )
    nl_gain, med_partial_total = _nonlinear_b_paths(R, M, Y)
    nl_indirect_mag = abs(a) * math.sqrt(med_partial_total)
    return {
        "root": root, "mediator": mediator, "target": target, "n": n,
        "a": a, "b": b, "c_total": c_total, "c_direct": c_direct,
        "indirect": indirect, "prop_mediated": prop,
        "a_p": _two_sided_p(_slope_t(a, n)), "indirect_p": indirect_p,
        "sign_consistent": sign_consistent, "unstable": bool(unstable),
        "nl_gain": nl_gain, "nl_indirect_mag": nl_indirect_mag,
        "nonlinear_b": bool(nl_gain > NONLINEAR_GAIN),
    }


# --------------------------------------------------------------------------------------
# Stratification guard (Simpson / confounding)
# --------------------------------------------------------------------------------------
def _strata_cols(columns, process_step: str) -> List[str]:
    candidates = [f"cat|eqp_ch|{process_step}", f"cat|ppid|{process_step}"]
    return [c for c in candidates if c in columns]


def stratified_stability(
    frame: pd.DataFrame,
    root: str,
    mediator: str,
    pooled_indirect: float,
    strata_col: str,
    target: str = TARGET_COL,
    min_stratum_n: int = MIN_STRATUM_N,
) -> Optional[bool]:
    """Does the pooled indirect effect keep its sign within strata?

    Returns None when the chain cannot be tested (no pooled effect, or fewer than
    two sufficiently-sized strata). Otherwise True when a majority of strata agree
    in sign with the pooled indirect effect, False when they flip/vanish (a
    Simpson's-paradox warning).
    """
    if not np.isfinite(pooled_indirect) or pooled_indirect == 0:
        return None
    indirects: List[float] = []
    for _, sub in frame.groupby(strata_col):
        m = mediation(sub, root, mediator, target, min_n=min_stratum_n)
        if m is not None and np.isfinite(m["indirect"]):
            indirects.append(m["indirect"])
    if len(indirects) < 2:
        return None
    same = sum(1 for v in indirects if v * pooled_indirect > 0)
    return same / len(indirects) >= 0.5


# --------------------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------------------
def measure_edges(
    feature_matrix: pd.DataFrame,
    target: pd.DataFrame,
    feature_dictionary: pd.DataFrame,
    min_n: int = DEFAULT_MIN_N,
    discovery_r: float = DISCOVERY_R,
    fdr_alpha: float = FDR_ALPHA,
) -> pd.DataFrame:
    """Measure mediation for every (root_cause, mediator) candidate pair.

    Candidates are seeded by the naming-based role + process_step prior, but a pair
    is also kept when correlation discovers it even though process_step differs.
    Returns one row per surviving pair (see MEDIATION_COLUMNS). Empty frame when
    nothing is measurable (e.g. N below `min_n`) -- callers fall back to naming.
    """
    if TARGET_COL not in target.columns:
        return pd.DataFrame(columns=MEDIATION_COLUMNS)
    frame = feature_matrix.merge(
        target[["root_lot_id", "wafer_id", TARGET_COL]],
        on=["root_lot_id", "wafer_id"], how="inner",
    )
    columns = set(frame.columns)
    role = feature_dictionary.set_index("feature_id")["causal_role"].to_dict()
    step = feature_dictionary.set_index("feature_id")["process_step"].to_dict()
    roots = [f for f, r in role.items() if r == "root_cause_candidate" and f in columns]
    mediators = [f for f, r in role.items() if r == "mediator_candidate" and f in columns]

    rows: List[dict] = []
    for med in mediators:
        med_step = str(step.get(med, "") or "")
        strata_cols = _strata_cols(columns, med_step)
        for root in roots:
            seeded = med_step not in ("", "-") and str(step.get(root, "")) == med_step
            m = mediation(frame, root, med, min_n=min_n)
            if m is None:
                continue
            # Keep seeded pairs always; keep unseeded only if correlation is strong.
            if not seeded and abs(m["a"]) < discovery_r:
                continue
            stable: Optional[bool] = None
            for sc in strata_cols:
                s = stratified_stability(frame, root, med, m["indirect"], sc)
                if s is not None:
                    stable = s if stable is None else (stable and s)
            m["seeded"] = bool(seeded)
            m["strata_stable"] = True if stable is None else bool(stable)
            m["strata_tested"] = stable is not None
            rows.append(m)

    if not rows:
        return pd.DataFrame(columns=MEDIATION_COLUMNS)

    # Benjamini-Hochberg across the whole (root x mediator) sweep so the many
    # pairwise mediation tests don't inflate false positives.
    q, reject = bh_fdr(np.array([r["indirect_p"] for r in rows]), alpha=fdr_alpha)
    for row, qv, rj in zip(rows, q, reject):
        row["indirect_q"] = float(qv)
        row["bh_reject"] = bool(rj)
    return pd.DataFrame(rows)[MEDIATION_COLUMNS]
