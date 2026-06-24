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

MEDIATION_COLUMNS = [
    "root", "mediator", "target", "n",
    "a", "b", "c_total", "c_direct", "indirect", "prop_mediated",
    "a_p", "indirect_p", "sign_consistent", "unstable",
    "seeded", "strata_stable", "strata_tested",
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
    return {
        "root": root, "mediator": mediator, "target": target, "n": n,
        "a": a, "b": b, "c_total": c_total, "c_direct": c_direct,
        "indirect": indirect, "prop_mediated": prop,
        "a_p": _two_sided_p(_slope_t(a, n)), "indirect_p": indirect_p,
        "sign_consistent": sign_consistent, "unstable": bool(unstable),
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
    return pd.DataFrame(rows)[MEDIATION_COLUMNS]
