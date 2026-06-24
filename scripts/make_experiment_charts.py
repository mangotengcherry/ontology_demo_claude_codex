"""Generate the charts used in docs/virtual_experiment_report.md.

Runs the virtual-data experiment deterministically (the same path as
`run_demo.py --mode virtual`), then renders five matplotlib figures that document
the process and result:

  1. cohort_shap_ranking.png  - why naive SHAP misleads (leakage #1, mediator #2)
  2. a_path_scatter.png       - measured root -> mediator association (a-path)
  3. b_path_scatter.png       - measured mediator -> target association (b-path)
  4. mediation_decomposition.png - total vs direct vs indirect effect, % mediated
  5. stratified_indirect.png  - indirect effect per chamber vs pooled (Simpson guard)
  6. causal_chain.png         - the measured root -> mediator -> defect chain

Usage:  python scripts/make_experiment_charts.py
Requires matplotlib (dev-only; not in requirements.txt).
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.causal_evidence import mediation
from src.real_dataset_adapter import build_standard_dataset
from src.virtual_data_generator import (
    CHAMBER_FEATURE,
    MEDIATOR_FEATURE,
    ROOT_FEATURE,
    generate_virtual_input_dataset,
)

ASSET_DIR = os.path.join("docs", "experiment_assets")
TARGET = "defect_rate"
ROLE_COLOR = {
    "leakage_or_post_outcome": "#C62828",
    "mediator_candidate": "#F9A825",
    "root_cause_candidate": "#2E7D32",
    "confounder": "#90A4AE",
}


def _build() -> dict:
    """Deterministically rebuild the virtual experiment's standard artifacts."""
    tmp = tempfile.mkdtemp(prefix="ont_charts_")
    input_dir = os.path.join(tmp, "input")
    data_dir = os.path.join(tmp, "data")
    generate_virtual_input_dataset(input_dir, n_wafers=250, seed=42)
    build_standard_dataset(input_dir, data_dir, bad_quantile=0.80)
    shap = pd.read_csv(os.path.join(data_dir, "shap_values.csv"))
    fdict = pd.read_csv(os.path.join(data_dir, "feature_dictionary.csv"))
    shap = shap.merge(fdict[["feature_id", "causal_role"]], on="feature_id", how="left")
    return {
        "feature_matrix": pd.read_csv(os.path.join(data_dir, "feature_matrix.csv")),
        "target": pd.read_csv(os.path.join(data_dir, "target.csv")),
        "shap": shap,
        "mediation": pd.read_csv(os.path.join(data_dir, "mediation.csv")),
    }


def _frame(d: dict) -> pd.DataFrame:
    return d["feature_matrix"].merge(
        d["target"][["root_lot_id", "wafer_id", TARGET, "bad_flag"]],
        on=["root_lot_id", "wafer_id"],
    )


def _short(feature_id: str) -> str:
    parts = str(feature_id).split("|")
    return parts[-2] if len(parts) >= 2 else feature_id


def _scatter_fit(ax, x, y, color):
    ax.scatter(x, y, s=14, alpha=0.55, color=color, edgecolor="none")
    b, a = np.polyfit(x, y, 1)
    xs = np.array([x.min(), x.max()])
    ax.plot(xs, b * xs + a, color="#222", lw=2)
    return float(np.corrcoef(x, y)[0, 1])


def chart_shap_ranking(d: dict) -> None:
    s = d["shap"].sort_values("abs_shap_value")
    colors = [ROLE_COLOR.get(r, "#90A4AE") for r in s["causal_role"]]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh([_short(f) for f in s["feature_id"]], s["abs_shap_value"], color=colors)
    ax.set_xlabel("mean |SHAP| (bad wafer cohort)")
    ax.set_title("Naive SHAP ranking — top is leakage, #2 is a metro mediator (not the root)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in
               ["#C62828", "#F9A825", "#2E7D32", "#90A4AE"]]
    ax.legend(handles, ["leakage (excluded)", "mediator (symptom)", "root (cause)", "confounder"],
              fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(ASSET_DIR, "cohort_shap_ranking.png"), dpi=130)
    plt.close(fig)


def chart_a_path(frame: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    r = _scatter_fit(ax, frame[ROOT_FEATURE], frame[MEDIATOR_FEATURE], "#2E7D32")
    ax.set_xlabel(f"root: {_short(ROOT_FEATURE)} (ERD/FDC)")
    ax.set_ylabel(f"mediator: {_short(MEDIATOR_FEATURE)} (metro)")
    ax.set_title(f"a-path: root → mediator  (Pearson r = {r:.2f})")
    fig.tight_layout()
    fig.savefig(os.path.join(ASSET_DIR, "a_path_scatter.png"), dpi=130)
    plt.close(fig)


def chart_b_path(frame: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    r = _scatter_fit(ax, frame[MEDIATOR_FEATURE], frame[TARGET], "#F9A825")
    ax.set_xlabel(f"mediator: {_short(MEDIATOR_FEATURE)} (metro)")
    ax.set_ylabel("target: defect_rate")
    ax.set_title(f"b-path: mediator → defect_rate  (Pearson r = {r:.2f})")
    fig.tight_layout()
    fig.savefig(os.path.join(ASSET_DIR, "b_path_scatter.png"), dpi=130)
    plt.close(fig)


def _row(d: dict) -> pd.Series:
    m = d["mediation"]
    return m[(m["root"] == ROOT_FEATURE) & (m["mediator"] == MEDIATOR_FEATURE)].iloc[0]


def chart_mediation(d: dict) -> None:
    r = _row(d)
    labels = ["total\n(c)", "direct\n(c′)", "indirect\n(a·b)"]
    vals = [r["c_total"], r["c_direct"], r["indirect"]]
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    bars = ax.bar(labels, vals, color=["#1565C0", "#90A4AE", "#2E7D32"])
    ax.set_ylabel("standardized effect on defect_rate")
    ax.set_title(f"Mediation decomposition — {r['prop_mediated']*100:.0f}% mediated (p={r['indirect_p']:.3f})")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}",
                ha="center", va="bottom", fontsize=10)
    ax.axhline(0, color="#222", lw=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSET_DIR, "mediation_decomposition.png"), dpi=130)
    plt.close(fig)


def chart_stratified(d: dict, frame: pd.DataFrame) -> None:
    pooled = _row(d)["indirect"]
    rows = [("pooled", pooled)]
    for ch, sub in frame.groupby(CHAMBER_FEATURE):
        m = mediation(sub, ROOT_FEATURE, MEDIATOR_FEATURE, min_n=20)
        if m is not None:
            rows.append((ch, m["indirect"]))
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = ["#1565C0"] + ["#2E7D32"] * (len(rows) - 1)
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.bar(labels, vals, color=colors)
    ax.set_ylabel("indirect effect a·b")
    ax.set_title("Stratification guard — effect stable across chambers")
    ax.axhline(0, color="#222", lw=0.8)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSET_DIR, "stratified_indirect.png"), dpi=130)
    plt.close(fig)


def chart_chain(d: dict) -> None:
    r = _row(d)
    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.axis("off")
    boxes = [
        (0.13, f"{_short(ROOT_FEATURE)}\n(ERD/FDC root)", "#2E7D32"),
        (0.5, f"{_short(MEDIATOR_FEATURE)}\n(metro mediator)", "#F9A825"),
        (0.87, "defect_rate\n(yield loss)", "#C62828"),
    ]
    for x, text, color in boxes:
        ax.text(x, 0.5, text, ha="center", va="center", fontsize=11, color="white",
                bbox=dict(boxstyle="round,pad=0.5", fc=color, ec="none"))
    ax.annotate("", xy=(0.36, 0.5), xytext=(0.26, 0.5),
                arrowprops=dict(arrowstyle="-|>", lw=2, color="#222"))
    ax.annotate("", xy=(0.74, 0.5), xytext=(0.63, 0.5),
                arrowprops=dict(arrowstyle="-|>", lw=2, color="#222"))
    ax.text(0.31, 0.62, f"a (r={r['a']:.2f})", ha="center", fontsize=9)
    ax.text(0.685, 0.62, f"b={r['b']:.2f}", ha="center", fontsize=9)
    ax.text(0.5, 0.12, f"indirect a·b = {r['indirect']:.2f}   |   "
                       f"{r['prop_mediated']*100:.0f}% mediated   |   p = {r['indirect_p']:.3f}",
            ha="center", fontsize=10, color="#222")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSET_DIR, "causal_chain.png"), dpi=130)
    plt.close(fig)


def main() -> None:
    os.makedirs(ASSET_DIR, exist_ok=True)
    d = _build()
    frame = _frame(d)
    chart_shap_ranking(d)
    chart_a_path(frame)
    chart_b_path(frame)
    chart_mediation(d)
    chart_stratified(d, frame)
    chart_chain(d)
    print(f"charts written to {ASSET_DIR}/")
    print(_row(d)[["a", "b", "c_total", "c_direct", "indirect", "prop_mediated", "indirect_p"]].to_string())


if __name__ == "__main__":
    main()
