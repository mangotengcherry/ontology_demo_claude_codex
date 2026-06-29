"""Matplotlib visualizations for the performance arm (scripts/model_comparison_demo).

Consumes a `Results` object (compute once, plot many). Each `plot_*` returns a
matplotlib Figure so it renders inline in a notebook; `save_all` writes every
chart to PNG for headless / report use. No seaborn -- matplotlib only.

The colour key is the *causal role*, so every chart tells the same story: red =
leakage (drop it), orange = mediator / symptom, green = controllable root (the
knob), grey = confounder.
"""
from __future__ import annotations

import os
from typing import List

import numpy as np
import matplotlib.pyplot as plt

ROLE_COLORS = {
    "leakage_or_post_outcome": "#d62728",   # red   -- drop it
    "mediator_candidate": "#ff7f0e",        # orange-- symptom
    "root_cause_candidate": "#2ca02c",      # green -- controllable root
    "confounder": "#9467bd",                # purple-- stratify
    "unknown": "#7f7f7f",                   # grey
}
ROLE_LABELS = {
    "leakage_or_post_outcome": "leakage / post-outcome",
    "mediator_candidate": "mediator (symptom)",
    "root_cause_candidate": "root (controllable)",
    "confounder": "confounder",
    "unknown": "unknown",
}


def _color(role: str) -> str:
    return ROLE_COLORS.get(role, ROLE_COLORS["unknown"])


def _short(feature: str, n: int = 30) -> str:
    s = str(feature)
    return s if len(s) <= n else "…" + s[-(n - 1):]


def _role_legend(fig, roles) -> None:
    handles = [plt.Rectangle((0, 0), 1, 1, color=_color(r)) for r in roles]
    fig.legend(handles, [ROLE_LABELS.get(r, r) for r in roles],
               loc="lower center", ncol=min(len(roles), 5), frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))


# --------------------------------------------------------------------------------------
def plot_performance(res) -> plt.Figure:
    """Grouped train/test R^2 per arm + the honest-ceiling line (leakage ablation)."""
    perf = res.performance
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = np.arange(len(perf))
    w = 0.36
    ax.bar(x - w / 2, perf["train_r2"], w, label="train R²", color="#bdbdbd")
    ax.bar(x + w / 2, perf["test_r2"], w, label="test R²", color="#1f77b4")
    for xi, (_, r) in zip(x, perf.iterrows()):
        ax.text(xi + w / 2, r["test_r2"] + 0.01, f"{r['test_r2']:.3f}", ha="center", fontsize=8)
    ceiling = perf.iloc[1]["test_r2"]
    ax.axhline(ceiling, ls="--", color="#2ca02c", lw=1.2)
    ax.text(-0.45, ceiling - 0.05, f"honest ceiling ≈ {ceiling:.3f}",
            ha="left", color="#2ca02c", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([a.replace(" (", "\n(") for a in perf["arm"]], fontsize=8)
    ax.set_ylabel("R²")
    ax.set_ylim(0, 1.05)
    ax.set_title("1) Performance — flat test R² is leakage-inflated;\nontology hits the honest ceiling automatically")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    return fig


def plot_attribution(res, top: int = 8) -> plt.Figure:
    """Flat vs ontology top features by mean|SHAP|, coloured by causal role."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=False)
    for ax, feat, title in [
        (axes[0], res.flat_feat.head(top), "FLAT model — credit on the symptom / leakage"),
        (axes[1], res.onto_feat.head(top), "ONTOLOGY model — credit on root / mediator"),
    ]:
        feat = feat.iloc[::-1]
        ax.barh(range(len(feat)), feat["pct"], color=[_color(r) for r in feat["role"]])
        ax.set_yticks(range(len(feat)))
        ax.set_yticklabels([_short(f) for f in feat["feature"]], fontsize=7)
        ax.set_xlabel("mean|SHAP| share (%)")
        ax.set_title(title, fontsize=10)
    roles = list(dict.fromkeys(list(res.flat_feat["role"]) + list(res.onto_feat["role"])))
    _role_legend(fig, roles)
    fig.suptitle("2) Attribution", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    return fig


def plot_ontology_rollup(res) -> plt.Figure:
    """Ontology-model SHAP rolled up by causal role."""
    roll = res.onto_roll
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(range(len(roll)), roll["pct"], color=[_color(r) for r in roll["role"]])
    ax.set_xticks(range(len(roll)))
    ax.set_xticklabels([ROLE_LABELS.get(r, r) for r in roll["role"]], fontsize=8, rotation=15)
    ax.set_ylabel("attribution share (%)")
    for i, v in enumerate(roll["pct"]):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=8)
    controllable = roll[roll["role"] == "root_cause_candidate"]["pct"].sum()
    ax.set_title(f"3) Ontology-level SHAP roll-up — {controllable:.0f}% on controllable roots (a knob)")
    fig.tight_layout()
    return fig


def plot_learning_curve(res) -> plt.Figure:
    """Test R² vs train N, flat (depth 8) vs ontology (depth 4), both leakage-free."""
    lc = res.learning_curve
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(lc["n"], lc["flat_r2"], "o-", color="#1f77b4", label="flat (depth 8)")
    ax.plot(lc["n"], lc["onto_r2"], "s-", color="#2ca02c", label="ontology (depth 4)")
    ax.fill_between(lc["n"], lc["flat_r2"], lc["onto_r2"],
                    where=(lc["onto_r2"] >= lc["flat_r2"]), color="#2ca02c", alpha=0.12,
                    label="ontology advantage")
    for _, r in lc.iterrows():
        if r["delta"] > 0:
            ax.text(r["n"], r["onto_r2"] + 0.012, f"+{r['delta']:.3f}", ha="center",
                    color="#2ca02c", fontsize=7)
    ax.set_xlabel("training wafers (N)")
    ax.set_ylabel("held-out test R²")
    ax.set_title("4) Learning curve — shallow semantic prior wins at small N;\ngap closes at large N (no large-N claim)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_causal_chain(res, top: int = 3) -> plt.Figure:
    """Node-arrow diagram of the measured root -> mediator -> target chains."""
    med = res.mediation
    fig, ax = plt.subplots(figsize=(11, 1.6 + 1.5 * max(1, min(top, 0 if med is None else len(med)))))
    ax.axis("off")
    if med is None or med.empty:
        ax.text(0.5, 0.5, "No measured chain.\nSee diagnosis checklist (naming / N<min / collinearity).",
                ha="center", va="center", fontsize=11)
        return fig
    chains = med.sort_values("indirect", key=lambda s: s.abs(), ascending=False).head(top)
    n = len(chains)
    for i, (_, r) in enumerate(chains.iterrows()):
        y = n - 1 - i
        box = dict(boxstyle="round,pad=0.4", lw=1.2)
        ax.text(0.04, y, _short(r["root"], 26), ha="left", va="center", fontsize=8,
                bbox={**box, "fc": "#d7f0d7", "ec": _color("root_cause_candidate")})
        ax.text(0.50, y, _short(r["mediator"], 22), ha="center", va="center", fontsize=8,
                bbox={**box, "fc": "#ffe6cc", "ec": _color("mediator_candidate")})
        ax.text(0.95, y, "defect_rate", ha="right", va="center", fontsize=8,
                bbox={**box, "fc": "#eeeeee", "ec": "#555555"})
        ax.annotate("", xy=(0.40, y), xytext=(0.20, y),
                    arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#444444"))
        ax.annotate("", xy=(0.84, y), xytext=(0.60, y),
                    arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#444444"))
        ax.text(0.30, y + 0.18, f"a={r['a']:.2f}", ha="center", fontsize=7)
        ax.text(0.72, y + 0.18, f"b={r['b']:.2f}", ha="center", fontsize=7)
        prop = r["prop_mediated"]
        prop_s = f"{prop*100:.0f}%" if np.isfinite(prop) else "n/a"
        flags = []
        if not r.get("strata_stable", True):
            flags.append("Simpson?")
        if not r.get("bh_reject", True):
            flags.append("not FDR-sig")
        if r.get("nonlinear_b", False):
            flags.append("nonlinear b")
        ftxt = ("  [" + ", ".join(flags) + "]") if flags else ""
        ax.text(0.50, y - 0.26,
                f"indirect a·b={r['indirect']:.2f}  (~{prop_s} mediated, q={r.get('indirect_q', float('nan')):.2g}, n={int(r['n'])}){ftxt}",
                ha="center", fontsize=7, color="#333333")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, n - 0.2)
    ax.set_title("5) Measured causal chain — controllable root drives the symptom drives yield")
    fig.tight_layout()
    return fig


def plot_credit_absorption(res) -> plt.Figure:
    """Per chain: measured root indirect vs model SHAP shares (why ranking misses cause)."""
    ca = res.credit
    fig, ax = plt.subplots(figsize=(9, 4.6))
    if ca is None or ca.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "No measured chains to diagnose.", ha="center", va="center")
        return fig
    labels = [_short(r, 22) for r in ca["root"]]
    x = np.arange(len(ca))
    w = 0.26
    ax.bar(x - w, ca["indirect"].abs(), w, label="measured |indirect| (data)", color="#2ca02c")
    ax.bar(x, ca["root_shap_share"], w, label="root model SHAP share", color="#7fbf7f")
    ax.bar(x + w, ca["mediator_shap_share"], w, label="mediator model SHAP share", color="#ff7f0e")
    for xi, (_, r) in zip(x, ca.iterrows()):
        if r["credit_absorbed"]:
            ax.text(xi, max(r["mediator_shap_share"], abs(r["indirect"])) + 0.02, "credit\nabsorbed",
                    ha="center", fontsize=7, color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=10)
    ax.set_ylabel("effect size / SHAP share")
    ax.set_title("Credit absorption — root drives the chain in the DATA,\nbut the model parks SHAP on the mediator")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------
def all_figures(res) -> dict:
    """Return every figure keyed by short name (for notebook display)."""
    return {
        "performance": plot_performance(res),
        "attribution": plot_attribution(res),
        "ontology_rollup": plot_ontology_rollup(res),
        "learning_curve": plot_learning_curve(res),
        "causal_chain": plot_causal_chain(res),
        "credit_absorption": plot_credit_absorption(res),
    }


def save_all(res, outdir: str) -> List[str]:
    """Render and save all charts to PNG; returns the list of written paths."""
    os.makedirs(outdir, exist_ok=True)
    paths: List[str] = []
    for name, fig in all_figures(res).items():
        path = os.path.join(outdir, f"{name}.png")
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths
