"""Render the static result PNGs embedded in the README (evaluation process & results).

Uses matplotlib (headless, no browser) so the figures commit cleanly and render on GitHub.
Run after data generation:  python scripts/make_readme_assets.py
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.loaders import load_all_data
from src.ontology_mapping import map_shap_to_feature_dictionary
from src.hypothesis_engine import naive_vs_ontology_summary

# Korean-capable font on macOS; falls back gracefully elsewhere.
KFONT = "DejaVu Sans"
for _f in ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Malgun Gothic"]:
    try:
        matplotlib.font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams["font.family"] = _f
        KFONT = _f
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

ROLE_COLORS = {
    "root_cause_candidate": "#2E7D32",
    "mediator_candidate": "#F9A825",
    "proxy_indicator": "#1565C0",
    "confounder": "#6A1B9A",
    "leakage_or_post_outcome": "#C62828",
    "unknown": "#9E9E9E",
}
ASSET_DIR = os.path.join("docs", "assets")


def _bad_wafers(target):
    return target.loc[target["bad_flag"] == 1, "wafer_id"].tolist()


def fig_comparison_panel(summary, path):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    match = summary["match"]
    cols = ["① naive SHAP top\n(증상 symptom)", "② ontology 추적 root\n(원인 cause)", "③ ground truth\n(정답)"]
    vals = [summary["naive_top_nonleak"], summary["ontology_root"], summary["ground_truth_root"]]
    colors = [ROLE_COLORS["mediator_candidate"], ROLE_COLORS["root_cause_candidate"], "#37474F"]
    ax.bar(cols, [1, 1, 1], color=colors)
    for i, v in enumerate(vals):
        ax.text(i, 0.5, v, ha="center", va="center", color="white", fontsize=10, wrap=True)
    badge = "✓ 일치 (ontology root == ground truth)" if match else "✗ 불일치"
    ax.set_title(f"naive SHAP는 증상을, ontology는 원인을 지목 — {badge}", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.2)
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_top_shap(mapped, target, summary, path):
    bad = _bad_wafers(target)
    mean_abs = (
        mapped[mapped["wafer_id"].isin(bad)]
        .groupby(["feature_id", "causal_role"])["abs_shap_value"].mean()
        .reset_index().sort_values("abs_shap_value", ascending=True).tail(8)
    )
    colors = [ROLE_COLORS.get(r, "#9E9E9E") for r in mean_abs["causal_role"]]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    bars = ax.barh(mean_abs["feature_id"], mean_abs["abs_shap_value"], color=colors)
    for b, role in zip(bars, mean_abs["causal_role"]):
        if role == "leakage_or_post_outcome":
            ax.text(b.get_width(), b.get_y() + b.get_height() / 2, "  ← 인과 해석 제외(leakage)",
                    va="center", color="#C62828", fontsize=9, fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ROLE_COLORS.values()]
    ax.legend(handles, ROLE_COLORS.keys(), fontsize=7, loc="lower right", title="causal_role")
    ax.set_xlabel("평균 |SHAP| (bad wafer group)")
    ax.set_title("불량 wafer 그룹의 raw SHAP 상위 feature\n(leakage가 1위 → 가드레일로 제외, 그 다음 mediator가 root보다 높음 = trap)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_shap_by_role(mapped, target, path):
    bad = _bad_wafers(target)
    agg = (mapped[mapped["wafer_id"].isin(bad)].groupby("causal_role")["abs_shap_value"]
           .mean().reset_index().sort_values("abs_shap_value"))
    colors = [ROLE_COLORS.get(r, "#9E9E9E") for r in agg["causal_role"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(agg["causal_role"], agg["abs_shap_value"], color=colors)
    ax.set_xlabel("평균 |SHAP|")
    ax.set_title("causal_role 별 SHAP 집계 — 역할 가드레일(role guardrail)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_causal_path(edges, summary, path):
    g = nx.DiGraph()
    chain = [summary["ground_truth_root"], summary["naive_top_nonleak"], "defect_rate"]
    labels = {
        summary["ground_truth_root"]: "CVD 압력 불안정\n(root, controllable)",
        summary["naive_top_nonleak"]: "edge 두께 비균일\n(mediator, 증상)",
        "defect_rate": "불량률\n(target)",
    }
    for a, b in zip(chain, chain[1:]):
        g.add_edge(a, b)
    pos = {chain[0]: (0, 0), chain[1]: (1, 0), chain[2]: (2, 0)}
    node_colors = [ROLE_COLORS["root_cause_candidate"], ROLE_COLORS["mediator_candidate"], "#37474F"]
    fig, ax = plt.subplots(figsize=(9.5, 2.8))
    nx.draw_networkx_nodes(g, pos, node_size=6500, node_color=node_colors, ax=ax)
    nx.draw_networkx_edges(g, pos, ax=ax, arrowsize=28, width=2.5, node_size=6500,
                           edge_color="#444", min_target_margin=42)
    nx.draw_networkx_labels(g, pos, labels=labels, font_size=9, font_color="white", ax=ax,
                            font_family=KFONT)
    edge_lbls = {(chain[0], chain[1]): "physicallyAffects", (chain[1], chain[2]): "mediates"}
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_lbls, font_size=8, ax=ax)
    ax.set_title("Ontology 인과관계 그래프 추적: root → mediator → target", fontsize=12, fontweight="bold")
    ax.margins(0.18)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_model_performance(data, path):
    if "model_predictions" not in data:
        return
    mp = data["model_predictions"]
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    ax.scatter(mp["y_true"], mp["y_pred"], s=18, alpha=0.6, color="#1565C0")
    lo, hi = mp[["y_true", "y_pred"]].min().min(), mp[["y_true", "y_pred"]].max().max()
    ax.plot([lo, hi], [lo, hi], "--", color="#C62828")
    ax.set_xlabel("실제 defect_rate (y_true)")
    ax.set_ylabel("예측 defect_rate (y_pred)")
    ax.set_title("CatBoost 회귀 적합도 (parity plot)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main(data_dir: str = "data") -> None:
    os.makedirs(ASSET_DIR, exist_ok=True)
    data = load_all_data(data_dir)
    mapped = map_shap_to_feature_dictionary(data["shap_values"], data["feature_dictionary"])
    summary = naive_vs_ontology_summary(
        mapped, data["target"], data["causal_edges"], data["feature_dictionary"], data["ground_truth"]
    )
    fig_comparison_panel(summary, os.path.join(ASSET_DIR, "comparison_panel.png"))
    fig_top_shap(mapped, data["target"], summary, os.path.join(ASSET_DIR, "shap_top_features.png"))
    fig_shap_by_role(mapped, data["target"], os.path.join(ASSET_DIR, "shap_by_role.png"))
    fig_causal_path(data["causal_edges"], summary, os.path.join(ASSET_DIR, "causal_path.png"))
    fig_model_performance(data, os.path.join(ASSET_DIR, "model_performance.png"))
    print(f"[make_readme_assets] wrote 5 PNGs to {ASSET_DIR}/  (match={summary['match']})")


if __name__ == "__main__":
    main()
