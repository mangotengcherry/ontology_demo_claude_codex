"""Plotly figures for the Streamlit app (interactive). README uses matplotlib separately."""
from __future__ import annotations

from typing import List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Consistent colors per causal role.
ROLE_COLORS = {
    "root_cause_candidate": "#2E7D32",      # green  — controllable origin
    "mediator_candidate": "#F9A825",        # amber  — symptom / process output
    "proxy_indicator": "#1565C0",           # blue   — early warning
    "confounder": "#6A1B9A",                # purple — grouping/background
    "leakage_or_post_outcome": "#C62828",   # red    — excluded
    "unknown": "#9E9E9E",
}


def bad_wafer_defect_bar(target_df: pd.DataFrame, top_k: int = 20) -> go.Figure:
    df = target_df.sort_values("defect_rate", ascending=False).head(top_k)
    fig = px.bar(
        df, x="wafer_id", y="defect_rate", color="bad_flag",
        color_continuous_scale=["#90CAF9", "#C62828"],
        labels={"wafer_id": "Wafer", "defect_rate": "불량률(defect rate, %)", "bad_flag": "Bad"},
        title=f"불량률 상위 {top_k} wafer (defect rate)",
    )
    fig.update_layout(xaxis_tickangle=-45, coloraxis_showscale=False)
    return fig


def top_shap_bar(top_shap_df: pd.DataFrame, wafer_id: str) -> go.Figure:
    """Top-SHAP horizontal bar for one wafer, colored by causal_role; leakage marked."""
    df = top_shap_df.copy().sort_values("abs_shap_value")
    df["label"] = df.apply(
        lambda r: f"{r['feature_id']}  ⛔" if r.get("is_leakage", False) else r["feature_id"], axis=1
    )
    fig = go.Figure()
    for role, sub in df.groupby("causal_role"):
        fig.add_bar(
            y=sub["label"], x=sub["shap_value"], orientation="h", name=role,
            marker_color=ROLE_COLORS.get(role, "#9E9E9E"),
            hovertemplate="%{y}<br>SHAP=%{x:.3f}<extra>" + role + "</extra>",
        )
    fig.update_layout(
        barmode="overlay", title=f"Raw SHAP — wafer {wafer_id} (⛔ = leakage, 인과 해석 제외)",
        xaxis_title="SHAP value (→ 불량률 증가)", yaxis_title="feature", legend_title="causal_role",
    )
    return fig


def shap_aggregation_bar(agg_df: pd.DataFrame, group_col: str) -> go.Figure:
    color = ROLE_COLORS if group_col == "causal_role" else None
    fig = px.bar(
        agg_df, x="mean_abs_shap", y=group_col, orientation="h",
        color=group_col if group_col == "causal_role" else None,
        color_discrete_map=color,
        labels={"mean_abs_shap": "평균 |SHAP|", group_col: group_col},
        title=f"Ontology 집계 — {group_col} 별 평균 |SHAP|",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
    return fig


def comparison_panel_figure(summary: dict) -> go.Figure:
    """Side-by-side: naive top (mediator), ontology root, ground truth — with ✓/✗."""
    match = summary["match"]
    badge = "✓ 일치(match)" if match else "✗ 불일치"
    cols = ["① naive SHAP top<br>(증상/symptom)", "② ontology 추적 root<br>(원인/cause)",
            "③ ground truth<br>(정답)"]
    vals = [summary["naive_top_nonleak_ko"], summary["ontology_root_ko"], summary["ground_truth_root_ko"]]
    colors = [ROLE_COLORS["mediator_candidate"], ROLE_COLORS["root_cause_candidate"], "#37474F"]
    fig = go.Figure()
    for i, (c, v, col) in enumerate(zip(cols, vals, colors)):
        fig.add_trace(go.Bar(x=[c], y=[1], marker_color=col, text=[v], textposition="inside",
                             insidetextanchor="middle", hoverinfo="text", hovertext=[v], showlegend=False))
    fig.update_layout(
        title=f"naive SHAP는 증상을, ontology는 원인을 지목 — {badge}",
        yaxis={"visible": False, "range": [0, 1.2]}, height=320,
    )
    return fig


def evidence_path_figure(evidence_path_text: str) -> go.Figure:
    """Simple left-to-right stepwise layout of a traced chain 'A → B → 불량률 상승'."""
    steps = [s.strip() for s in evidence_path_text.split("→")]
    n = len(steps)
    fig = go.Figure()
    for i, s in enumerate(steps):
        x = i / max(n - 1, 1)
        color = "#2E7D32" if i == 0 else ("#37474F" if i == n - 1 else "#F9A825")
        fig.add_trace(go.Scatter(
            x=[x], y=[0], mode="markers+text", marker=dict(size=46, color=color),
            text=[s], textposition="bottom center", showlegend=False,
            hovertext=[s], hoverinfo="text",
        ))
        if i < n - 1:
            fig.add_annotation(x=(i + 0.5) / max(n - 1, 1), y=0, ax=x, ay=0,
                               xref="x", yref="y", axref="x", ayref="y",
                               showarrow=True, arrowhead=3, arrowsize=1.5, arrowwidth=2, arrowcolor="#555")
    fig.update_layout(
        title="인과 증거 경로(evidence path)", height=240,
        xaxis={"visible": False, "range": [-0.15, 1.15]},
        yaxis={"visible": False, "range": [-0.6, 0.6]},
    )
    return fig
