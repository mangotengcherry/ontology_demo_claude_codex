"""Markdown report writer for ontology-based SHAP analysis outputs."""
from __future__ import annotations

import os
from typing import Dict

import pandas as pd


def write_markdown_report(
    data: Dict[str, pd.DataFrame],
    mapped_shap: pd.DataFrame,
    cards: pd.DataFrame,
    ontology_summary: pd.DataFrame,
    path: str,
    top_n: int = 20,
) -> None:
    """Write a compact, source-backed Markdown report for the latest run."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    target = data["target"]
    top_shap = (
        mapped_shap.sort_values("abs_shap_value", ascending=False)
        .head(top_n)
        .loc[:, _available_cols(mapped_shap, [
            "feature_id",
            "display_name_ko",
            "shap_value",
            "abs_shap_value",
            "causal_role",
            "source_type",
            "process_step",
            "mechanism_group",
            "bad_recurrence",
            "bad_good_separation_simple",
        ])]
    )

    lines = [
        "# Ontology SHAP Analysis Report",
        "",
        "## Dataset Summary",
        "",
        f"- Wafers: {len(target)}",
        f"- Bad wafers: {int(target['bad_flag'].sum())}",
        f"- Features with SHAP: {mapped_shap['feature_id'].nunique()}",
        f"- SHAP scope: {_shap_scope(mapped_shap)}",
        "",
        "## Bad wafer cohort SHAP",
        "",
        _to_markdown(top_shap),
        "",
        "## Ontology-level Summary",
        "",
        _to_markdown(ontology_summary.head(top_n)),
        "",
        "## Causal Hypothesis Cards",
        "",
    ]
    if cards.empty:
        lines.append("No causal hypothesis cards were generated.")
    else:
        for _, card in cards.iterrows():
            card_lines = [
                f"### {card['hypothesis_id']} - Grade {card['hypothesis_grade']}",
                "",
                f"- Evidence path: {card['evidence_path_text']}",
                f"- Evidence basis: {card.get('evidence_basis', '') or '-'}",
            ]
            if str(card.get("evidence_basis", "")) == "measured":
                card_lines.append(
                    f"- Measured mediation: a·b={card.get('measured_indirect_effect', '')}"
                    f", % mediated={card.get('measured_prop_mediated', '')}"
                    f", p={card.get('measured_indirect_p', '')}"
                    f", strata_stable={card.get('strata_stable', '')}"
                )
            card_lines.extend(
                [
                    f"- Root candidates: {card['root_cause_candidate_features'] or '-'}",
                    f"- Mediators: {card['mediator_candidate_features'] or '-'}",
                    f"- SHAP strength: {card['shap_strength']}",
                    f"- Bad recurrence: {card['bad_recurrence']}",
                    f"- Bad-good separation: {card['bad_good_separation_simple']}",
                    f"- Chamber: {card['primary_chamber_if_available'] or '-'}",
                    f"- Interpretation: {card['interpretation_text']}",
                    f"- Recommended validation: {card['recommended_validation']}",
                    "",
                ]
            )
            lines.extend(card_lines)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _available_cols(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [c for c in columns if c in df.columns]


def _shap_scope(mapped_shap: pd.DataFrame) -> str:
    if "shap_scope" not in mapped_shap.columns:
        return "wafer_level"
    scopes = sorted(set(mapped_shap["shap_scope"].dropna().astype(str)))
    return ", ".join(scopes) if scopes else "unknown"


def _to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    clean = df.fillna("").astype(str)
    headers = list(clean.columns)
    rows = clean.values.tolist()
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        escaped = [cell.replace("|", "\\|").replace("\n", " ") for cell in row]
        out.append("| " + " | ".join(escaped) + " |")
    return "\n".join(out)
