"""Turn mapped SHAP + the causal-edge graph into graded, validation-ready causal *hypotheses*.

This is the second (and decisive) ontology upgrade: instead of reporting the top raw-SHAP
feature, we traverse the explicit causal graph (networkx) upstream from high-SHAP mediators
to the controllable root cause, consolidate credit to the most-upstream root, and grade the
resulting hypothesis. Leakage / post-outcome features are excluded from causal reasoning
(kept visible as `excluded_leakage_features`).
"""
from __future__ import annotations

from typing import Dict, List

import networkx as nx
import numpy as np
import pandas as pd

TARGET_NODE = "defect_rate"

# Human-readable Korean phrase per mechanism group (used to build the evidence path text).
_MECHANISM_KO: Dict[str, str] = {
    "CVD_pressure_instability": "CVD 압력 불안정(pressure instability)",
    "CVD_RF_instability": "CVD RF 파워 불안정(RF instability)",
    "Etch_thermal": "Etch 열적 변동(thermal variation)",
    "edge_thickness_nonuniformity": "edge 두께 비균일(non-uniformity)",
    "CD_center": "center CD 변동",
    "queue_time": "대기시간 증가(queue time)",
    "prior_defect_signal": "이전 결함 신호(prior defect)",
}
_TARGET_KO = "불량률 상승"


def build_hypothesis_cards(
    mapped_shap_df: pd.DataFrame,
    target_df: pd.DataFrame,
    process_history_df: pd.DataFrame,
    causal_edges_df: pd.DataFrame,
    feature_dict_df: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """Build causal-hypothesis cards for the bad-wafer group of the (single) target."""
    role_map = feature_dict_df.set_index("feature_id")["causal_role"].to_dict()
    mech_map = feature_dict_df.set_index("feature_id")["mechanism_group"].to_dict()
    temporal_map = feature_dict_df.set_index("feature_id")["temporal_relation_to_target"].to_dict()
    control_map = feature_dict_df.set_index("feature_id")["controllability"].to_dict()
    name_map = feature_dict_df.set_index("feature_id")["display_name_ko"].to_dict()

    bad_wafers = target_df.loc[target_df["bad_flag"] == 1, "wafer_id"].tolist()
    n_bad = len(bad_wafers)

    # --- Aggregate evidence over the bad-wafer group.
    bad_shap = mapped_shap_df[mapped_shap_df["wafer_id"].isin(bad_wafers)]
    mean_abs = bad_shap.groupby("feature_id")["abs_shap_value"].mean()
    max_nonleak = mean_abs[[f for f in mean_abs.index
                            if role_map.get(f) != "leakage_or_post_outcome"]].max()

    anomaly = _feature_anomaly(mapped_shap_df, bad_wafers)          # standardized bad-vs-good gap
    recurrence = _feature_recurrence(mapped_shap_df, bad_wafers)    # fraction of bad wafers elevated

    graph = _build_graph(causal_edges_df)
    excluded_leakage = sorted(
        f for f in mean_abs.index if role_map.get(f) == "leakage_or_post_outcome"
    )

    # --- Trace each high-SHAP mediator upstream to a root; group mediators by root.
    mediators = [
        f for f in mean_abs.index
        if role_map.get(f) == "mediator_candidate" and mean_abs[f] > 0.05 * max_nonleak
    ]
    root_groups: Dict[str, Dict] = {}
    for med in sorted(mediators, key=lambda f: -mean_abs[f]):
        root = _trace_upstream_root(graph, med, role_map, anomaly)
        key = root if root is not None else med  # mediator with no traceable root: stands alone
        grp = root_groups.setdefault(key, {"root": root, "mediators": []})
        grp["mediators"].append(med)

    # --- Direct root causes that act on the target without a metro mediator (e.g. queue time).
    for f in mean_abs.index:
        if role_map.get(f) == "root_cause_candidate" and f not in root_groups:
            if graph.has_edge(f, TARGET_NODE) and mean_abs[f] > 0.05 * max_nonleak:
                root_groups.setdefault(f, {"root": f, "mediators": []})

    primary_chamber = _bad_chamber(process_history_df, bad_wafers)
    ppid_confounded = anomaly.get("recipe_ppid_77a_flag", 0.0) > 0.5

    cards: List[Dict] = []
    for i, (key, grp) in enumerate(
        sorted(root_groups.items(),
               key=lambda kv: -_group_strength(kv[1], mean_abs)),
        start=1,
    ):
        card = _make_card(
            idx=i, key=key, grp=grp, mean_abs=mean_abs, max_nonleak=max_nonleak,
            anomaly=anomaly, recurrence=recurrence, n_bad=n_bad,
            role_map=role_map, mech_map=mech_map, temporal_map=temporal_map,
            control_map=control_map, name_map=name_map,
            primary_chamber=primary_chamber, ppid_confounded=ppid_confounded,
            excluded_leakage=excluded_leakage,
        )
        cards.append(card)

    cards = cards[:top_n]

    # --- One explicit grade-X card to demonstrate the leakage guardrail.
    if excluded_leakage:
        cards.append(_leakage_card(len(cards) + 1, excluded_leakage, mean_abs, name_map, n_bad))

    return pd.DataFrame(cards)


# --------------------------------------------------------------------------------------
# Graph traversal
# --------------------------------------------------------------------------------------
def _build_graph(causal_edges_df: pd.DataFrame) -> nx.DiGraph:
    g = nx.DiGraph()
    for _, e in causal_edges_df.iterrows():
        g.add_edge(e["source_feature"], e["target_feature"],
                   relation=e["relation"], confidence=float(e["confidence"]))
    return g


def _trace_upstream_root(
    graph: nx.DiGraph, mediator: str, role_map: Dict, anomaly: Dict, _depth: int = 0
) -> str | None:
    """Walk upstream along `physicallyAffects` edges to the most-upstream anomalous root."""
    if mediator not in graph or _depth > 10:
        return None
    upstream = [
        u for u, _, d in graph.in_edges(mediator, data=True)
        if d.get("relation") == "physicallyAffects"
    ]
    # Prefer an anomalous root_cause_candidate; recurse to find a still-more-upstream one.
    best = None
    best_score = -np.inf
    for u in upstream:
        deeper = _trace_upstream_root(graph, u, role_map, anomaly, _depth + 1)
        candidate = deeper if deeper is not None else (
            u if role_map.get(u) == "root_cause_candidate" else None
        )
        if candidate is None:
            continue
        score = anomaly.get(candidate, 0.0)  # prefer the more anomalous (elevated) root
        if score > best_score:
            best_score, best = score, candidate
    return best


# --------------------------------------------------------------------------------------
# Evidence statistics
# --------------------------------------------------------------------------------------
def _feature_anomaly(mapped_shap_df: pd.DataFrame, bad_wafers: List[str]) -> Dict[str, float]:
    """Standardized gap between bad-wafer mean and overall mean of each feature_value."""
    out: Dict[str, float] = {}
    fv = mapped_shap_df[["wafer_id", "feature_id", "feature_value"]].drop_duplicates()
    for feat, sub in fv.groupby("feature_id"):
        vals = pd.to_numeric(sub["feature_value"], errors="coerce")
        if vals.isna().all():
            out[feat] = 0.0
            continue
        overall_mean, overall_std = vals.mean(), vals.std() + 1e-9
        bad_mean = vals[sub["wafer_id"].isin(bad_wafers)].mean()
        out[feat] = float((bad_mean - overall_mean) / overall_std)
    return out


def _feature_recurrence(mapped_shap_df: pd.DataFrame, bad_wafers: List[str]) -> Dict[str, float]:
    """Fraction of bad wafers where the feature value exceeds the overall (mean + 0.5 std)."""
    out: Dict[str, float] = {}
    fv = mapped_shap_df[["wafer_id", "feature_id", "feature_value"]].drop_duplicates()
    for feat, sub in fv.groupby("feature_id"):
        vals = pd.to_numeric(sub["feature_value"], errors="coerce")
        if vals.isna().all():
            out[feat] = 0.0
            continue
        thresh = vals.mean() + 0.5 * (vals.std() + 1e-9)
        bad_vals = vals[sub["wafer_id"].isin(bad_wafers)]
        out[feat] = float((bad_vals > thresh).mean()) if len(bad_vals) else 0.0
    return out


def _bad_chamber(process_history_df: pd.DataFrame, bad_wafers: List[str]) -> str | None:
    """Most common CVD chamber among bad wafers (chamber concentration)."""
    cvd = process_history_df[
        (process_history_df["process_step"] == "CVD")
        & (process_history_df["wafer_id"].isin(bad_wafers))
    ]
    if cvd.empty:
        return None
    return cvd["chamber_id"].value_counts().idxmax()


def _group_strength(grp: Dict, mean_abs: pd.Series) -> float:
    feats = ([grp["root"]] if grp.get("root") else []) + grp["mediators"]
    return float(sum(mean_abs.get(f, 0.0) for f in feats))


# --------------------------------------------------------------------------------------
# Card construction + grading
# --------------------------------------------------------------------------------------
def _make_card(idx, key, grp, mean_abs, max_nonleak, anomaly, recurrence, n_bad,
               role_map, mech_map, temporal_map, control_map, name_map,
               primary_chamber, ppid_confounded, excluded_leakage) -> Dict:
    root = grp.get("root")
    mediators = grp["mediators"]
    root_feats = [root] if root else []
    proxies = [f for f in mean_abs.index if role_map.get(f) == "proxy_indicator"]

    group_feats = root_feats + mediators
    strength_val = sum(mean_abs.get(f, 0.0) for f in group_feats)
    rel_strength = strength_val / (max_nonleak + 1e-9)
    shap_strength = "strong" if rel_strength >= 1.2 else "moderate" if rel_strength >= 0.6 else "weak"

    primary_mech = mech_map.get(root) if root else (mech_map.get(mediators[0]) if mediators else "unknown")
    primary_step = "CVD" if root and "CVD" in str(mech_map.get(root, "")) else (
        "QUEUE" if root == "queue_time_before_etch" else "CVD_POST_METRO")

    # Recurrence of the mechanism: use the root if present, else the strongest mediator.
    anchor = root if root else (mediators[0] if mediators else key)
    bad_recurrence = recurrence.get(anchor, 0.0)
    separation = anomaly.get(anchor, 0.0)

    temporal_ok = all(
        temporal_map.get(f, "before_target") == "before_target" for f in group_feats
    )
    temporal_validity = "valid(before_target)" if temporal_ok else "invalid"
    controllability = control_map.get(root, "not_directly_controllable") if root else "not_directly_controllable"

    # Confounding: PPID_77A elevated in bad wafers and on this CVD mechanism.
    confounding = "high" if (ppid_confounded and primary_step == "CVD") else "low"
    leakage_risk = "low"  # roots/mediators here are low; leakage features are excluded separately

    grade = _grade(shap_strength, bad_recurrence, temporal_ok, leakage_risk,
                   confounding, bool(root), bool(mediators))

    evidence_path_text = _evidence_path_ko(root, mediators, mech_map)
    chamber_txt = f" (집중 챔버: {primary_chamber})" if primary_chamber and primary_step == "CVD" else ""

    interpretation_text = _interpretation_ko(
        root, mediators, name_map, mech_map, shap_strength, bad_recurrence,
        confounding, controllability, primary_chamber if primary_step == "CVD" else None,
        ppid_confounded,
    )
    recommended = _recommended_validation_ko(root, mediators, primary_step,
                                             primary_chamber if primary_step == "CVD" else None,
                                             ppid_confounded)

    return {
        "hypothesis_id": f"HYP_{idx:03d}",
        "target_id": TARGET_NODE,
        "bad_wafer_group_size": n_bad,
        "primary_process_step": primary_step,
        "primary_mechanism_group": primary_mech,
        "primary_chamber_if_available": primary_chamber if primary_step == "CVD" else "",
        "root_cause_candidate_features": ", ".join(root_feats),
        "mediator_candidate_features": ", ".join(mediators),
        "proxy_features": ", ".join(proxies),
        "excluded_leakage_features": ", ".join(excluded_leakage),
        "shap_strength": shap_strength,
        "bad_recurrence": round(bad_recurrence, 3),
        "bad_good_separation_simple": round(separation, 3),
        "temporal_validity": temporal_validity,
        "leakage_risk": leakage_risk,
        "confounding_risk": confounding,
        "controllability": controllability,
        "hypothesis_grade": grade,
        "evidence_path_text": evidence_path_text + chamber_txt,
        "interpretation_text": interpretation_text,
        "recommended_validation": " | ".join(recommended),
    }


def _grade(shap_strength, recurrence, temporal_ok, leakage_risk,
           confounding, has_root, has_mediator) -> str:
    """Deterministic A/B/C/X grading."""
    if leakage_risk == "high":
        return "X"
    strong = shap_strength == "strong"
    recurrent = recurrence >= 0.5
    if strong and recurrent and temporal_ok and confounding == "low" and has_root:
        return "A"
    if strong and (confounding != "low" or not has_root or has_mediator):
        return "B"
    if not recurrent or shap_strength == "weak":
        return "C"
    return "B"


def _evidence_path_ko(root, mediators, mech_map) -> str:
    """Build the Korean evidence chain: root mechanism -> mediator mechanism -> 불량률 상승."""
    parts: List[str] = []
    if root:
        parts.append(_MECHANISM_KO.get(mech_map.get(root, ""), mech_map.get(root, root)))
    if mediators:
        med_mech = mech_map.get(mediators[0], "")
        parts.append(_MECHANISM_KO.get(med_mech, med_mech or mediators[0]))
    parts.append(_TARGET_KO)
    return " → ".join(parts)


def _interpretation_ko(root, mediators, name_map, mech_map, shap_strength, recurrence,
                       confounding, controllability, chamber, ppid_confounded) -> str:
    root_ko = name_map.get(root, root) if root else "상류 원인 미상"
    med_ko = name_map.get(mediators[0], mediators[0]) if mediators else "-"
    txt = (
        f"이것은 인과 가설(causal hypothesis)이며 확정 원인이 아닙니다. "
        f"naive SHAP는 증상에 해당하는 매개 신호 '{med_ko}'를 상위로 지목하지만, "
        f"ontology 인과관계 그래프를 상류로 추적하면 controllable 상류 원인 후보 '{root_ko}'가 도출됩니다. "
        f"SHAP 강도는 {shap_strength}, 불량 wafer 재현율(recurrence)은 {recurrence:.0%} 입니다."
    )
    if chamber:
        txt += f" 불량은 {chamber} 챔버에 집중되어 있습니다."
    if ppid_confounded and confounding == "high":
        txt += " 단, PPID_77A 레시피가 교란(confounding) 요인으로 함께 변동하므로 분리 검증이 필요합니다."
    return txt


def _recommended_validation_ko(root, mediators, step, chamber, ppid_confounded) -> List[str]:
    items: List[str] = []
    if chamber:
        items.append(f"{chamber} 챔버 FDC 압력/RF 트렌드 및 PM 이력 점검")
    if root == "fdc_cvd_pressure_slope_p95" or step == "CVD":
        items.append("CVD 압력 안정화 후 edge 두께 비균일 변화 A/B 비교")
    if ppid_confounded:
        items.append("PPID_77A vs 정상 레시피로 층화하여 교란(confounding) 분리")
    if mediators:
        items.append("edge 두께 매개 경로를 통한 불량률 회귀 검증")
    if not items:
        items.append("해당 feature 상류 공정 파라미터 점검")
    return items


def naive_vs_ontology_summary(
    mapped_shap_df: pd.DataFrame,
    target_df: pd.DataFrame,
    causal_edges_df: pd.DataFrame,
    feature_dict_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
) -> Dict:
    """The headline: naive top-SHAP (a mediator) vs ontology-traced root vs ground truth.

    Returns the overall raw-SHAP #1 (usually the leakage feature — the guardrail point),
    the naive #1 among non-leakage features (a mediator), the ontology-traced root cause,
    the ground-truth root, and whether the ontology root matches ground truth (✓/✗).
    """
    role_map = feature_dict_df.set_index("feature_id")["causal_role"].to_dict()
    name_map = feature_dict_df.set_index("feature_id")["display_name_ko"].to_dict()

    bad_wafers = target_df.loc[target_df["bad_flag"] == 1, "wafer_id"].tolist()
    bad_shap = mapped_shap_df[mapped_shap_df["wafer_id"].isin(bad_wafers)]
    mean_abs = bad_shap.groupby("feature_id")["abs_shap_value"].mean().sort_values(ascending=False)

    naive_top_overall = mean_abs.index[0]
    nonleak = [f for f in mean_abs.index if role_map.get(f) != "leakage_or_post_outcome"]
    naive_top_nonleak = nonleak[0] if nonleak else naive_top_overall

    anomaly = _feature_anomaly(mapped_shap_df, bad_wafers)
    graph = _build_graph(causal_edges_df)
    ontology_root = None
    if role_map.get(naive_top_nonleak) == "mediator_candidate":
        ontology_root = _trace_upstream_root(graph, naive_top_nonleak, role_map, anomaly)
    if ontology_root is None and role_map.get(naive_top_nonleak) == "root_cause_candidate":
        ontology_root = naive_top_nonleak

    gt_root = ground_truth_df.iloc[0]["true_root_feature"]
    return {
        "naive_top_overall": naive_top_overall,
        "naive_top_overall_ko": name_map.get(naive_top_overall, naive_top_overall),
        "naive_top_overall_role": role_map.get(naive_top_overall, "unknown"),
        "naive_top_nonleak": naive_top_nonleak,
        "naive_top_nonleak_ko": name_map.get(naive_top_nonleak, naive_top_nonleak),
        "naive_top_nonleak_role": role_map.get(naive_top_nonleak, "unknown"),
        "ontology_root": ontology_root,
        "ontology_root_ko": name_map.get(ontology_root, ontology_root) if ontology_root else "미상",
        "ground_truth_root": gt_root,
        "ground_truth_root_ko": name_map.get(gt_root, gt_root),
        "match": bool(ontology_root == gt_root),
        "mean_abs_by_feature": mean_abs,
        "role_map": role_map,
        "name_map": name_map,
    }


def _leakage_card(idx, excluded_leakage, mean_abs, name_map, n_bad) -> Dict:
    leak_ko = ", ".join(name_map.get(f, f) for f in excluded_leakage)
    return {
        "hypothesis_id": f"HYP_{idx:03d}",
        "target_id": TARGET_NODE,
        "bad_wafer_group_size": n_bad,
        "primary_process_step": "FINAL_TEST",
        "primary_mechanism_group": "post_outcome_proxy",
        "primary_chamber_if_available": "",
        "root_cause_candidate_features": "",
        "mediator_candidate_features": "",
        "proxy_features": "",
        "excluded_leakage_features": ", ".join(excluded_leakage),
        "shap_strength": "strong",
        "bad_recurrence": 0.0,
        "bad_good_separation_simple": 0.0,
        "temporal_validity": "invalid(after_or_derived)",
        "leakage_risk": "high",
        "confounding_risk": "n/a",
        "controllability": "not_controllable",
        "hypothesis_grade": "X",
        "evidence_path_text": "누설(leakage) feature — 인과 경로 추적 대상 아님",
        "interpretation_text": (
            f"'{leak_ko}'는 target(defect_rate)에서 파생된 누설(leakage)로 raw SHAP에서 가장 큰 값을 갖지만, "
            f"인과 해석에서 제외(guardrail)합니다. 이를 제외하지 않으면 SHAP이 원인 분석을 지배합니다."
        ),
        "recommended_validation": "해당 feature를 모델/인과 해석에서 제외 유지",
    }
