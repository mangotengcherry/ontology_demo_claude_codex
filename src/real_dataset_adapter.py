"""Convert company dataset CSVs into the project's standard ontology-SHAP artifacts."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

TARGET_ID = "defect_rate"
COHORT_ID = "__BAD_WAFER_COHORT__"
RAW_DATA_FILE = "raw_data.csv"
SHAP_FILE = "x_feature_shap_value.csv"
RELATION_FILE = "prc_metro_relation.csv"
BAD_WAFERS_FILE = "bad_wafers.csv"
ID_COLUMNS = ["root_lot_id", "wafer_id"]
RAW_NON_FEATURE_COLUMNS = {"root_lot_id", "wafer_id", "tkout_time", "target"}


@dataclass(frozen=True)
class StandardDatasetArtifacts:
    feature_matrix: pd.DataFrame
    target: pd.DataFrame
    shap_values: pd.DataFrame
    feature_dictionary: pd.DataFrame
    causal_edges: pd.DataFrame
    process_history: pd.DataFrame
    engineer_feedback: pd.DataFrame
    ground_truth: pd.DataFrame


def build_standard_dataset(
    input_dir: str = "input",
    output_dir: str = "data",
    bad_quantile: float = 0.80,
) -> StandardDatasetArtifacts:
    """Build standard project CSVs from the real dataset contract in dataset_info.md.

    `x_feature_shap_value.csv` is interpreted as bad-wafer cohort mean SHAP, not
    per-wafer SHAP. The resulting `shap_values.csv` therefore contains one row per
    feature with `shap_scope = bad_wafer_mean`.
    """
    raw = _read_required_csv(input_dir, RAW_DATA_FILE)
    shap_input = _read_required_csv(input_dir, SHAP_FILE)
    relation = _read_required_csv(input_dir, RELATION_FILE)
    _validate_raw(raw)
    _validate_shap(shap_input)
    relation = _normalize_relation(relation)

    feature_cols = [c for c in raw.columns if c not in RAW_NON_FEATURE_COLUMNS]
    feature_matrix = raw[ID_COLUMNS + feature_cols].copy()
    bad_wafers = _read_optional_bad_wafers(input_dir)
    target = _build_target(raw, bad_quantile, bad_wafers=bad_wafers)
    feature_ids = _ordered_union(feature_cols, shap_input["feature"].astype(str).tolist())
    feature_dictionary = _build_feature_dictionary(feature_ids, relation)
    shap_values = _build_feature_level_shap(shap_input, raw, target)
    causal_edges = _build_causal_edges(feature_dictionary, relation)
    process_history = _build_process_history(raw)
    engineer_feedback = _empty_engineer_feedback()
    ground_truth = _unknown_ground_truth()

    artifacts = StandardDatasetArtifacts(
        feature_matrix=feature_matrix,
        target=target,
        shap_values=shap_values,
        feature_dictionary=feature_dictionary,
        causal_edges=causal_edges,
        process_history=process_history,
        engineer_feedback=engineer_feedback,
        ground_truth=ground_truth,
    )
    write_standard_dataset(artifacts, output_dir)
    return artifacts


def write_standard_dataset(artifacts: StandardDatasetArtifacts, output_dir: str = "data") -> None:
    os.makedirs(output_dir, exist_ok=True)
    artifacts.feature_matrix.to_csv(os.path.join(output_dir, "feature_matrix.csv"), index=False)
    artifacts.target.to_csv(os.path.join(output_dir, "target.csv"), index=False)
    artifacts.shap_values.to_csv(os.path.join(output_dir, "shap_values.csv"), index=False)
    artifacts.feature_dictionary.to_csv(os.path.join(output_dir, "feature_dictionary.csv"), index=False)
    artifacts.process_history.to_csv(os.path.join(output_dir, "process_history.csv"), index=False)
    artifacts.engineer_feedback.to_csv(os.path.join(output_dir, "engineer_feedback.csv"), index=False)
    artifacts.ground_truth.to_csv(os.path.join(output_dir, "ground_truth.csv"), index=False)
    _write_causal_edges(artifacts.causal_edges, os.path.join(output_dir, "causal_edges.csv"))


def input_files_exist(input_dir: str = "input") -> bool:
    return all(os.path.exists(os.path.join(input_dir, name)) for name in [RAW_DATA_FILE, SHAP_FILE, RELATION_FILE])


def _read_required_csv(input_dir: str, filename: str) -> pd.DataFrame:
    path = os.path.join(input_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required input file: {path}")
    return pd.read_csv(path)


def _validate_raw(raw: pd.DataFrame) -> None:
    missing = [c for c in ["root_lot_id", "wafer_id", "target"] if c not in raw.columns]
    if missing:
        raise ValueError(f"{RAW_DATA_FILE} is missing required columns: {missing}")


def _validate_shap(shap_input: pd.DataFrame) -> None:
    missing = [c for c in ["feature", "shap_value"] if c not in shap_input.columns]
    if missing:
        raise ValueError(f"{SHAP_FILE} is missing required columns: {missing}")


def _normalize_relation(relation: pd.DataFrame) -> pd.DataFrame:
    required = ["prc_step", "metro_step", "metro_item", "subitem_id", "metro_grade"]
    missing = [c for c in required if c not in relation.columns]
    if missing:
        raise ValueError(f"{RELATION_FILE} is missing required columns: {missing}")
    out = relation[required].copy()
    for col in required:
        out[col] = out[col].fillna("").astype(str)
    return out


def _read_optional_bad_wafers(input_dir: str) -> pd.DataFrame | None:
    path = os.path.join(input_dir, BAD_WAFERS_FILE)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _build_target(raw: pd.DataFrame, bad_quantile: float, bad_wafers: pd.DataFrame | None = None) -> pd.DataFrame:
    target = raw[["root_lot_id", "wafer_id", "target"]].copy()
    target = target.rename(columns={"target": TARGET_ID})
    y = pd.to_numeric(target[TARGET_ID], errors="coerce")
    if y.isna().all():
        raise ValueError("raw_data.csv target column must contain numeric values.")
    if bad_wafers is not None:
        bad_keys = _bad_wafer_keys(bad_wafers)
        raw_keys = _combined_keys(target)
        missing = sorted(bad_keys - set(raw_keys))
        if missing:
            sample = ", ".join(missing[:5])
            raise ValueError(
                f"{BAD_WAFERS_FILE} contains wafer IDs not present in raw_data.csv: {sample}"
            )
        target["bad_flag"] = raw_keys.isin(bad_keys).astype(int)
    else:
        threshold = y.quantile(bad_quantile)
        target["bad_flag"] = (y >= threshold).astype(int)
    return target


def _bad_wafer_keys(bad_wafers: pd.DataFrame) -> set[str]:
    if {"root_lot_id", "wafer_id"}.issubset(bad_wafers.columns):
        normalized = bad_wafers[["root_lot_id", "wafer_id"]].dropna().astype(str)
        return set(_combined_keys(normalized))
    if "root_lot_wafer_id" in bad_wafers.columns:
        return set(bad_wafers["root_lot_wafer_id"].dropna().astype(str))
    raise ValueError(
        f"{BAD_WAFERS_FILE} must contain either root_lot_id, wafer_id columns "
        "or a root_lot_wafer_id column using 'root_lot_id|wafer_id'."
    )


def _combined_keys(df: pd.DataFrame) -> pd.Series:
    return df["root_lot_id"].astype(str) + "|" + df["wafer_id"].astype(str)


def _ordered_union(first: Iterable[str], second: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in list(first) + list(second):
        if item not in seen:
            seen.add(item)
            out.append(str(item))
    return out


def _build_feature_dictionary(feature_ids: List[str], relation: pd.DataFrame) -> pd.DataFrame:
    relation_lookup = _relation_lookup(relation)
    rows = [_feature_metadata(feature_id, relation_lookup) for feature_id in feature_ids]
    return pd.DataFrame(rows)


def _relation_lookup(relation: pd.DataFrame) -> Dict[tuple, dict]:
    out: Dict[tuple, dict] = {}
    for _, row in relation.iterrows():
        key = (row["metro_step"], row["metro_item"], row["subitem_id"])
        out[key] = row.to_dict()
    return out


def _feature_metadata(feature_id: str, relation_lookup: Dict[tuple, dict]) -> dict:
    tokens = str(feature_id).split("|")
    kind = tokens[0] if tokens else ""
    lower = str(feature_id).lower()
    is_leakage = any(word in lower for word in ["target", "defect", "yield"])

    base = {
        "feature_id": feature_id,
        "display_name_ko": feature_id,
        "source_type": "UNKNOWN",
        "process_step": "-",
        "module": "-",
        "metric_type": "-",
        "aggregation": "-",
        "wafer_region": "-",
        "wl_or_layer": "-",
        "observed_stage": "before_target",
        "temporal_relation_to_target": "before_target",
        "causal_role": "unknown",
        "leakage_risk": "low",
        "controllability": "unknown",
        "mechanism_group": "unknown",
        "interpretation_template_ko": f"{feature_id} feature 기반 인과 가설 후보입니다.",
        "metro_grade": "",
    }
    if is_leakage:
        base.update(
            causal_role="leakage_or_post_outcome",
            leakage_risk="high",
            temporal_relation_to_target="after_or_derived_from_target",
            observed_stage="after_or_derived",
            controllability="not_controllable",
            mechanism_group="post_outcome_proxy",
        )
        return base

    if kind == "cat":
        category = tokens[1] if len(tokens) > 1 else "category"
        step = tokens[2] if len(tokens) > 2 else "-"
        base.update(
            source_type="PROCESS_HISTORY",
            process_step=step,
            module=category.upper(),
            metric_type=category,
            observed_stage="process_history",
            causal_role="confounder",
            controllability="semi_controllable" if category in {"ppid", "eqp", "eqp_ch"} else "not_controllable",
            mechanism_group=f"{category}_grouping",
            interpretation_template_ko=f"{category} 공정 이력 범주형 변수입니다. 교란 또는 층화 기준으로 해석합니다.",
        )
        return base

    if kind == "num":
        if len(tokens) > 2 and tokens[2] == "erd":
            sensor_item = tokens[4] if len(tokens) > 4 else "erd_sensor"
            base.update(
                source_type="ERD",
                process_step=tokens[1] if len(tokens) > 1 else "-",
                module=tokens[5] if len(tokens) > 5 else "ERD",
                metric_type=sensor_item,
                aggregation=tokens[3] if len(tokens) > 3 else "-",
                observed_stage="in_situ_process",
                causal_role="root_cause_candidate",
                controllability="semi_controllable",
                mechanism_group=sensor_item,
                interpretation_template_ko=f"{sensor_item} ERD 신호는 상류 공정 원인 후보입니다.",
            )
            return base

        relation_key = (
            tokens[1] if len(tokens) > 1 else "",
            tokens[2] if len(tokens) > 2 else "",
            tokens[3] if len(tokens) > 3 else "",
        )
        if relation_key in relation_lookup:
            rel = relation_lookup[relation_key]
            base.update(
                source_type="METROLOGY",
                process_step=rel["prc_step"],
                module="METRO",
                metric_type=rel["metro_item"],
                aggregation=rel["subitem_id"],
                observed_stage="post_process_metrology",
                causal_role="mediator_candidate",
                controllability="not_directly_controllable",
                mechanism_group=f"metro_{rel['metro_item']}",
                interpretation_template_ko=f"{rel['metro_item']} 계측값은 공정 영향이 반영된 매개 신호 후보입니다.",
                metro_grade=rel["metro_grade"],
            )
            return base

        metric = tokens[2] if len(tokens) > 2 else "value"
        base.update(
            source_type="VM",
            process_step=tokens[1] if len(tokens) > 1 else "-",
            module="VM",
            metric_type=metric,
            aggregation=tokens[3] if len(tokens) > 3 else "-",
            observed_stage="in_situ_process",
            causal_role="root_cause_candidate",
            controllability="semi_controllable",
            mechanism_group=metric,
            interpretation_template_ko=f"{metric} VM 신호는 상류 공정 원인 후보입니다.",
        )
        return base

    return base


def _build_feature_level_shap(shap_input: pd.DataFrame, raw: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    bad_wafers = set(target.loc[target["bad_flag"] == 1, "wafer_id"])
    rows: List[dict] = []
    for _, row in shap_input.iterrows():
        feature_id = str(row["feature"])
        shap_value = float(row["shap_value"])
        stats = _feature_value_stats(raw, bad_wafers, feature_id)
        rows.append(
            {
                "root_lot_id": COHORT_ID,
                "wafer_id": COHORT_ID,
                "target_id": TARGET_ID,
                "feature_id": feature_id,
                "feature_value": stats["feature_value"],
                "shap_value": round(shap_value, 6),
                "abs_shap_value": round(abs(shap_value), 6),
                "shap_direction": "+" if shap_value >= 0 else "-",
                "shap_scope": "bad_wafer_mean",
                "bad_mean_feature_value": stats["bad_mean"],
                "overall_mean_feature_value": stats["overall_mean"],
                "bad_recurrence": stats["bad_recurrence"],
                "bad_good_separation_simple": stats["bad_good_separation"],
            }
        )
    return pd.DataFrame(rows)


def _feature_value_stats(raw: pd.DataFrame, bad_wafers: set, feature_id: str) -> dict:
    empty = {
        "feature_value": "",
        "bad_mean": np.nan,
        "overall_mean": np.nan,
        "bad_recurrence": 0.0,
        "bad_good_separation": 0.0,
    }
    if feature_id not in raw.columns:
        return empty

    wafer_ids = raw["wafer_id"]
    values = pd.to_numeric(raw[feature_id], errors="coerce")
    bad_values = values[wafer_ids.isin(bad_wafers)]
    if not values.isna().all():
        overall_mean = float(values.mean())
        overall_std = float(values.std() + 1e-9)
        bad_mean = float(bad_values.mean()) if len(bad_values) else np.nan
        threshold = overall_mean + 0.5 * overall_std
        recurrence = float((bad_values > threshold).mean()) if len(bad_values) else 0.0
        separation = float((bad_mean - overall_mean) / overall_std) if not np.isnan(bad_mean) else 0.0
        return {
            "feature_value": round(bad_mean, 6) if not np.isnan(bad_mean) else "",
            "bad_mean": round(bad_mean, 6) if not np.isnan(bad_mean) else np.nan,
            "overall_mean": round(overall_mean, 6),
            "bad_recurrence": round(recurrence, 6),
            "bad_good_separation": round(separation, 6),
        }

    categorical = raw[feature_id].fillna("").astype(str)
    bad_categorical = categorical[wafer_ids.isin(bad_wafers)]
    if bad_categorical.empty:
        return empty
    mode = bad_categorical.mode().iloc[0] if not bad_categorical.mode().empty else ""
    bad_share = float((bad_categorical == mode).mean()) if mode else 0.0
    overall_share = float((categorical == mode).mean()) if mode else 0.0
    return {
        "feature_value": mode,
        "bad_mean": np.nan,
        "overall_mean": np.nan,
        "bad_recurrence": round(bad_share, 6),
        "bad_good_separation": round(bad_share - overall_share, 6),
    }


def _build_causal_edges(feature_dictionary: pd.DataFrame, relation: pd.DataFrame) -> pd.DataFrame:
    grade_conf = {"A": 0.90, "B": 0.75, "C": 0.60, "D": 0.45}
    fd = feature_dictionary.copy()
    roots = fd[fd["causal_role"] == "root_cause_candidate"]
    mediators = fd[fd["causal_role"] == "mediator_candidate"]
    rows: List[dict] = []

    connected_roots = set()
    for _, med in mediators.iterrows():
        med_roots = roots[roots["process_step"] == med["process_step"]]
        confidence = grade_conf.get(str(med.get("metro_grade", "")).upper(), 0.50)
        for _, root in med_roots.iterrows():
            rows.append(
                {
                    "source_feature": root["feature_id"],
                    "target_feature": med["feature_id"],
                    "relation": "physicallyAffects",
                    "confidence": confidence,
                }
            )
            connected_roots.add(root["feature_id"])
        rows.append(
            {
                "source_feature": med["feature_id"],
                "target_feature": TARGET_ID,
                "relation": "mediates",
                "confidence": confidence,
            }
        )

    for _, root in roots.iterrows():
        if root["feature_id"] not in connected_roots:
            rows.append(
                {
                    "source_feature": root["feature_id"],
                    "target_feature": TARGET_ID,
                    "relation": "triggers",
                    "confidence": 0.40,
                }
            )

    if not rows:
        rows.append({"source_feature": "unknown", "target_feature": TARGET_ID, "relation": "unknown", "confidence": 0.0})
    edges = pd.DataFrame(rows)
    return edges.drop_duplicates().reset_index(drop=True)


def _build_process_history(raw: pd.DataFrame) -> pd.DataFrame:
    cat_cols = [c for c in raw.columns if str(c).startswith("cat|")]
    steps = sorted({str(c).split("|")[2] for c in cat_cols if len(str(c).split("|")) > 2})
    if not steps:
        steps = ["UNKNOWN"]

    records: List[dict] = []
    for _, row in raw.iterrows():
        for order, step in enumerate(steps, start=1):
            records.append(
                {
                    "root_lot_id": row["root_lot_id"],
                    "wafer_id": row["wafer_id"],
                    "step_id": f"{row['wafer_id']}_{step}",
                    "process_step": step,
                    "step_order": order,
                    "tool_id": _row_value(row, f"cat|eqp|{step}"),
                    "chamber_id": _row_value(row, f"cat|eqp_ch|{step}"),
                    "recipe_id": "",
                    "ppid": _row_value(row, f"cat|ppid|{step}"),
                    "slot_n": _row_value(row, f"cat|slot_n|{step}"),
                    "start_time": "",
                    "end_time": _row_value(row, "tkout_time"),
                    "days_since_pm": "",
                }
            )
    return pd.DataFrame(records)


def _row_value(row: pd.Series, column: str) -> str:
    if column not in row.index or pd.isna(row[column]):
        return ""
    return str(row[column])


def _empty_engineer_feedback() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["hypothesis_id", "engineer_judgment", "action_type", "action_status", "outcome", "note"]
    )


def _unknown_ground_truth() -> pd.DataFrame:
    return pd.DataFrame(
        [{"true_root_feature": "", "true_chain_text": "not_provided_for_real_dataset", "true_chamber": ""}]
    )


def _write_causal_edges(edges: pd.DataFrame, path: str) -> None:
    header_comment = "# 이 파일을 편집해 도메인 인과관계를 추가/수정한다 (ontology market)\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(header_comment)
        edges.to_csv(f, index=False)
