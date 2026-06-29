"""Preflight validation of the input/ contract so teams can swap data and run.

`check_inputs(input_dir)` returns a structured report (errors / warnings / info)
without raising, and `format_report` renders it as a ✅/⚠️/❌ checklist. The
notebook runs this before building so common mistakes -- a missing file, a target
that is not numeric, or wide-SHAP column names that do not match the raw feature
names (which would otherwise silently map to nothing) -- surface immediately.
"""
from __future__ import annotations

import os
from typing import Dict, List

import pandas as pd

from src import shap_inputs

RAW_FILE = "raw_data.csv"
RELATION_FILE = "prc_metro_relation.csv"
RELATION_COLUMNS = ["prc_step", "metro_step", "metro_item", "subitem_id", "metro_grade"]
RAW_NON_FEATURE = {"root_lot_id", "wafer_id", "tkout_time", "target"}
BAD_WAFER_FILES = ("bad_wafers.csv", "bad_wafer_list.csv")


def check_inputs(input_dir: str = "input", require_shap: bool = True) -> Dict[str, object]:
    """Validate the input/ contract. Returns {ok, errors, warnings, info}.

    `require_shap=False` (notebook MODE 1: train CatBoost) treats a missing SHAP
    export as informational rather than an error, since the model makes its own.
    """
    errors: List[str] = []
    warnings: List[str] = []
    info: List[str] = []

    def path(name: str) -> str:
        return os.path.join(input_dir, name)

    # --- raw_data.csv (required) ---------------------------------------------------
    if not os.path.exists(path(RAW_FILE)):
        errors.append(f"{RAW_FILE} 없음 (필수)")
        return _result(errors, warnings, info)
    raw = pd.read_csv(path(RAW_FILE))
    for col in ("root_lot_id", "wafer_id", "target"):
        if col not in raw.columns:
            errors.append(f"{RAW_FILE}: 필수 컬럼 '{col}' 없음")
    feature_cols = [c for c in raw.columns if c not in RAW_NON_FEATURE]
    if "target" in raw.columns and pd.to_numeric(raw["target"], errors="coerce").isna().all():
        errors.append(f"{RAW_FILE}: target 컬럼이 숫자가 아닙니다 (클수록 나쁨 방향이어야 함)")
    if not feature_cols:
        errors.append(f"{RAW_FILE}: feature 컬럼이 없습니다")
    info.append(f"{RAW_FILE}: wafer {len(raw)}행, feature {len(feature_cols)}개")

    # --- prc_metro_relation.csv (required) ----------------------------------------
    if not os.path.exists(path(RELATION_FILE)):
        errors.append(f"{RELATION_FILE} 없음 (필수)")
    else:
        rel = pd.read_csv(path(RELATION_FILE))
        missing = [c for c in RELATION_COLUMNS if c not in rel.columns]
        if missing:
            errors.append(f"{RELATION_FILE}: 누락 컬럼 {missing}")
        else:
            info.append(f"{RELATION_FILE}: 공정-계측 관계 {len(rel)}건")

    # --- SHAP input ---------------------------------------------------------------
    shap_files = [shap_inputs.BAD_SHAP_FILE, shap_inputs.ALL_SHAP_FILE, shap_inputs.LEGACY_SHAP_FILE]
    present = [f for f in shap_files if os.path.exists(path(f))]
    if not present:
        msg = "SHAP 입력 없음 (bad_wafer_shap_value.csv / all_wafer_shap_value.csv / x_feature_shap_value.csv)"
        if require_shap:
            errors.append(msg + " — MODE 2/해석에는 필수")
        else:
            info.append("SHAP 입력 없음 — MODE 1은 CatBoost 학습으로 SHAP 을 생성합니다")
    else:
        info.append("SHAP 입력: " + ", ".join(present))
        for wide_file in (shap_inputs.BAD_SHAP_FILE, shap_inputs.ALL_SHAP_FILE):
            if os.path.exists(path(wide_file)):
                _check_wide_overlap(pd.read_csv(path(wide_file)), wide_file, feature_cols, errors, warnings, info)

    # --- bad wafer labelling ------------------------------------------------------
    _check_bad_wafers(input_dir, raw, errors, info)

    return _result(errors, warnings, info)


def _check_wide_overlap(wide, wide_file, feature_cols, errors, warnings, info) -> None:
    matched = [c for c in shap_inputs.detect_feature_columns(wide, feature_cols) if c in set(feature_cols)]
    n_feat = len(feature_cols)
    if not matched:
        errors.append(
            f"{wide_file}: feature 컬럼이 raw_data 의 feature명과 하나도 일치하지 않습니다 → SHAP 매핑 불가 "
            "(wide 컬럼명을 raw_data feature명과 동일하게 맞추세요)"
        )
    elif n_feat and len(matched) < 0.5 * n_feat:
        warnings.append(f"{wide_file}: raw feature {n_feat}개 중 {len(matched)}개만 일치 (컬럼명 확인 권장)")
    else:
        info.append(f"{wide_file}: feature {len(matched)}/{n_feat} 일치")


def _check_bad_wafers(input_dir, raw, errors, info) -> None:
    bad_path = next((os.path.join(input_dir, f) for f in BAD_WAFER_FILES
                     if os.path.exists(os.path.join(input_dir, f))), None)
    if bad_path is None:
        info.append("bad wafer 리스트 없음 → bad_quantile 값으로 판정 (없으면 실행 중단)")
        return
    bad = pd.read_csv(bad_path)
    bad_keys = _bad_keys(bad)
    if bad_keys is None:
        errors.append(f"{os.path.basename(bad_path)}: root_lot_id+wafer_id 또는 root_lot_wafer_id 컬럼이 필요합니다")
        return
    raw_keys = set((raw["root_lot_id"].astype(str) + "|" + raw["wafer_id"].astype(str)))
    missing = sorted(bad_keys - raw_keys)
    if missing:
        errors.append(
            f"{os.path.basename(bad_path)}: raw_data 에 없는 wafer {len(missing)}개 "
            f"(예: {', '.join(missing[:3])})"
        )
    else:
        info.append(f"bad wafer 판정: 리스트 사용 ({len(bad_keys)}개)")


def _bad_keys(bad: pd.DataFrame):
    if {"root_lot_id", "wafer_id"}.issubset(bad.columns):
        b = bad[["root_lot_id", "wafer_id"]].dropna().astype(str)
        return set(b["root_lot_id"] + "|" + b["wafer_id"])
    if "root_lot_wafer_id" in bad.columns:
        return set(bad["root_lot_wafer_id"].dropna().astype(str))
    return None


def _result(errors, warnings, info) -> Dict[str, object]:
    return {"ok": not errors, "errors": errors, "warnings": warnings, "info": info}


def format_report(report: Dict[str, object]) -> str:
    """Render the report as a ✅/⚠️/❌ checklist string."""
    lines = ["입력 점검 (preflight)", "=" * 40]
    for msg in report["info"]:
        lines.append(f"  ✅ {msg}")
    for msg in report["warnings"]:
        lines.append(f"  ⚠️  {msg}")
    for msg in report["errors"]:
        lines.append(f"  ❌ {msg}")
    lines.append("-" * 40)
    lines.append("  결과: " + ("✅ 통과 — 실행 가능" if report["ok"] else "❌ 실패 — 위 항목 수정 필요"))
    return "\n".join(lines)
