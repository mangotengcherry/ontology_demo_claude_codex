"""Ontology 기반 SHAP 해석 데모 (Streamlit).

이 앱은 CSV만 읽습니다 — catboost/shap 없이 동작합니다 (데이터 생성 시 1회 계산됨).
실행:  streamlit run app.py   (먼저 python run_demo.py 로 CSV 생성)
"""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from src.loaders import load_all_data
from src.ontology_mapping import map_shap_to_feature_dictionary
from src.shap_interpreter import get_bad_wafers, get_top_shap_features, aggregate_shap_by_context
from src.hypothesis_engine import build_hypothesis_cards, naive_vs_ontology_summary
from src import visualization as viz

DATA_DIR = "data"
OUT_DIR = "outputs"
GRADE_COLOR = {"A": "#2E7D32", "B": "#F9A825", "C": "#1565C0", "X": "#C62828"}

st.set_page_config(page_title="Ontology 기반 SHAP 해석 데모", layout="wide")


@st.cache_data(show_spinner=False)
def _load():
    data = load_all_data(DATA_DIR)
    mapped = map_shap_to_feature_dictionary(data["shap_values"], data["feature_dictionary"])
    return data, mapped


if not os.path.exists(os.path.join(DATA_DIR, "shap_values.csv")):
    st.error("데이터가 없습니다. 먼저 `python run_demo.py` 를 실행해 CSV를 생성하세요.")
    st.stop()

data, mapped = _load()
target = data["target"]
feature_dict = data["feature_dictionary"]

# ── 1. 제목 ─────────────────────────────────────────────────────────────────
st.title("반도체 수율 분석을 위한 Ontology 기반 SHAP 해석 데모")
st.caption("Ontology-driven SHAP Root-Cause Demo for Semiconductor Yield · 매니저 컨셉 데모")

# ── 2. 컨셉 가드레일 ─────────────────────────────────────────────────────────
st.info(
    "**SHAP은 인과를 증명하지 않습니다.** 본 데모는 ontology(feature 메타데이터 + 인과관계 그래프)를 "
    "이용해 raw SHAP feature를 공정 맥락의 인과 *가설(causal hypothesis)*로 변환하며, "
    "최종 검증은 엔지니어가 수행합니다."
)

# ── 3. 데이터셋 개요 ─────────────────────────────────────────────────────────
st.subheader("1) 데이터셋 개요 (dataset overview)")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Wafer 수", len(target))
c2.metric("Bad wafer 수", int(target["bad_flag"].sum()))
c3.metric("Feature 수", mapped["feature_id"].nunique())
c4.metric("Process steps", data["process_history"]["process_step"].nunique())
c5.metric("Source types", feature_dict["source_type"].nunique())

# ── 4. ★ Naive vs Ontology 비교 패널 (헤드라인) ──────────────────────────────
st.subheader("2) ★ Naive SHAP vs Ontology 비교 (headline)")
cmp = naive_vs_ontology_summary(
    mapped, target, data["causal_edges"], feature_dict, data["ground_truth"]
)
b1, b2, b3 = st.columns(3)
b1.markdown(f"**① naive SHAP #1 (증상/symptom)**\n\n🟡 `{cmp['naive_top_nonleak']}`\n\n{cmp['naive_top_nonleak_ko']}")
b2.markdown(f"**② ontology 추적 root (원인/cause)**\n\n🟢 `{cmp['ontology_root']}`\n\n{cmp['ontology_root_ko']}")
b3.markdown(f"**③ ground truth (정답)**\n\n⚫ `{cmp['ground_truth_root']}`\n\n{cmp['ground_truth_root_ko']}")
if cmp["match"]:
    st.success("✓ **일치(match)** — ontology가 추적한 root cause가 ground truth와 일치합니다. "
               "**naive SHAP는 증상을, ontology는 원인을 지목**합니다.")
else:
    st.error("✗ 불일치 — 추적 결과가 ground truth와 다릅니다.")
st.caption(f"참고: 누설(leakage)을 제외하지 않으면 raw SHAP 1위는 `{cmp['naive_top_overall']}` "
           f"({cmp['naive_top_overall_role']}) 입니다 → 가드레일로 제외.")
st.plotly_chart(viz.comparison_panel_figure(cmp), width="stretch")

# ── 5. Bad wafer 선택 ────────────────────────────────────────────────────────
st.subheader("3) Bad wafer 선택 (selector)")
top_k = st.slider("상위 N개 bad wafer", 5, 50, 20, 5)
bad_df = get_bad_wafers(target, top_k=top_k)
st.plotly_chart(viz.bad_wafer_defect_bar(target, top_k=top_k), width="stretch")
sel_wafer = st.selectbox("분석할 wafer 선택", bad_df["wafer_id"].tolist())
sel_lot = target.loc[target["wafer_id"] == sel_wafer, "root_lot_id"].iloc[0]

# ── 6. Raw SHAP view ─────────────────────────────────────────────────────────
st.subheader("4) Raw SHAP view — 선택 wafer")
top_n = st.slider("top-N feature", 5, 12, 10, 1, key="topn")
top_shap = get_top_shap_features(mapped, sel_wafer, sel_lot, top_n=top_n)
left, right = st.columns([1.1, 1])
with left:
    show_cols = ["feature_id", "feature_value", "shap_value", "source_type",
                 "process_step", "causal_role", "leakage_risk", "mechanism_group"]
    disp = top_shap[show_cols].copy()
    disp.insert(0, "⛔", top_shap["is_leakage"].map({True: "⛔", False: ""}))
    st.dataframe(disp, width="stretch", hide_index=True)
    st.caption("⛔ 표시 = 인과 해석 제외(leakage). raw SHAP에서는 크지만 인과 가설에서 배제됩니다.")
with right:
    st.plotly_chart(viz.top_shap_bar(top_shap, sel_wafer), width="stretch")

# ── 7. Ontology-level 집계 ───────────────────────────────────────────────────
st.subheader("5) Ontology-level SHAP 집계 (aggregation)")
group_col = st.radio("집계 기준", ["causal_role", "process_step", "source_type", "mechanism_group"],
                     horizontal=True)
agg = aggregate_shap_by_context(mapped, [group_col], wafer_ids=bad_df["wafer_id"].tolist())
st.plotly_chart(viz.shap_aggregation_bar(agg, group_col), width="stretch")

# ── 8. 인과 가설 카드 ────────────────────────────────────────────────────────
st.subheader("6) 인과 가설 카드 (causal hypothesis cards)")
cards = build_hypothesis_cards(
    mapped, target, data["process_history"], data["causal_edges"], feature_dict, top_n=5
)
for _, card in cards.iterrows():
    grade = card["hypothesis_grade"]
    color = GRADE_COLOR.get(grade, "#555")
    with st.container(border=True):
        st.markdown(
            f"### {card['hypothesis_id']} · "
            f"<span style='color:{color}'>Grade {grade}</span> · 인과 가설(causal hypothesis)",
            unsafe_allow_html=True,
        )
        st.markdown(f"**증거 경로(evidence path):** {card['evidence_path_text']}")
        st.plotly_chart(viz.evidence_path_figure(card["evidence_path_text"]),
                        width="stretch", key=f"path_{card['hypothesis_id']}")
        g1, g2, g3 = st.columns(3)
        g1.markdown(f"- **root_cause_candidate:** {card['root_cause_candidate_features'] or '-'}\n"
                    f"- **mediator_candidate:** {card['mediator_candidate_features'] or '-'}")
        g2.markdown(f"- **proxy:** {card['proxy_features'] or '-'}\n"
                    f"- **excluded leakage:** {card['excluded_leakage_features'] or '-'}")
        g3.markdown(f"- **SHAP 강도:** {card['shap_strength']} · **재현율:** {card['bad_recurrence']}\n"
                    f"- **confounding:** {card['confounding_risk']} · **chamber:** "
                    f"{card['primary_chamber_if_available'] or '-'}")
        st.markdown(f"**해석:** {card['interpretation_text']}")
        st.markdown("**권장 검증(recommended validation):**")
        for item in str(card["recommended_validation"]).split(" | "):
            st.markdown(f"  - {item}")

# ── 9. Feature/Ontology Market 운영 모델 ─────────────────────────────────────
with st.expander("7) Feature/Ontology Market 운영 모델 (governance)"):
    gov = pd.DataFrame(
        [
            ["Data Analytics 팀", "feature_id, source_type, aggregation, leakage risk, 모델 사용 여부", "신규 feature/모델 릴리스"],
            ["제품 엔지니어", "target 의미, defect bin 해석, 제품별 민감도", "신규 제품/ECO"],
            ["공정 엔지니어", "공정 메커니즘, step 관계, controllability, 조치 가이드", "공정/recipe/tool 변경"],
            ["설비 엔지니어", "chamber/tool 거동, FDC 해석, PM 연관", "tool 이슈/PM/변경"],
            ["YE 엔지니어", "가설 피드백, 검증 결과, case memory", "이슈 리뷰/조치 후"],
        ],
        columns=["책임 주체(Owner)", "관리 정보(Maintained Info)", "업데이트 트리거(Update Trigger)"],
    )
    st.table(gov)
    st.caption("기본 골격은 Data Analytics 팀이 CSV로 시딩하고, 심도 있는 인과관계는 제품·공정·설비 "
               "엔지니어가 주기적으로 채워 자산으로 축적한다.")
    st.markdown("**엔지니어 피드백(engineer_feedback, read-only):**")
    st.dataframe(data["engineer_feedback"], width="stretch", hide_index=True)

# ── 10. Export ───────────────────────────────────────────────────────────────
st.subheader("8) Export")
if st.button("outputs/ 로 내보내기 (hypothesis cards + ontology SHAP summary)"):
    os.makedirs(OUT_DIR, exist_ok=True)
    cards.to_csv(os.path.join(OUT_DIR, "demo_hypothesis_cards.csv"), index=False)
    role_agg = aggregate_shap_by_context(mapped, ["causal_role"], wafer_ids=bad_df["wafer_id"].tolist())
    role_agg.to_csv(os.path.join(OUT_DIR, "ontology_level_shap_summary.csv"), index=False)
    st.success(f"저장 완료: {OUT_DIR}/demo_hypothesis_cards.csv, {OUT_DIR}/ontology_level_shap_summary.csv")
