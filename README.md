# Ontology 기반 SHAP 분석 파이프라인

bad wafer 기준 평균 SHAP 결과를 공정 ontology와 연결해 root-cause candidate와 causal hypothesis report를 생성하는 오프라인 분석 도구입니다.

## 실행 모드

| 모드 | 목적 | 입력 |
|---|---|---|
| `virtual` | 평가/시연용 가상 데이터 생성 후 분석 | 없음. `input/*.csv`를 자동 생성 |
| `real` | 회사 데이터 분석 | `input/raw_data.csv`, `input/prc_metro_relation.csv`, SHAP 입력(아래 참고), optional `input/bad_wafers.csv` |
| `auto` | 기본값 | `input/`에 `raw_data.csv` + `prc_metro_relation.csv` + SHAP 입력이 있으면 `real`, 없으면 `virtual` |

## 빠른 실행

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 평가용 가상 데이터 생성 + 분석
python3 run_demo.py --mode virtual

# 실제 데이터 분석. bad_wafers.csv가 있으면 bad wafer list를 우선 사용합니다.
python3 run_demo.py --mode real --input-dir input --bad-quantile 0.80

# 해석(①) + 성능 비교(②, flat vs ontology CatBoost)를 한 번에
python3 run_demo.py --mode real --input-dir input --with-model-comparison

# 성능 arm만 단독 실행 (+ 차트 PNG 저장)
python3 scripts/model_comparison_demo.py --input-dir input --save-charts outputs/charts

# 가이드 노트북 (팀원용): 평가 실행 + 결과 확인을 한 곳에서
jupyter notebook notebooks/evaluation_guide.ipynb

# 테스트
python3 -m pytest tests/ -q
```

## 실제 데이터 입력 계약

`input/`에 아래 파일을 둡니다. SHAP은 **wide form**(wafer 1행 × feature별 SHAP 컬럼) 두 파일을 사용합니다.

| 파일 | 필수 컬럼 | 의미 |
|---|---|---|
| `raw_data.csv` | `root_lot_id`, `wafer_id`, `tkout_time`, `target`, feature columns | wafer별 원천 데이터 |
| `prc_metro_relation.csv` | `prc_step`, `metro_step`, `metro_item`, `subitem_id`, `metro_grade` | 공정 step과 metro feature의 관계 seed |
| `bad_wafer_shap_value.csv` | `root_lot_id`+`wafer_id`(또는 `root_lot_wafer_id`) + feature 컬럼 | bad wafer에 대한 feature별 SHAP (wide). 평균이 bad-cohort SHAP이 됩니다. |
| `all_wafer_shap_value.csv` | 동일 (good + bad) | good·bad 전체 wafer SHAP (wide). good vs bad SHAP 비교(`shap_cohort_comparison.csv`)에 사용. |
| `bad_wafers.csv` | `root_lot_id`, `wafer_id` 또는 `root_lot_wafer_id` | 선택 입력. SHAP mean 계산에 사용한 bad wafer cohort |

- SHAP 입력 우선순위: `bad_wafer_shap_value.csv` → 없으면 `all_wafer_shap_value.csv`(bad-flag 행으로 cohort 산출) → 없으면 legacy `x_feature_shap_value.csv`(`feature`, `shap_value` long form)도 그대로 지원합니다.
- wide SHAP 파일에 `base_value`, `prediction`, `target` 같은 비-feature 컬럼이 있어도 자동 무시합니다. feature 컬럼명은 `raw_data.csv`의 feature명과 일치해야 매핑됩니다.
- `bad_wafers.csv`가 있으면 이 리스트로 `bad_flag`를 생성합니다. 없을 때만 `--bad-quantile` 기준으로 `target` 상위 wafer를 fallback bad wafer로 잡습니다. `root_lot_wafer_id` 단일 컬럼을 사용할 경우 값은 `root_lot_id|wafer_id` 형식이어야 합니다.

```text
cat|ppid|공정step
cat|eqp|공정step
cat|eqp_ch|공정step
cat|slot_n|공정step
num|metro_step|metro_item|subitem_id
num|prc_step|erd|sensor구분|sensor아이템|sensor구분2
```

## 산출물

실행하면 아래 파일이 생성됩니다. `input/`, `data/`, `outputs/`는 로컬 생성물이며 git에는 올리지 않습니다.

| 경로 | 내용 |
|---|---|
| `data/feature_matrix.csv` | 표준화된 wafer별 feature matrix |
| `data/target.csv` | `target`을 `defect_rate`로 변환하고 `bad_flag` 생성 |
| `data/shap_values.csv` | bad wafer 평균 SHAP 표준 테이블 (importance = mean&#124;SHAP&#124;) |
| `data/shap_cohort_comparison.csv` | good vs bad SHAP 비교 (feature별 bad/good mean&#124;SHAP&#124;, gap, role) — `all_wafer_shap_value.csv`가 있을 때 |
| `data/feature_dictionary.csv` | feature metadata, causal role, leakage flag |
| `data/causal_edges.csv` | ontology graph (root→metro→target), 측정된 `pearson_r`·`indirect`로 가중 |
| `data/mediation.csv` | raw_data로 측정한 root→metro→target 매개효과 (a, b, indirect, %매개, p, 층화 안정성) |
| `outputs/hypothesis_cards.csv` | causal hypothesis card (측정 컬럼 포함) |
| `outputs/ontology_level_shap_summary.csv` | role/process/mechanism 기준 SHAP 집계 |
| `outputs/report.md` | 공유용 분석 report (good vs bad SHAP 비교 섹션 포함) |
| `outputs/charts/*.png` | 성능 arm 차트 6종 (`--save-charts` 또는 `--with-model-comparison`) |

## 해석 원칙

SHAP은 인과를 증명하지 않습니다. 이 프로젝트는 SHAP ranking을 공정 ontology로 보강해 다음을 수행합니다.

1. feature ID를 source/process/role 맥락으로 번역합니다.
2. `leakage_or_post_outcome` feature를 인과 해석에서 제외합니다.
3. metro/inspection 등 mediator feature가 상위 SHAP으로 나와도 `causal_edges.csv`를 upstream으로 추적합니다.
4. 최종 결과를 confirmed root cause가 아니라 causal hypothesis로 보고합니다.

상세 사용 방법과 평가 컨셉은 [docs/team_usage_guide.md](docs/team_usage_guide.md)를 참고하세요.

성능 arm(flat vs ontology CatBoost)의 정직한 win/non-win 해석은 [docs/model_comparison_findings.md](docs/model_comparison_findings.md)에 정리되어 있습니다.

팀원이 가이드를 따라 평가를 실행하고 결과를 확인하려면 [notebooks/evaluation_guide.ipynb](notebooks/evaluation_guide.ipynb)를 사용하세요. 노트북 상단의 `MODE` 로 두 흐름을 선택합니다.

- **MODE 1** — 실데이터 → 온톨로지 → CatBoost 학습 → SHAP·ontology SHAP → 해석 (제공 SHAP 불필요)
- **MODE 2** — 실데이터 → 온톨로지 → 제공된 SHAP → ontology SHAP → 해석 (학습 없음)
