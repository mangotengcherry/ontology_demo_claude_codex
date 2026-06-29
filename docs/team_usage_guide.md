# Team Usage Guide: Ontology SHAP 분석

## 0. 빠른 시작 — 설치와 실행 순서

처음 쓰는 팀원은 이 순서만 따라가면 됩니다. 코드 파일을 차례로 실행할 필요 없이 **진입점은 `run_demo.py` 하나**입니다.

### 0-1. 1회 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 0-2. 실제 데이터 넣기

`input/` 폴더에 아래 파일을 둡니다 (형식은 4장 참고).

```text
input/raw_data.csv                # 필수 — wafer별 원천값 + target
input/bad_wafer_shap_value.csv    # 필수 — bad wafer SHAP (wide: wafer 1행 × feature 컬럼)
input/all_wafer_shap_value.csv    # 권장 — good+bad SHAP (wide). good vs bad 비교에 사용
input/prc_metro_relation.csv      # 필수 — 공정-계측 관계 seed
input/bad_wafers.csv              # 권장 — SHAP 계산에 쓴 bad wafer cohort
# (legacy: input/x_feature_shap_value.csv 의 long form [feature, shap_value]도 그대로 지원)
```

### 0-3. 분석 실행 (한 줄)

```bash
python3 run_demo.py --mode real --input-dir input --bad-quantile 0.80
```

이 한 줄이 내부적으로 순서대로 수행합니다.

1. `src/real_dataset_adapter.py` — input 3종을 표준 `data/*.csv`로 변환하고, **raw_data의 wafer별 값으로 root→metro→target 매개효과를 측정**해 `data/mediation.csv`·`data/causal_edges.csv`를 만듭니다.
2. `src/pipeline.py` → `src/hypothesis_engine.py` — 측정값 기준으로 hypothesis card와 grade를 만들어 `outputs/`에 저장합니다.
3. `src/reporting.py` — `outputs/report.md`를 작성합니다.

### 0-4. 결과 확인

```text
outputs/report.md                      # 공유용 요약 (여기부터 읽기)
outputs/hypothesis_cards.csv           # 카드 원본 (측정 컬럼 포함)
outputs/ontology_level_shap_summary.csv
data/mediation.csv                     # 측정된 매개효과 원본 (a, b, indirect, %매개, p, 층화 안정성)
```

### 0-5. 가이드 노트북 (권장)

```bash
jupyter notebook notebooks/evaluation_guide.ipynb
```

> 평가 실행 + 결과 확인 + 성능 비교 + 모델 해석을 한 노트북에서 위에서부터 실행하면 됩니다.
> 형식만 먼저 보고 싶으면 데이터 없이 `python3 run_demo.py --mode virtual` 로 가상 데이터를 생성·분석해 산출물 형식을 확인할 수 있습니다.

### 0-6. 성능 arm — flat vs ontology CatBoost (선택)

해석(①) 외에 **학습(②)** 측면 — ontology semantic layer 가 단순 tabular CatBoost 대비
도움이 되는지 — 를 5블록(PERFORMANCE/ATTRIBUTION/ONTOLOGY-LEVEL SHAP/MEASURED CAUSAL
CHAIN/LEARNING CURVE)으로 보여줍니다.

```bash
# 단독 실행 (input 3종이 없으면 가상데이터 자동 생성)
python3 scripts/model_comparison_demo.py --input-dir input

# 시각화까지 PNG 로 저장 (5블록 + 인과사슬 + credit absorption 차트)
python3 scripts/model_comparison_demo.py --input-dir input --save-charts outputs/charts

# 해석 run 뒤에 이어서 실행
python3 run_demo.py --mode real --input-dir input --with-model-comparison
```

**노트북으로 분석:** `notebooks/evaluation_guide.ipynb` 를 열어 위에서부터 실행하면
역할 분류·측정 사슬·5블록 차트·credit absorption·good vs bad SHAP 비교를 한 곳에서 본다.
`input/` 에 실제 데이터를 넣고 `Restart & Run All` 하면 동일 셀이 실데이터로 채워진다.

> **주의:** 주장은 "큰 N 정확도 향상"이 아니라 **누수 차단·저데이터 inductive bias·귀속이
> 손잡이로** 세 가지 win 으로만 한다. 정직한 비-win 과 함께 `docs/model_comparison_findings.md`
> 를 먼저 읽고 발표할 것.

## 1. 평가 컨셉

이 프로젝트의 목적은 bad wafer 기준 평균 SHAP ranking을 그대로 원인으로 확정하지 않고, 공정 ontology를 이용해 검증 가능한 causal hypothesis로 바꾸는 것입니다.

평가 관점은 세 가지입니다.

1. **SHAP 해석 보강**: feature ID와 SHAP 숫자를 공정 step, source type, causal role로 번역합니다.
2. **가드레일**: target 이후 값이나 target proxy로 의심되는 feature는 `leakage_or_post_outcome`으로 표시하고 인과 해석에서 제외합니다.
3. **upstream 추적**: metro feature가 SHAP 상위에 있더라도 `prc_metro_relation.csv`로 만든 ontology edge를 따라 upstream 공정 parameter를 root-cause candidate로 제안합니다.

결과는 확정 원인이 아니라 **causal hypothesis**입니다. 엔지니어 검증, recipe/chamber stratification, FDC trend 확인이 뒤따라야 합니다.

## 2. 모드 구분

### Virtual mode

가상 회사 데이터 3종을 `input/`에 생성하고, 실제 데이터와 동일한 adapter/pipeline으로 분석합니다.

```bash
python3 run_demo.py --mode virtual
```

생성되는 입력:

```text
input/raw_data.csv
input/all_wafer_shap_value.csv    # wide
input/bad_wafer_shap_value.csv    # wide
input/x_feature_shap_value.csv    # legacy long (back-compat)
input/prc_metro_relation.csv
input/bad_wafers.csv
```

이 모드는 팀원 온보딩, 실행 환경 점검, report 형식 검토에 사용합니다.

### Real mode

회사 데이터를 `input/`에 넣고 분석합니다.

```bash
python3 run_demo.py --mode real --input-dir input --bad-quantile 0.80
```

실제 데이터에서 SHAP mean을 계산한 bad wafer 목록을 알고 있다면 `input/bad_wafers.csv`를 함께 넣는 것을 권장합니다. 이 파일이 있으면 `--bad-quantile`은 사용되지 않고, 입력 리스트 기준으로 `bad_flag`가 생성됩니다. `--bad-quantile 0.80`은 `bad_wafers.csv`가 없을 때만 `target` 상위 20%를 fallback bad wafer로 잡는 옵션입니다.

### Auto mode

기본값입니다.

```bash
python3 run_demo.py
```

`input/`에 `raw_data.csv` + `prc_metro_relation.csv` + SHAP 입력이 모두 있으면 real mode, 없으면 virtual mode로 실행합니다.

## 3. 데이터 교체 절차

1. 기존 `input/` 폴더를 비웁니다.
2. 아래 파일을 넣습니다.

```text
input/raw_data.csv
input/bad_wafer_shap_value.csv    # wide (또는 legacy x_feature_shap_value.csv)
input/all_wafer_shap_value.csv    # wide, 권장 (good vs bad 비교용)
input/prc_metro_relation.csv
```

가능하면 SHAP mean 계산에 사용한 bad wafer cohort(`bad_wafers.csv`)도 같이 넣습니다.

```text
input/bad_wafers.csv
```

3. 실행합니다.

```bash
python3 run_demo.py --mode real --input-dir input --bad-quantile 0.80
```

4. 결과를 확인합니다.

```text
outputs/hypothesis_cards.csv
outputs/ontology_level_shap_summary.csv
outputs/report.md
```

5. 결과를 확인합니다 (report 또는 가이드 노트북).

```bash
cat outputs/report.md
jupyter notebook notebooks/evaluation_guide.ipynb
```

## 4. 입력 파일 상세

### raw_data.csv

필수 컬럼:

```text
root_lot_id, wafer_id, tkout_time, target
```

나머지 컬럼은 feature입니다. 현재 parser는 아래 naming rule을 사용해 metadata를 자동 생성합니다.

| 패턴 | 해석 |
|---|---|
| `cat|ppid|STEP` | recipe/PPID categorical feature |
| `cat|eqp|STEP` | equipment categorical feature |
| `cat|eqp_ch|STEP` | chamber categorical feature |
| `cat|slot_n|STEP` | slot categorical feature |
| `num|METRO_STEP|METRO_ITEM|SUBITEM` | metro numeric feature |
| `num|PRC_STEP|erd|SENSOR_GROUP|SENSOR_ITEM|SENSOR_GROUP2` | upstream ERD/process numeric feature |
| 기타 `num|STEP|ITEM|VALUE` | VM/process numeric feature |

### bad_wafer_shap_value.csv / all_wafer_shap_value.csv (wide form)

SHAP 입력은 wafer 1행 × feature별 SHAP 컬럼의 **wide form** 두 파일을 사용합니다.

```text
root_lot_id, wafer_id, <feature_1>, <feature_2>, ...      # 또는 root_lot_wafer_id 단일 컬럼
```

- `bad_wafer_shap_value.csv` — bad wafer 대상 SHAP. 컬럼별 평균이 bad-cohort 평균 SHAP(`shap_values.csv`)이 됩니다.
- `all_wafer_shap_value.csv` — good+bad 전체 SHAP. good vs bad SHAP 비교(`data/shap_cohort_comparison.csv`)와 성능 arm의 "provided model SHAP" 블록에 사용합니다.
- importance는 mean(|SHAP|), 방향은 signed mean으로 분리 계산합니다(양방향 feature가 과소평가되지 않도록).
- feature 컬럼명은 `raw_data.csv`의 feature column 이름과 일치해야 매핑됩니다. `base_value`/`prediction`/`target` 같은 비-feature 컬럼은 자동 무시됩니다.
- 입력 우선순위: `bad_wafer_shap_value.csv` → `all_wafer_shap_value.csv`(bad-flag 행으로 cohort 산출) → legacy `x_feature_shap_value.csv`(`feature`, `shap_value` long form).
- wafer별 SHAP이 아니라 cohort 평균으로 집계되므로, 대시보드 SHAP view는 bad wafer cohort 기준 ranking으로 표시됩니다.

### bad_wafers.csv

선택 입력이지만 실제 데이터 분석에서는 사용하는 것을 권장합니다. 이 파일은 SHAP mean을 계산할 때 사용한 bad wafer cohort와 동일해야 합니다.

지원 형식 1:

```text
root_lot_id, wafer_id
```

지원 형식 2:

```text
root_lot_wafer_id
```

`root_lot_wafer_id` 값은 아래처럼 `|` delimiter를 사용합니다.

```text
LOT123|W05
```

`bad_wafers.csv`에 있는 wafer가 `raw_data.csv`에 없으면 실행을 중단합니다. 이는 SHAP mean cohort와 raw data cohort가 어긋난 상태를 조용히 통과시키지 않기 위한 검증입니다.

### prc_metro_relation.csv

필수 컬럼:

```text
prc_step, metro_step, metro_item, subitem_id, metro_grade
```

이 파일은 ontology edge의 seed입니다. 예를 들어 아래 row는 `num|CVD|...` 계열 upstream process feature가 `num|MET_CVD|THK_EDGE|AVG` metro feature에 영향을 줄 수 있다는 edge를 만듭니다.

```text
prc_step= CVD
metro_step= MET_CVD
metro_item= THK_EDGE
subitem_id= AVG
metro_grade= A
```

## 4.5 측정된 인과 사슬 조건 (measured vs naming_only)

이번 버전은 인과 엣지를 feature 이름으로 '단정'하지 않고, raw_data의 wafer별 값으로 root→metro→target 매개효과를 **측정**합니다. 카드의 `evidence_basis`가 두 가지로 나옵니다.

- `measured`: 매개효과를 데이터로 측정했고 `r`, `indirect a·b`, `% 매개`, `p`가 카드/리포트에 표기됩니다. (원하는 "센서 → thk/cd → 불량률" 형태의 측정된 사슬)
- `naming_only`: 측정이 불가능해 이름 기반 엣지로 fallback한 경우. 측정 숫자가 없습니다.

`measured`로 나오게 하려면:

1. **root·mediator feature의 wafer별 수치가 `raw_data.csv`에 컬럼으로 들어 있어야 합니다.** SHAP 파일은 feature별 평균 한 줄이라 측정에 못 쓰고, 측정은 raw_data의 per-wafer 값으로 합니다.
2. **wafer 수가 충분해야 합니다** (기본 최소 30, `src/causal_evidence.py`의 `DEFAULT_MIN_N`). 너무 적으면 naming_only로 떨어집니다.
3. **root feature와 metro(mediator) feature가 실제로 상관**이 있어야 측정 엣지가 생깁니다. 이름의 `prc_step`이 달라도 상관이 강하면(기본 |r|≥0.30, `DISCOVERY_R`) 자동으로 엣지를 찾습니다.
4. **target은 숫자**여야 하고 "클수록 나쁨" 방향이어야 합니다. 아니면 `bad_wafers.csv`로 bad cohort를 직접 지정합니다.

`data/mediation.csv`를 열면 root→mediator 쌍별 측정값(a, b, indirect, prop_mediated, p, strata_stable)을 직접 볼 수 있습니다. 비어 있으면 위 1~4 중 하나가 충족되지 않은 것입니다.

## 5. 산출물 읽는 법

### outputs/report.md

공유용 요약 보고서입니다. 다음 순서로 읽습니다.

1. Dataset Summary: wafer 수, bad wafer 수, SHAP feature 수
2. Bad wafer cohort SHAP: bad wafer 평균 SHAP 상위 feature
3. Ontology-level Summary: causal role, process step, mechanism group별 집계
4. Causal Hypothesis Cards: 측정된 evidence path(`root ↑ → metro (r=..) → 불량률 (a·b=.., %매개, p=..)`), root-cause candidate, mediator, 권장 검증

### outputs/hypothesis_cards.csv

주요 컬럼:

| 컬럼 | 의미 |
|---|---|
| `root_cause_candidate_features` | ontology가 upstream으로 추적한 원인 후보 |
| `mediator_candidate_features` | SHAP 상위에 등장한 증상/계측 후보 |
| `excluded_leakage_features` | target proxy로 판단되어 제외한 feature |
| `hypothesis_grade` | A/B/C/X 등급. 측정된 indirect effect·유의성·층화 안정성 기준. X는 leakage |
| `evidence_basis` | `measured`(데이터로 측정) 또는 `naming_only`(이름 fallback) |
| `measured_indirect_effect` | 측정된 매개효과 a·b (표준화) |
| `measured_prop_mediated` | 전체효과 중 매개 비율 |
| `measured_indirect_p` | 매개효과 유의확률 (Sobel, 정규근사) |
| `strata_stable` | chamber/PPID 층화 시 사슬이 유지되는지 (Simpson 가드) |
| `recommended_validation` | 엔지니어 검증 action |

## 6. 결과 해석 시 주의사항

- `hypothesis_grade=A/B`도 확정 원인이 아닙니다.
- `cat|ppid|...`, `cat|eqp_ch|...`는 원인이라기보다 stratification/confounding 확인 축으로 보는 것이 안전합니다.
- `target`, `defect`, `yield` 문자열이 들어간 feature는 leakage 후보로 자동 분류됩니다.
- `prc_metro_relation.csv` 품질이 낮으면 upstream tracing 품질도 낮아집니다.

## 7. 권장 운영 방식

1. Data Analytics 팀이 feature naming과 SHAP 파일을 생성합니다.
2. 공정/설비 엔지니어가 `prc_metro_relation.csv`의 relation과 `metro_grade`를 관리합니다.
3. 분석자는 `outputs/report.md`를 리뷰하고, 주요 card의 validation action을 엔지니어에게 전달합니다.
4. 검증 결과는 다음 iteration에서 ontology relation과 feature role 개선에 반영합니다.
