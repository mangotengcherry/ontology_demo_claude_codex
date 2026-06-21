# Team Usage Guide: Ontology SHAP 분석

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
input/x_feature_shap_value.csv
input/prc_metro_relation.csv
```

이 모드는 팀원 온보딩, 실행 환경 점검, report 형식 검토에 사용합니다.

### Real mode

회사 데이터를 `input/`에 넣고 분석합니다.

```bash
python3 run_demo.py --mode real --input-dir input --bad-quantile 0.80
```

`--bad-quantile 0.80`은 `target` 상위 20%를 bad wafer로 정의한다는 뜻입니다. 이미 SHAP은 bad wafer만 대상으로 계산된 평균값이어야 합니다.

### Auto mode

기본값입니다.

```bash
python3 run_demo.py
```

`input/`에 실제 입력 3개가 모두 있으면 real mode, 없으면 virtual mode로 실행합니다.

## 3. 데이터 교체 절차

1. 기존 `input/` 폴더를 비웁니다.
2. 아래 3개 파일을 넣습니다.

```text
input/raw_data.csv
input/x_feature_shap_value.csv
input/prc_metro_relation.csv
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

5. 대시보드를 엽니다.

```bash
streamlit run app.py
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

### x_feature_shap_value.csv

필수 컬럼:

```text
feature, shap_value
```

중요한 전제:

- `shap_value`는 bad wafer만 대상으로 계산한 feature별 평균 SHAP입니다.
- wafer별 SHAP이 아니므로 대시보드의 SHAP view는 선택 wafer가 아니라 bad wafer cohort 기준 ranking으로 표시됩니다.
- `feature` 값은 `raw_data.csv`의 feature column 이름과 일치해야 합니다.

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

## 5. 산출물 읽는 법

### outputs/report.md

공유용 요약 보고서입니다. 다음 순서로 읽습니다.

1. Dataset Summary: wafer 수, bad wafer 수, SHAP feature 수
2. Bad wafer cohort SHAP: bad wafer 평균 SHAP 상위 feature
3. Ontology-level Summary: causal role, process step, mechanism group별 집계
4. Causal Hypothesis Cards: root-cause candidate, mediator, 권장 검증

### outputs/hypothesis_cards.csv

주요 컬럼:

| 컬럼 | 의미 |
|---|---|
| `root_cause_candidate_features` | ontology가 upstream으로 추적한 원인 후보 |
| `mediator_candidate_features` | SHAP 상위에 등장한 증상/계측 후보 |
| `excluded_leakage_features` | target proxy로 판단되어 제외한 feature |
| `hypothesis_grade` | A/B/C/X 등급. X는 leakage 제외 카드 |
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
