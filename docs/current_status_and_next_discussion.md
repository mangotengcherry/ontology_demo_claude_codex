# Current Status and Next Discussion Notes

## 1. 현재 프로젝트 위치

현재 프로젝트는 production-grade 수율 분석 시스템이라기보다는, **ontology-ready 데이터 구조로 SHAP 해석 가능성을 검증하는 PoC** 단계입니다.

핵심 목표는 다음입니다.

1. bad wafer 기준 평균 SHAP 결과를 입력으로 받는다.
2. feature naming rule과 `prc_metro_relation.csv`를 이용해 최소 ontology graph를 만든다.
3. SHAP 상위 feature를 공정 맥락으로 번역한다.
4. metro/inspection feature가 상위에 나오더라도 upstream process candidate를 추적한다.
5. 결과를 확정 원인이 아니라 causal hypothesis로 report한다.

즉 현재 버전은 “SHAP + ontology 기반 원인 후보 분석이 실제 데이터에서 의미 있는 방향으로 작동하는가”를 확인하기 위한 첫 번째 실험판입니다.

## 2. 현재까지 구현된 내용

### 실행 모드 분리

`run_demo.py`에 세 가지 모드를 분리했습니다.

| 모드 | 목적 |
|---|---|
| `virtual` | 실제 입력 스키마와 동일한 가상 데이터 3종을 생성한 뒤 분석 |
| `real` | 회사 실제 데이터 3종을 입력받아 분석 |
| `auto` | `input/`에 실제 입력 3종이 있으면 real, 없으면 virtual |

대표 실행 명령:

```bash
python3 run_demo.py --mode virtual
python3 run_demo.py --mode real --input-dir input --bad-quantile 0.80
streamlit run app.py
```

### 실제 데이터 입력 계약

실제 데이터는 아래 3개 파일만 교체하면 됩니다.

```text
input/raw_data.csv
input/x_feature_shap_value.csv
input/prc_metro_relation.csv
```

전제:

- `x_feature_shap_value.csv`의 `shap_value`는 bad wafer를 대상으로 계산한 feature별 평균 SHAP입니다.
- wafer별 SHAP이 아니므로 현재 분석은 bad wafer cohort-level SHAP ranking 중심입니다.
- `raw_data.csv`는 bad wafer 정의, feature value anomaly, recurrence, chamber/PPID context 보강에 사용됩니다.

### 주요 모듈

| 파일 | 역할 |
|---|---|
| `src/virtual_data_generator.py` | 평가용 가상 입력 3종 생성 |
| `src/real_dataset_adapter.py` | 실제 입력 3종을 표준 `data/*.csv`로 변환 |
| `src/pipeline.py` | 표준 데이터 로드, hypothesis card, ontology summary, report 생성 |
| `src/reporting.py` | `outputs/report.md` 생성 |
| `src/hypothesis_engine.py` | SHAP feature를 ontology graph로 upstream tracing |
| `docs/team_usage_guide.md` | 팀 사용 가이드 |
| `docs/experiment_result.md` | virtual mode 실험 과정 및 결과 |

### 정리한 항목

기존 Claude Code 작업에서 생성된 synthetic output 중심 파일을 제거했습니다.

- tracked `data/*.csv` 제거
- tracked `outputs/*.csv` 제거
- tracked `docs/assets/*.png` 제거
- 예전 `src/data_generator.py` 제거
- 예전 `scripts/make_readme_assets.py` 제거

현재는 `input/`, `data/`, `outputs/`, `docs/assets/`를 로컬 생성물로 보고 `.gitignore` 처리했습니다.

## 3. 현재 PoC의 의미

현재 구현은 다음 질문에 답하기 위한 수준입니다.

> “bad wafer 기준 평균 SHAP ranking만 가지고도, feature naming rule과 간단한 공정-metro relation을 붙이면 원인 후보를 더 공정 맥락에 맞게 설명할 수 있는가?”

현재 virtual mode에서는 의도적으로 다음 구조를 만듭니다.

```text
upstream process root
  num|CVD|erd|pcf|PRESSURE_SLOPE_P95|CVD

mediator / symptom
  num|MET_CVD|THK_EDGE|AVG
```

실험 결과는 다음 방향을 확인했습니다.

```text
naive top-SHAP excluding leakage:
  num|MET_CVD|THK_EDGE|AVG [mediator_candidate]

ontology-traced root candidate:
  num|CVD|erd|pcf|PRESSURE_SLOPE_P95|CVD
```

즉 SHAP은 “중요한 증상/측정값”을 우선순위화하고, ontology는 “그 증상과 연결된 upstream 공정 후보”를 제안하는 역할을 합니다.

## 4. 내일 실제 데이터 적용 시 확인할 내용

실제 데이터 적용 시에는 먼저 아래 항목을 확인해야 합니다.

### 입력 파일 검증

- `raw_data.csv`에 `root_lot_id`, `wafer_id`, `tkout_time`, `target`이 있는지
- `x_feature_shap_value.csv`의 `feature`가 `raw_data.csv` feature column과 얼마나 매칭되는지
- `prc_metro_relation.csv`의 `metro_step`, `metro_item`, `subitem_id`가 실제 metro feature naming과 일치하는지
- target 방향이 “클수록 나쁨”인지 확인
- `--bad-quantile` 값이 실제 bad wafer 정의와 맞는지 확인

### 첫 실행 후 확인할 산출물

```text
outputs/report.md
outputs/hypothesis_cards.csv
outputs/ontology_level_shap_summary.csv
```

특히 봐야 할 지점:

- SHAP top feature가 모두 `unknown`으로 나오지 않는지
- metro feature가 `mediator_candidate`로 잘 분류되는지
- ERD/VM/process numeric feature가 `root_cause_candidate`로 잘 분류되는지
- leakage 후보가 과하게 잡히거나 너무 적게 잡히지 않는지
- ontology-traced root candidate가 공정적으로 말이 되는지

## 5. 현재 한계

현재 버전은 의도적으로 단순합니다.

1. **진짜 ontology system은 아님**
   - CSV 기반 ontology-ready graph입니다.
   - Palantir Ontology의 object/action/relationship 운영 모델까지는 아직 아닙니다.

2. **SHAP이 feature별 평균값만 있음**
   - wafer별 SHAP variance, confidence interval, stability 분석은 아직 없습니다.
   - feature ranking의 불확실성을 알기 어렵습니다.

3. **stratification이 약함**
   - chamber, PPID, product, route, time window별 bad rate와 SHAP/context 분해가 아직 제한적입니다.

4. **ontology relation이 자동 seed 수준**
   - `prc_metro_relation.csv` 기반 edge만 사용합니다.
   - 공정/설비 엔지니어가 보정하는 override layer가 필요합니다.

5. **검증 workflow가 없음**
   - hypothesis를 만들지만, owner/action/status/post-action effect를 관리하는 루프는 아직 없습니다.

## 6. 다음 고도화 논의 주제

### A. Data Quality Report

우선순위가 높습니다.

추가하면 좋은 항목:

- SHAP에는 있는데 raw data에는 없는 feature
- raw data에는 있는데 SHAP에는 없는 feature
- missing rate
- constant/near-constant feature
- high cardinality categorical feature
- leakage keyword 후보
- feature naming rule 위반 목록
- `prc_metro_relation.csv`와 매칭되지 않는 metro feature

### B. Stratification Analysis

여기서 stratification은 선형관계가 강하다는 뜻이 아니라, 분석을 층별로 나눠 보는 것입니다.

우선순위 후보:

- chamber별 bad rate
- PPID별 bad rate
- product별 bad rate
- route별 bad rate
- time window별 bad rate
- chamber x PPID 조합별 concentration
- top SHAP feature가 특정 chamber/PPID/time window에 집중되는지

목적:

- product mix나 recipe mix에 속는 것을 줄입니다.
- 전체 평균에서는 약하지만 특정 조건에서 강한 원인 후보를 찾습니다.
- chamber 문제인지 recipe 문제인지 분리 검증할 단서를 줍니다.

### C. Ontology Override Layer

자동 생성 ontology와 사람이 보정한 ontology를 분리하는 구조가 필요합니다.

예상 파일:

```text
ontology_overrides/feature_dictionary_override.csv
ontology_overrides/causal_edges_override.csv
```

목적:

- 자동 role 분류 오류 수정
- 공정 엔지니어가 알고 있는 causal edge 추가
- leakage/proxy/confounder 판단 보정
- controllability, recommended validation 보강

### D. Model/SHAP Metadata Manifest

현재 SHAP 파일만으로는 분석 재현 조건을 알기 어렵습니다.

추가하면 좋은 파일:

```text
input/run_metadata.yaml
```

포함할 정보:

- target 정의
- bad wafer 정의
- SHAP 계산 대상 기간
- 모델 종류와 버전
- feature set 버전
- baseline dataset
- SHAP 값이 signed mean인지 mean absolute인지
- target 방향성

### E. Hypothesis Validation Workflow

향후에는 hypothesis card를 action item으로 관리해야 합니다.

추가 컬럼 후보:

- `owner`
- `validation_status`
- `validation_due_date`
- `action_taken`
- `pre_action_bad_rate`
- `post_action_bad_rate`
- `engineer_judgment`
- `evidence_link`

### F. Palantir Ontology 관점 고도화

object type 후보:

- `Wafer`
- `Lot`
- `Product`
- `Route`
- `ProcessStep`
- `Tool`
- `Chamber`
- `Recipe`
- `Measurement`
- `Feature`
- `ModelRun`
- `ShapAttribution`
- `Hypothesis`
- `ValidationAction`

relationship 후보:

- `belongsToLot`
- `processedOn`
- `usedRecipe`
- `measuredBy`
- `hasFeature`
- `hasAttribution`
- `physicallyAffects`
- `mediates`
- `suggestedBy`
- `validatedBy`

action 후보:

- assign validation owner
- request chamber trend review
- request recipe stratification
- mark hypothesis as plausible/rejected
- record post-action result

## 7. 다음 회의에서 결정하면 좋은 질문

1. 실제 `target`은 클수록 bad가 맞는가?
2. bad wafer 정의는 quantile로 충분한가, 아니면 bin/spec 기반으로 해야 하는가?
3. 실제 feature naming rule에서 예외 패턴이 얼마나 있는가?
4. `product`, `route`는 raw_data feature column에 들어오는가, 별도 metadata로 들어오는가?
5. `prc_metro_relation.csv`는 공정팀이 관리할 수 있는가?
6. SHAP 값은 signed mean인가, mean absolute인가?
7. wafer별 SHAP을 추가로 받을 수 있는가?
8. 첫 번째 고도화는 data quality report와 stratification 중 무엇을 먼저 할 것인가?

## 8. 추천 다음 단계

내일 실제 데이터 적용 후에는 아래 순서가 좋습니다.

1. `python3 run_demo.py --mode real --input-dir input --bad-quantile 0.80` 실행
2. `outputs/report.md`에서 unknown/leakage/mediator/root 분류 품질 확인
3. feature 매칭 실패 목록과 relation 매칭 실패 목록을 수동 점검
4. 실제 공정적으로 말이 되는 hypothesis가 나오는지 리뷰
5. 다음 iteration에서 data quality report와 stratification report를 추가
