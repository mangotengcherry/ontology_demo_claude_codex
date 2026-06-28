# 성능 arm 결과 — flat vs ontology CatBoost

> 실행: `python3 scripts/model_comparison_demo.py --input-dir input`
> (input 3종이 없으면 가상데이터를 자동 생성해 동일 파이프라인으로 검증한다.)
>
> 이 문서는 **학습(②) 측면**의 정직한 결과와 한계다. 해석(①) 측면은
> `docs/diagnosis_and_improvement_plan.md` + `src/causal_evidence.py` 를 본다.

---

## 0. 한 문장

> 잘 튜닝된 flat CatBoost 는 강한 baseline 이다. **"큰 N에서 정확도 향상"은 주장하지 않는다.**
> 비판적 청중 앞에서 깨지지 않는 win 은 **누수 차단 · 저데이터 inductive bias · 귀속이 손잡이로** 세 가지뿐이며,
> 이 스크립트는 그 셋만 보여준다.

---

## 1. 평가 셋업 (공정 비교 가드레일)

| 항목 | 값 | 이유 |
|---|---|---|
| target | `defect_rate` (연속) 회귀, R² | bad_flag 분류가 아니라 연속값 — 누수 효과가 R²에 직접 드러남 |
| split / seed / iterations | **두 arm 동일** | 튜닝 트릭 의심 차단 |
| depth | flat **8** / ontology **4** | depth 차이가 곧 inductive-bias 레버. gap을 항상 같이 보고 |
| 누수 제외 | **causal role 기반** (`leakage_or_post_outcome`) | 정답을 알고 떨구는 oracle 아님. 이름이 `...FINAL...`/`target`/`defect`/`yield` 면 role 부여 |
| 노이즈 feature | **떨구지 않음** | 오라클 금지. QUEUE_TIME 등은 그대로 둠 |
| monotone | **A-grade metro 에만**, train 상관 부호로만 | 비단조(U자) 관계엔 해로우므로 좁게 적용 |

콘솔은 5블록 출력: PERFORMANCE / ATTRIBUTION / ONTOLOGY-LEVEL SHAP / MEASURED CAUSAL CHAIN / LEARNING CURVE.

---

## 2. 견고한 win 3가지 (가상데이터 N=250 기준 실측)

### win 1 — 누수 차단 (정직한 달성치를 드러냄)

| arm | depth | train R² | test R² | gap |
|---|---|---|---|---|
| flat (전체, 누수 포함) | 8 | 0.999 | **0.981** | 0.018 |
| flat 절제 (누수 수동 제거) | 8 | 0.989 | 0.891 | 0.098 |
| ontology (role 필터 + shallow) | 4 | 0.961 | **0.897** | 0.064 |

- flat 의 test R² 0.981 은 **post-outcome proxy(`num\|FINAL\|TARGET_PROXY\|VALUE`, SHAP 1위)** 가 부풀린 가짜다.
  양산 시점엔 존재하지 않을 변수.
- 손으로 누수를 떼면 정직한 천장은 ~0.891. **ontology 는 이 천장을 causal role 로 자동 달성**(0.897).
- 절제 arm 을 같이 보여줘 "gap = 누수"임을 증명한다(튜닝 아님).

### win 2 — 저데이터 inductive bias (N이 작을수록 ontology 우위)

> 누수는 win 1 에서 이미 보였으므로, learning curve 는 **양 arm 모두 누수 제외**하고
> depth(8 vs 4) + monotone 만 차이로 남겨 *순수 inductive-bias 효과*만 측정한다.

| train N | flat test R² | onto test R² | onto − flat |
|---|---|---|---|
| 17 | 0.611 | 0.723 | **+0.111** |
| 35 | 0.810 | 0.859 | +0.049 |
| 70 | 0.850 | 0.871 | +0.021 |
| 122 | 0.880 | 0.895 | +0.015 |
| 175 | 0.891 | 0.905 | +0.014 |

- N이 작을수록 격차가 크다 → **신규 제품·희소 BIN·신규 결함 모드**처럼 표본이 적은 국면이 ontology 가 사는 자리.
- 큰 N으로 갈수록 격차가 좁아진다. **큰 N 정확도 향상은 주장하지 않는다.**

### win 3 — 귀속(attribution)이 손잡이(control handle)로

ATTRIBUTION (held-out fold, mean|SHAP|):

```
FLAT     43.2% [leakage] FINAL|TARGET_PROXY   19.0% [mediator] THK_EDGE   14.8% [root] PRESSURE_SLOPE
ONTOLOGY 51.9% [mediator] THK_EDGE            17.4% [root] QUEUE_TIME     16.2% [root] PRESSURE_SLOPE
```

ONTOLOGY-LEVEL roll-up (causal role 별): mediator 51.9% / **root 38.0%** / confounder 10.1%.

MEASURED CAUSAL CHAIN (데이터로 측정한 매개):

```
ERD PRESSURE_SLOPE_P95 -> THK_EDGE  (a=0.94, b=0.86, indirect a*b=0.80, ~92% 매개, p≈0, n=250)
CVD RF_TIME_1          -> THK_EDGE  (a=0.55, b=0.90, indirect a*b=0.50, ~93% 매개, p≈0, n=250)
```

- flat SHAP 은 계측(증상)인 THK_EDGE 를 지목.
- ontology + mediation 은 **controllable root**(ERD PRESSURE_SLOPE)를 지목 → 엔지니어가 돌릴 수 있는 손잡이.
- `ERD slope → metro THK (a·b≈0.80, ~92% 완전매개)`.

---

## 3. 정직한 비-win (데모에서 먼저 말한다)

1. **큰 N 정확도 향상 없음.** learning curve 의 격차는 N이 클수록 닫힌다. 가상데이터에선 생성과정이
   depth-4 와 잘 맞아 큰 N에서도 미세하게 onto 가 앞서지만(+0.014), 이는 **가상데이터 아티팩트**다.
   실데이터에선 flat 이 따라잡을 것으로 본다.
2. **CV 안정성은 헤드라인 금지.** setup 의존이며, 누수가 있으면 flat 이 오히려 안정적으로 보인다.
3. **`prop_mediated > 100%`** 는 노이즈/억제(suppression) 아티팩트 — 사슬 카드에 플래그로 표시한다.
4. **monotone 제약은 비단조(U자) 관계에 해롭다** → A-grade 에만, train 상관 부호로만 적용.

---

## 4. 실데이터에서 깨지면 보는 곳 (체크리스트)

- 사슬이 안 나오면 → `docs/diagnosis_and_improvement_plan.md` §3, `src/causal_evidence.py` 의
  `DISCOVERY_R`/`DEFAULT_MIN_N`/`COLLINEARITY_D`.
- 이름 규칙 예외로 mediator 미분류 → root↔metro 엣지 0개 → SHAP 랭킹 회귀.
  `_feature_metadata` 토큰 매칭, `prc_metro_relation` 의 `prc_step` 문자열 일치 확인.
- 진짜 상류 root 의 모델 SHAP 이 낮음(다중공선성으로 mediator 가 credit 흡수) →
  ATTRIBUTION 에서 root 비중이 낮게 나오면 P1 의 per-wafer SHAP 진단으로 확인.

---

## 5. 한계 / 다음

- 이 스크립트는 `run_demo.py`/`app.py` 와 아직 분리돼 있다(P2 통합 대상).
- learning curve 는 가상데이터에서 검증됨. **실데이터 5블록 캡처는 사내에서 실행**해야 한다(실데이터 사외 반출 금지).
- mediation 의 다중비교 보정(BH-FDR)·비선형 옵션·per-wafer SHAP 진단은 P1 에서 보강.
