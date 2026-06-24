# 진단 및 개선안: Ontology-SHAP Root-Cause PoC

> 팀 논의용 문서. `current_status_and_next_discussion.md` 의 후속.
>
> **구현 상태 (2026-06):** P0(측정 엔진 `src/causal_evidence.py`), P0b(엣지를 측정값으로 가중),
> P1(층화 Simpson 가드), P1b(측정값 기반 evidence path·grade·리포트·앱) **구현 완료**. 가상/실데이터
> 모두에서 `PRESSURE_SLOPE ↑ → THK_EDGE (r=0.94) → 불량률 (a·b=0.80, ~92% 매개, p≈0)` 형태의
> 측정된 인과 사슬을 출력하며 Grade A로 등급. 테스트 15개 통과(`tests/test_causal_evidence.py` 포함).
> P2(per-wafer/interaction SHAP 활용), P2b(data quality 리포트)는 다음 iteration 과제로 남김.

---

## Context — 왜 이 검토를 하는가

실데이터와 유사하게 입력을 맞춰 실험했더니 결과가 **그냥 SHAP value 순서로 보는 것과 별반 다르지 않았다.**
기대했던 출력은 *"어떤 FDC/ERD 센서 값이 thk/cd(계측)를 변동시켜 불량률이 올라갔다"* 같은 **측정된 인과사슬**이지만,
실제 출력은 *"어떤 인자의 SHAP이 높다/낮다"* 수준의 **라벨 + 랭킹**에 그쳤다.

아래는 (1) 왜 그렇게 되는지에 대한 코드 근거 기반 진단, (2) 무엇을 바꿔야 원하는 사슬이 나오는지에 대한 우선순위별 개선안이다.

---

## 1. 핵심 진단 (한 문장)

> **인과 구조를 데이터로 '측정'하지 않고, feature 이름 규칙 + 룩업 테이블로 '단정'한다.
> SHAP은 오직 랭킹에만 쓰인다. 그래서 측정 레이어가 통째로 빠져 있고, 실데이터에서는 SHAP 랭킹으로 회귀한다.**

### 코드 근거

| 위치 | 현재 동작 | 문제 |
|---|---|---|
| `real_dataset_adapter._feature_metadata()` | `causal_role` 을 **이름 접두사로** 부여 (`...erd...`·VM→`root_cause_candidate`, 룩업 매칭된 계측→`mediator_candidate`, `cat\|...`→`confounder`) | "root/mediator" 가 *데이터로 입증된 역할*이 아니라 *이름 종류 라벨*. 데이터가 전혀 개입 안 함 |
| `real_dataset_adapter._build_causal_edges()` | root→mediator 엣지를 `process_step` **문자열이 정확히 같을 때만** 생성. `confidence` 는 `metro_grade` 글자(A/B/C/D) | 센서가 실제로 계측값을 움직였다는 **데이터 증거 없음**. 신뢰도가 사람이 매긴 등급 글자 |
| `hypothesis_engine._trace_upstream_root()` | 이름 기반 `physicallyAffects` 엣지를 거슬러 올라감. 엣지 없으면 mediator 가 "혼자 남음"(`key = med`) | 엣지가 안 만들어지면 결과가 **SHAP 1등 feature 로 붕괴** → 정확히 "랭킹과 똑같다"는 증상 |
| `real_dataset_adapter._feature_value_stats()` | 데이터로 계산하는 유일한 통계 = bad vs 전체 **평균차(단변량)** (`bad_good_separation_simple`, `bad_recurrence`) | feature **사이의 관계**(root↔metro↔target)는 어디서도 측정하지 않음 |

→ 본질적으로 이건 **매개분석(mediation) 문제**인데, 코드에는 매개를 측정하는 레이어가 없다.
가상데이터(`virtual_data_generator.py`)는 `mediator = 1.1·root + noise`, `target = f(mediator)` 로 사슬을 **손으로 심어** 두었기 때문에 데모가 "동작"하는 것처럼 보일 뿐, 실데이터에서는 그 심어둔 구조가 없다.

### 실데이터에서 특히 무너지는 4가지 메커니즘

1. **네이밍 예외 → 문자열 매칭 실패 → 엣지 0개 → 사슬 없음 → SHAP 랭킹.** (root token[1] 과 relation 의 `prc_step` 이 한 글자라도 다르면 엣지 미생성)
2. **진짜 상류 root 의 모델 SHAP 이 낮은 경우가 많다.** 다중공선성으로 다운스트림 계측(mediator)이 credit 을 흡수 → root 가 SHAP 상위 목록에 없으면 trace 가 시작점조차 못 잡음.
3. **엣지 신뢰도가 데이터가 아니라 사람이 매긴 grade 글자**라 강·약을 데이터로 구분 못 함.
4. **행복 경로(가상)조차 사슬에 '크기/숫자'가 없다.** `evidence_path_text` 가 `기구A → 기구B → 불량률 상승` 식 라벨이라, 사슬이 나와도 결국 라벨처럼 읽힌다.

---

## 2. 기대 출력과의 갭

- **원하는 문장**: `ERD PRESSURE_SLOPE_P95 ↑ → THK_EDGE (r=0.62) → 불량률 (slope +0.34), 전체효과의 ~58% 매개`
- **이를 만들려면 필요한 것** (현재 셋 다 없음):
  - (a) root↔metro 연관을 **데이터로 측정**
  - (b) metro↔target 연관을 **데이터로 측정**
  - (c) **매개분해**: indirect(a·b) / direct / proportion-mediated
- 핵심 가용 자원: `input/raw_data.csv` 에 **wafer별 모든 feature 값 + target 이 이미 있다.**
  → **per-wafer SHAP 없이도** raw 값만으로 위 (a)(b)(c) 를 측정할 수 있다. (SHAP 은 "어떤 feature 를 먼저 볼지" 우선순위에만 쓰면 됨)

---

## 3. 개선 방향 (우선순위)

> 전략: 빠진 **"측정 레이어"** 를 넣는다. 역할(role)은 이름으로 *seed* 하되, **무엇이 실제 사슬인지·얼마나 강한지는 데이터로 결정**한다.

### P0 — 데이터 기반 측정 엔진 (mediation / association) ★ 가장 중요
- 새 모듈 `src/causal_evidence.py` (순수 함수, **numpy/pandas 만** — 닫힌형 OLS 로 충분, statsmodels 불필요해 의존성 유지)
- `raw_data` per-wafer 값 + `target` 으로:
  - root→mediator, mediator→target, root→target **연관**(Pearson/Spearman, 표준화 OLS slope) + p값/CI
  - **매개분해**: `indirect = a·b`, direct(c′), `proportion_mediated = a·b / c`, Sobel p
  - 표준화 회귀의 항등식 `a·b ≈ c − c′` 를 내부 검증으로 사용
- 작은 N·강한 공선성(`1−r² ≈ 0`)·결측은 가드/플래그 처리

### P0b — 엣지를 데이터로 구성·가중 (`_build_causal_edges` 교체/보강)
- 이름+룩업으로 후보 엣지를 **seed** 하되, 이름이 안 맞아도 **상관으로 누락 링크 발견**, 모든 엣지를 **측정값으로 가중**(`confidence = |r|`, grade 글자는 표시용으로 강등)
- **측정 링크가 없는 root 는 demote** (사슬로 보고하지 않음) → "모든 root 가 사슬로 나온다" 현상 제거
- 기존 4컬럼 스키마는 유지하고 측정 컬럼(`pearson_r`, `slope_std`, `p_value`, `n`, `indirect`, `prop_mediated`)을 **추가**(하위호환)

### P1 — 층화(stratification) 가드 (confounding / Simpson 방지)
- chamber / PPID 층별로 매개효과 재계산. 사슬이 **층 안에서도 같은 부호로 살아남는지** 확인
- 전체 평균에선 강한데 모든 챔버 안에서 사라지면 → Simpson 경고, 등급 상한 C
- `current_status_and_next_discussion.md` 의 **B항목(Stratification)** 과 직결 — 인과 주장 신뢰도의 핵심. 기존 `_bad_chamber`, `_has_ppid_confounding`, `process_history` 재사용

### P1b — 리포트를 '측정된 사슬'로 (`reporting.py` / `hypothesis_engine` 문자열·grade)
- `evidence_path_text` / `interpretation_text` 에 **측정 숫자**(r, slope, % 매개, p) 표기 → 라벨이 아니라 메커니즘으로 읽힘
- `_grade` 재정의: **측정된 indirect-effect 유의성 + 재현율 + 층 안정성** 기준. **SHAP 크기는 등급 결정자에서 랭킹/타이브레이크로 강등**

### P2 — per-wafer SHAP 활용 ★ (수급 가능 — 강력 추천)
- per-wafer SHAP(mediator) vs raw(root) 상관 → **모델이 실제로 그 매개경로를 쓰는지** 모델-데이터 일치 검증
- **"root 는 데이터 indirect effect 가 큰데 모델 SHAP 은 낮다" = mediator 가 credit 흡수** → 바로 *그* 문제(실데이터에서 SHAP 랭킹이 무력했던 이유)를 **숫자로 증명**
- (가능 시) **interaction SHAP** 으로 b-path 를 모델 내부에서 교차검증 → 스키마에 nullable 컬럼만 추가하면 나중에 무중단으로 끼움

### P2b — Data Quality / 매칭 리포트 (doc 의 A항목)
- SHAP엔 있는데 raw엔 없는 feature / 네이밍 규칙 위반 / relation 매칭 실패 목록을 가시화
  → 측정 엔진이 **조용히 엣지 0개**가 되는 상황을 사람이 즉시 알게 함

> **기존 우선순위 재배치 제안**: doc 은 "A(데이터품질) vs B(층화) 중 무엇 먼저"를 물었지만, 본 진단 기준
> *증상의 근본 원인*은 둘 다 아니라 **측정 레이어 부재(P0)** 다. 권장 순서: **P0 → P0b → P1(층화) → P1b → P2(per-wafer SHAP) → P2b**.

---

## 4. 검증 방법 (구현 단계에서)

- `virtual_data_generator.py` 에:
  - ground-truth 계수(`true_a`, `true_b`, `true_prop_mediated`)를 `ground_truth.csv` 에 기록 (현재 가상 사슬은 root 효과가 mediator 로 **완전 매개** → `prop_mediated ≈ 1.0` 가 정답)
  - **네이밍 불일치 케이스**(prc_step 토큰이 룩업과 다른데 상관은 강함) 추가
  - **Simpson 교란 케이스**(층 안에선 무관, 풀링하면 가짜 상관) 추가
- `tests/test_causal_evidence.py`:
  - `a·b ≈ c − c′` 항등식, `prop_mediated ≈ 1.0` 복원
  - **이름이 안 맞아도 상관으로 사슬 복원** → 회귀 문제 해결 증명
  - 무관한 root 는 사슬 카드 미생성(demote)
  - Simpson 케이스에서 `strata_stable = False` + 등급 상한
- 회귀 안전: 기존 `tests/test_real_dataset_adapter.py` 의 토이 프레임은 N 이 작으므로, 측정 N 부족 시 **이름 seed 엣지를 `evidence="naming_only"` 로 표시**해 남기는 fallback 로 그린 유지

---

## 5. 다음 회의 논의 포인트

1. **측정 엔진(P0)을 1순위로** 진행하는 데 합의 — doc 의 A/B 보다 선행.
2. **per-wafer SHAP / interaction SHAP** 실제 수급 가능 시점·범위 (모델-데이터 일치 검증의 핵심 입력).
3. target 방향성, bad wafer 정의, SHAP 이 signed mean 인지 abs 인지 (doc 7장 미결 항목 — 측정 부호 해석에 직접 영향).
4. 다중비교(root×metro 다수 쌍) 유의성 보정 수준 (효과크기 우선 + BH 플래그 정도).
5. `prc_metro_relation.csv` 외에 **엔지니어 override 레이어**(doc C) — 측정이 놓치는 도메인 엣지 보정.

---

## 6. 결론 (한 줄)

현재 PoC 는 **"이름으로 인과를 단정 + SHAP 으로 랭킹"** 구조라 *측정 레이어가 없어* SHAP 랭킹으로 회귀한다.
`raw_data` 의 wafer별 값으로 **매개효과를 측정**하는 레이어(P0)를 넣고, 엣지·등급·리포트를 **측정값 기반**으로 바꾸면
원하시는 **"센서 → thk/cd → 불량률 (크기·% 매개 포함)"** 사슬이 나온다. per-wafer SHAP 까지 붙이면
*왜 기존 SHAP 랭킹이 상류 원인을 놓쳤는지*(mediator 의 credit 흡수)를 숫자로 증명할 수 있다.
