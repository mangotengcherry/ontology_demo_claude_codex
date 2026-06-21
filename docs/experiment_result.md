# Experiment Result: Virtual Ontology SHAP Run

## Purpose

This experiment validates the project flow before replacing virtual input files with company data. Virtual mode generates the same three input files expected in real mode, then runs the real adapter and ontology-SHAP analysis pipeline.

## Command

```bash
python3 run_demo.py --mode virtual
```

## Generated Input Contract

Virtual mode creates:

```text
input/raw_data.csv
input/x_feature_shap_value.csv
input/prc_metro_relation.csv
```

These files intentionally follow the real-data contract, so the same pipeline can be used for both evaluation and production-like runs.

## Latest Observed Result

The latest virtual run produced:

```text
virtual wafers / features         : 250 / 9
standard artifacts                : data/
naive top-SHAP (excl. leakage)    : num|MET_CVD|THK_EDGE|AVG [mediator_candidate]
ontology-traced root candidate    : num|CVD|erd|pcf|PRESSURE_SLOPE_P95|CVD
hypothesis cards                  : 3
outputs                           : outputs/hypothesis_cards.csv
report                            : outputs/report.md
```

## Interpretation

The virtual dataset is designed so the top non-leakage SHAP feature is a metrology mediator, not the upstream process root. The ontology relation from `prc_metro_relation.csv` lets the hypothesis engine trace from:

```text
num|MET_CVD|THK_EDGE|AVG
```

to the upstream root-cause candidate:

```text
num|CVD|erd|pcf|PRESSURE_SLOPE_P95|CVD
```

This demonstrates the intended evaluation concept: SHAP provides prioritization, while ontology provides process context and upstream causal-hypothesis tracing.

## Verification Commands

```bash
python3 -m unittest discover -s tests
python3 -m compileall src run_demo.py app.py
python3 run_demo.py --help
```

Expected result:

```text
5 tests pass
compileall exits successfully
CLI exposes --mode {auto,virtual,real}
```
