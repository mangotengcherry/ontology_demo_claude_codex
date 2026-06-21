"""Run ontology-SHAP analysis with virtual input data or real company inputs.

Default behavior is `--mode auto`: if `input/raw_data.csv`,
`input/x_feature_shap_value.csv`, and `input/prc_metro_relation.csv` exist, the
real dataset adapter is used. Otherwise virtual company-style input CSVs are
generated first, then the same real-data pipeline is executed.
"""
from __future__ import annotations

import argparse
import os

from src.pipeline import run_real_dataset_pipeline
from src.real_dataset_adapter import input_files_exist
from src.virtual_data_generator import generate_virtual_input_dataset

DATA_DIR = "data"
OUT_DIR = "outputs"
INPUT_DIR = "input"


def main() -> None:
    args = _parse_args()
    mode = _resolve_mode(args.mode, args.input_dir)
    if mode == "real":
        result = run_real_dataset_pipeline(
            input_dir=args.input_dir,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            bad_quantile=args.bad_quantile,
        )
        _print_real_summary(result, args.data_dir, args.output_dir)
        return

    virtual_summary = generate_virtual_input_dataset(args.input_dir, n_wafers=args.virtual_wafers)
    result = run_real_dataset_pipeline(
        input_dir=args.input_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        bad_quantile=args.bad_quantile,
    )
    _print_virtual_summary(virtual_summary, result, args.input_dir, args.data_dir, args.output_dir)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ontology-SHAP analysis runner")
    parser.add_argument("--mode", choices=["auto", "virtual", "real"], default="auto")
    parser.add_argument("--input-dir", default=INPUT_DIR)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--output-dir", default=OUT_DIR)
    parser.add_argument(
        "--bad-quantile",
        type=float,
        default=0.80,
        help="Target quantile used to flag bad wafers when raw_data.csv has no bad_flag column.",
    )
    parser.add_argument("--virtual-wafers", type=int, default=250, help="Number of wafers to generate in virtual mode.")
    return parser.parse_args()


def _resolve_mode(mode: str, input_dir: str) -> str:
    if mode != "auto":
        return mode
    return "real" if input_files_exist(input_dir) else "virtual"


def _print_real_summary(result: dict, data_dir: str, output_dir: str) -> None:
    data = result["data"]
    summary = result["summary"]
    cards = result["cards"]
    print("\n" + "=" * 78)
    print("ONTOLOGY-SHAP REAL DATASET RUN")
    print("=" * 78)
    print(f"standard artifacts               : {data_dir}/")
    print(f"wafers / bad wafers              : {len(data['target'])} / {int(data['target']['bad_flag'].sum())}")
    print(f"features with SHAP               : {data['shap_values']['feature_id'].nunique()}")
    print(f"naive top-SHAP (excl. leakage)   : {summary['naive_top_nonleak']} [{summary['naive_top_nonleak_role']}]")
    print(f"ontology-traced root candidate   : {summary['ontology_root'] or '미상'}")
    print(f"hypothesis cards                 : {len(cards)}")
    print(f"outputs                          : {output_dir}/hypothesis_cards.csv")
    print(f"report                           : {os.path.join(output_dir, 'report.md')}")
    print("=" * 78)
    print("다음 단계: streamlit run app.py")


def _print_virtual_summary(virtual_summary: dict, result: dict, input_dir: str, data_dir: str, output_dir: str) -> None:
    summary = result["summary"]
    cards = result["cards"]
    print("\n" + "=" * 78)
    print("ONTOLOGY-SHAP VIRTUAL DATASET RUN")
    print("=" * 78)
    print(f"virtual input files               : {input_dir}/")
    print(
        f"virtual wafers / bad wafers / features: "
        f"{virtual_summary['n_wafers']} / {virtual_summary['n_bad_wafers']} / {virtual_summary['n_features']}"
    )
    print(f"standard artifacts                : {data_dir}/")
    print(f"naive top-SHAP (excl. leakage)    : {summary['naive_top_nonleak']} [{summary['naive_top_nonleak_role']}]")
    print(f"ontology-traced root candidate    : {summary['ontology_root'] or '미상'}")
    print(f"hypothesis cards                  : {len(cards)}")
    print(f"outputs                           : {output_dir}/hypothesis_cards.csv")
    print(f"report                            : {os.path.join(output_dir, 'report.md')}")
    print("=" * 78)
    print("다음 단계: streamlit run app.py")


if __name__ == "__main__":
    main()
