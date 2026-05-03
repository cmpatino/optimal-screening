#!/usr/bin/env python
"""Train tabular baselines, compute screening curves, and save JSON artifacts.

Usage:
    python scripts/run_tabular_models.py --hf-dataset cmpatino/landmine-detection --outcome mines_outcome \
        --strata Municipio --beta 0.1 --test-municipio GRANADA \
        --output runs/screening_tabular_landmines_granada_test.json

    python scripts/run_tabular_models.py --hf-dataset cmpatino/acs-income-2018 --outcome "PINCP > 50k" \
        --strata AGEP COW SCHL MAR OCCP POBP RELP WKHP SEX RAC1P --beta 0.3 --strata-only \
        --test-size 0.5 --output runs/screening_tabular_acs2018_all_strata_50pct.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


sys.path.insert(0, "src")

from optimal_screening.analysis import compute_optimal_screening_curve, compute_random_screening_curve
from optimal_screening.data_sources import load_dataframe


def _is_positive(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return int(value.strip().lower() in {"true", "1", "yes", "t", "y"})
    if isinstance(value, (int, float, np.integer, np.floating)):
        return int(value > 0)
    return int(bool(value))


def _detect_column_types(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    for col in features.columns:
        try:
            pd.to_numeric(features[col], errors="raise")
            numeric_cols.append(col)
        except (TypeError, ValueError):
            categorical_cols.append(col)
    return numeric_cols, categorical_cols


def _build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    transformers: list[tuple[str, object, list[str]]] = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _build_pipelines(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            [
                ("pre", preprocessor),
                ("clf", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("pre", preprocessor),
                ("clf", RandomForestClassifier(n_estimators=200, random_state=42)),
            ]
        ),
        "XGBoost": Pipeline(
            [
                ("pre", preprocessor),
                ("clf", XGBClassifier(n_estimators=200, random_state=42, eval_metric="logloss")),
            ]
        ),
    }


def _auc_or_none(y_true: pd.Series, probabilities: np.ndarray) -> float | None:
    if y_true.nunique() < 2:
        return None
    return float(roc_auc_score(y_true, probabilities))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tabular models and compute screening curves")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", help="Path to a local CSV data file")
    source.add_argument("--hf-dataset", help="Hugging Face dataset repository ID")
    parser.add_argument("--hf-split", default="train", help="Hugging Face dataset split")
    parser.add_argument("--hf-revision", default=None, help="Optional Hugging Face dataset revision")
    parser.add_argument("--outcome", required=True, help="Outcome column name")
    parser.add_argument("--strata", nargs="+", required=True, help="Feature columns defining risk strata")
    parser.add_argument("--beta", type=float, required=True, help="Treatment budget beta")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    parser.add_argument("--test-municipio", default=None, help="Use this Municipio value as the test set")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--strata-only", action="store_true", help="Restrict ML models to strata features only")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    df, dataset_label = load_dataframe(
        csv_path=args.csv,
        hf_dataset=args.hf_dataset,
        hf_split=args.hf_split,
        hf_revision=args.hf_revision,
    )

    required_cols = {args.outcome, *args.strata}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {dataset_label}: {missing}")

    outcome_col = args.outcome
    feature_cols = args.strata if args.strata_only else [col for col in df.columns if col != outcome_col]
    y = df[outcome_col].map(_is_positive).astype(int)
    features = df[feature_cols]

    print(f"Dataset: {dataset_label}")
    print(f"Samples: {len(features)}, Features: {features.shape[1]}, Positive rate: {y.mean():.2%}")

    if args.test_municipio:
        if "Municipio" not in df.columns:
            raise ValueError("--test-municipio requires a Municipio column")
        municipio_mask = df["Municipio"].astype(str).str.upper() == args.test_municipio.upper()
        if not municipio_mask.any():
            raise ValueError(f"No rows matched Municipio={args.test_municipio!r}")
        x_train, x_test = features[~municipio_mask], features[municipio_mask]
        y_train, y_test = y[~municipio_mask], y[municipio_mask]
        print(f"Train: {len(x_train)} (all except {args.test_municipio}), Test: {len(x_test)} ({args.test_municipio})")
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            y,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=y,
        )
        print(f"Train: {len(x_train)}, Test: {len(x_test)}")

    numeric_cols, categorical_cols = _detect_column_types(x_train)
    preprocessor = _build_preprocessor(numeric_cols, categorical_cols)
    pipelines = _build_pipelines(preprocessor)

    model_probs: dict[str, np.ndarray] = {}
    for name, pipeline in pipelines.items():
        print(f"Training {name}...")
        pipeline.fit(x_train, y_train)
        probs = pipeline.predict_proba(x_test)[:, 1]
        model_probs[name] = probs
        auc = _auc_or_none(y_test, probs)
        auc_text = "undefined (single class test set)" if auc is None else f"{auc:.4f}"
        print(f"  AUC: {auc_text}")

    test_df = x_test.copy()
    test_df[outcome_col] = y_test.values
    for name, probs in model_probs.items():
        test_df[f"_risk_{name}"] = probs
    rows = test_df.to_dict("records")

    print(f"\nComputing screening curves (beta={args.beta:.1%})...")
    all_results: dict[str, dict[str, Any]] = {}
    for name in model_probs:
        col_name = f"_risk_{name}"
        print(f"  {name}...")
        all_results[name] = compute_optimal_screening_curve(
            rows=rows,
            outcome_col=outcome_col,
            strata_features=args.strata,
            prediction_col=col_name,
            beta=args.beta,
            use_custom_risk_col=col_name,
        )

    first_col = f"_risk_{next(iter(model_probs))}"
    print("  Random baseline...")
    all_results["Random"] = compute_random_screening_curve(
        rows=rows,
        outcome_col=outcome_col,
        strata_features=args.strata,
        prediction_col=first_col,
        beta=args.beta,
        seed=args.seed,
        use_custom_risk_col=first_col,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_to_save: dict[str, Any] = {}
    for label, res in all_results.items():
        results_to_save[label] = {
            "alpha_values": [float(alpha) for alpha in res["alpha_values"]],
            "true_positives": [int(tp) for tp in res["true_positives"]],
        }

    first_result = next(iter(all_results.values()))
    results_to_save["meta"] = {
        "beta": args.beta,
        "total_positive": first_result["total_positive"],
        "total_samples": first_result["total_samples"],
        "dataset": dataset_label,
    }
    results_to_save["risk_scores"] = {name: probs.tolist() for name, probs in model_probs.items()}
    results_to_save["risk_scores"]["y_test"] = y_test.tolist()

    output_path.write_text(json.dumps(results_to_save, indent=2))
    print(f"\nData saved to {output_path}")


if __name__ == "__main__":
    main()
