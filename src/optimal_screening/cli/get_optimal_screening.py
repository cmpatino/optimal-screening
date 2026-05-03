from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from optimal_screening.analysis import compute_optimal_screening_actions
from optimal_screening.data_sources import load_dataframe


REQUIRED_FIELDS = {"alpha", "beta", "outcome", "strata"}
DEFAULT_ACTION_COL = "screening_decision"


def _read_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    text = path.read_text()
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    elif path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    else:
        raise ValueError("Config file must be YAML or JSON")

    if not isinstance(data, dict):
        raise ValueError("Config must be a mapping")
    return data


def _validate_config(config: dict[str, Any]) -> dict[str, Any]:
    if "alpha_quantiles" in config:
        raise ValueError("Use alpha for one screening budget; alpha_quantiles is only for curve outputs")

    missing = sorted(REQUIRED_FIELDS - set(config))
    if missing:
        raise ValueError(f"Missing required config fields: {missing}")

    has_csv = config.get("csv") is not None
    has_hf_dataset = config.get("hf_dataset") is not None
    if has_csv == has_hf_dataset:
        raise ValueError("Config must provide exactly one data source: csv or hf_dataset")

    strata = config["strata"]
    if not isinstance(strata, list) or not strata or not all(isinstance(item, str) for item in strata):
        raise ValueError("strata must be a non-empty list of column names")

    beta = float(config["beta"])
    if not 0 < beta <= 1:
        raise ValueError("beta must be in the interval (0, 1]")

    alpha = float(config["alpha"])
    if not 0 <= alpha <= beta:
        raise ValueError(f"alpha must be between 0 and beta={beta}")

    action_col = str(config.get("action_col", DEFAULT_ACTION_COL))
    if not action_col:
        raise ValueError("action_col must not be empty")

    return {
        "csv": str(config["csv"]) if has_csv else None,
        "hf_dataset": str(config["hf_dataset"]) if has_hf_dataset else None,
        "hf_split": str(config.get("hf_split", "train")),
        "hf_revision": str(config["hf_revision"]) if config.get("hf_revision") is not None else None,
        "outcome": str(config["outcome"]),
        "strata": strata,
        "beta": beta,
        "alpha": alpha,
        "prediction_col": str(config.get("prediction_col", "probability")),
        "risk_col": str(config["risk_col"]) if config.get("risk_col") is not None else None,
        "action_col": action_col,
        "output": str(config.get("output", "runs/optimal_screening.csv")),
    }


def get_optimal_screening_from_config(config_path: Path) -> Path:
    config = _validate_config(_read_config(config_path))

    df, dataset_label = load_dataframe(
        csv_path=config["csv"],
        hf_dataset=config["hf_dataset"],
        hf_split=config["hf_split"],
        hf_revision=config["hf_revision"],
    )

    required_cols = {config["outcome"], *config["strata"]}
    if config["risk_col"]:
        required_cols.add(config["risk_col"])
    elif config["prediction_col"] in df.columns:
        required_cols.add(config["prediction_col"])

    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        raise ValueError(f"Missing required columns in {dataset_label}: {missing_cols}")

    if config["action_col"] in df.columns:
        raise ValueError(f"Output action column already exists in {dataset_label}: {config['action_col']}")

    df[config["action_col"]] = compute_optimal_screening_actions(
        rows=df.to_dict("records"),
        outcome_col=config["outcome"],
        strata_features=config["strata"],
        prediction_col=config["prediction_col"],
        beta=config["beta"],
        alpha=config["alpha"],
        use_custom_risk_col=config["risk_col"],
    )

    output_path = Path(config["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Write optimal screening actions from a YAML or JSON config")
    parser.add_argument("config", help="Path to a YAML or JSON config file")
    args = parser.parse_args()

    output_path = get_optimal_screening_from_config(Path(args.config))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
