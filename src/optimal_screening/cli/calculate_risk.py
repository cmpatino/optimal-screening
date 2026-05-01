from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from optimal_screening.analysis import compute_optimal_screening_curve


REQUIRED_FIELDS = {"csv", "outcome", "strata", "beta"}


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


def _as_float_sequence(values: Any, field: str) -> list[float] | None:
    if values is None:
        return None
    if not isinstance(values, list | tuple):
        raise ValueError(f"{field} must be a list of numbers")
    return [float(value) for value in values]


def _validate_config(config: dict[str, Any]) -> dict[str, Any]:
    missing = sorted(REQUIRED_FIELDS - set(config))
    if missing:
        raise ValueError(f"Missing required config fields: {missing}")

    strata = config["strata"]
    if not isinstance(strata, list) or not strata or not all(isinstance(item, str) for item in strata):
        raise ValueError("strata must be a non-empty list of column names")

    beta = float(config["beta"])
    if not 0 < beta <= 1:
        raise ValueError("beta must be in the interval (0, 1]")

    alpha_quantiles = _as_float_sequence(config.get("alpha_quantiles"), "alpha_quantiles")
    if alpha_quantiles is not None:
        invalid = [alpha for alpha in alpha_quantiles if alpha < 0 or alpha > beta]
        if invalid:
            raise ValueError(f"alpha_quantiles must be between 0 and beta={beta}; invalid values: {invalid}")

    return {
        "csv": str(config["csv"]),
        "outcome": str(config["outcome"]),
        "strata": strata,
        "beta": beta,
        "prediction_col": str(config.get("prediction_col", "probability")),
        "risk_col": str(config["risk_col"]) if config.get("risk_col") is not None else None,
        "alpha_quantiles": alpha_quantiles,
        "output": str(config.get("output", "runs/optimal_screening_curve.json")),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def calculate_from_config(config_path: Path) -> Path:
    config = _validate_config(_read_config(config_path))

    csv_path = Path(config["csv"])
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {config["outcome"], *config["strata"]}
    if config["risk_col"]:
        required_cols.add(config["risk_col"])
    elif config["prediction_col"] in df.columns:
        required_cols.add(config["prediction_col"])

    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        raise ValueError(f"Missing required columns in {csv_path}: {missing_cols}")

    result = compute_optimal_screening_curve(
        rows=df.to_dict("records"),
        outcome_col=config["outcome"],
        strata_features=config["strata"],
        prediction_col=config["prediction_col"],
        beta=config["beta"],
        alpha_quantiles=config["alpha_quantiles"],
        use_custom_risk_col=config["risk_col"],
    )

    output_path = Path(config["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_json_safe(result), indent=2))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute an optimal screening curve from a YAML or JSON config")
    parser.add_argument("config", help="Path to a YAML or JSON config file")
    args = parser.parse_args()

    output_path = calculate_from_config(Path(args.config))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
