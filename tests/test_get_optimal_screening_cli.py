from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from optimal_screening.cli.get_optimal_screening import _validate_config, get_optimal_screening_from_config


def test_validate_config_requires_core_fields() -> None:
    with pytest.raises(ValueError, match="Missing required config fields"):
        _validate_config({"csv": "data.csv"})


def test_validate_config_rejects_alpha_quantiles() -> None:
    with pytest.raises(ValueError, match="Use alpha"):
        _validate_config(
            {
                "csv": "data.csv",
                "outcome": "outcome",
                "strata": ["group"],
                "beta": 0.2,
                "alpha_quantiles": [0.0, 0.1],
            }
        )


def test_validate_config_rejects_alpha_above_beta() -> None:
    with pytest.raises(ValueError, match="alpha must be between"):
        _validate_config(
            {
                "csv": "data.csv",
                "outcome": "outcome",
                "strata": ["group"],
                "beta": 0.2,
                "alpha": 0.3,
            }
        )


def test_validate_config_requires_one_data_source() -> None:
    base = {
        "outcome": "outcome",
        "strata": ["group"],
        "beta": 0.2,
        "alpha": 0.1,
    }

    with pytest.raises(ValueError, match="exactly one data source"):
        _validate_config(base)

    with pytest.raises(ValueError, match="exactly one data source"):
        _validate_config(base | {"csv": "data.csv", "hf_dataset": "owner/name"})


def test_get_optimal_screening_from_json_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    Path("data").mkdir()
    Path("configs").mkdir()
    Path("data/example.csv").write_text("risk,outcome,group\n0.9,1,a\n0.8,1,a\n0.2,0,b\n0.1,0,b\n")

    config = {
        "csv": "data/example.csv",
        "outcome": "outcome",
        "strata": ["group"],
        "risk_col": "risk",
        "beta": 0.5,
        "alpha": 0.25,
        "action_col": "action",
        "output": "runs/result.csv",
    }
    Path("configs/example.json").write_text(json.dumps(config))

    output_path = get_optimal_screening_from_config(Path("configs/example.json"))

    assert output_path == Path("runs/result.csv")
    result = pd.read_csv(output_path)
    assert list(result.columns) == ["risk", "outcome", "group", "action"]
    assert list(result["action"]) == [1, 1, 2, 0]
