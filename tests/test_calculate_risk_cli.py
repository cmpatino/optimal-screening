from __future__ import annotations

import json
from pathlib import Path

import pytest

from optimal_screening.cli.calculate_risk import _validate_config, calculate_from_config


def test_validate_config_requires_core_fields() -> None:
    with pytest.raises(ValueError, match="Missing required config fields"):
        _validate_config({"csv": "data.csv"})


def test_validate_config_rejects_alpha_above_beta() -> None:
    with pytest.raises(ValueError, match="alpha_quantiles"):
        _validate_config(
            {
                "csv": "data.csv",
                "outcome": "outcome",
                "strata": ["group"],
                "beta": 0.2,
                "alpha_quantiles": [0.0, 0.3],
            }
        )


def test_calculate_from_json_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
        "alpha_quantiles": [0.0, 0.5],
        "output": "runs/result.json",
    }
    Path("configs/example.json").write_text(json.dumps(config))

    output_path = calculate_from_config(Path("configs/example.json"))

    assert output_path == Path("runs/result.json")
    result = json.loads(output_path.read_text())
    assert result["total_samples"] == 4
    assert result["alpha_values"] == [0.0, 0.5]
