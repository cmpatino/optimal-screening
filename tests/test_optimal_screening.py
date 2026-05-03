from __future__ import annotations

from optimal_screening.analysis import compute_optimal_screening_actions, compute_optimal_screening_curve


def test_optimal_screening_curve_uses_custom_risk_col() -> None:
    rows = [
        {"risk": 0.9, "outcome": 1, "group": "a"},
        {"risk": 0.8, "outcome": 1, "group": "a"},
        {"risk": 0.2, "outcome": 0, "group": "b"},
        {"risk": 0.1, "outcome": 0, "group": "b"},
    ]

    result = compute_optimal_screening_curve(
        rows=rows,
        outcome_col="outcome",
        strata_features=["group"],
        beta=0.5,
        alpha_quantiles=[0.0, 0.5],
        use_custom_risk_col="risk",
    )

    assert result["alpha_values"] == [0.0, 0.5]
    assert result["total_samples"] == 4
    assert result["total_positive"] == 2
    assert result["true_positives"][1] >= result["true_positives"][0]
    assert len(result["band_info"]) == 2


def test_optimal_screening_curve_validates_alpha_budget() -> None:
    rows = [{"risk": 0.9, "outcome": 1, "group": "a"}]

    try:
        compute_optimal_screening_curve(
            rows=rows,
            outcome_col="outcome",
            strata_features=["group"],
            beta=0.2,
            alpha_quantiles=[0.3],
            use_custom_risk_col="risk",
        )
    except AssertionError as exc:
        assert "exceeds treatment budget" in str(exc)
    else:
        raise AssertionError("Expected alpha > beta to fail")


def test_optimal_screening_actions_preserve_input_order() -> None:
    rows = [
        {"risk": 0.9, "outcome": 1, "group": "a"},
        {"risk": 0.8, "outcome": 1, "group": "a"},
        {"risk": 0.2, "outcome": 0, "group": "b"},
        {"risk": 0.1, "outcome": 0, "group": "b"},
    ]

    actions = compute_optimal_screening_actions(
        rows=rows,
        outcome_col="outcome",
        strata_features=["group"],
        beta=0.5,
        alpha=0.25,
        use_custom_risk_col="risk",
    )

    assert actions == [1, 1, 2, 0]
