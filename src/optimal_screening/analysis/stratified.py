from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np


SIMULATION_SIZE = 100_000

RISK_PRESETS: dict[str, tuple[float, float] | float] = {
    "uniform": (1.0, 1.0),
    "bimodal": (0.1, 0.1),
    "unimodal": (10.0, 10.0),
    "delta_half": 0.5,
}


def compute_empirical_probabilities(
    rows: list[dict[str, Any]],
    outcome_col: str,
    strata_features: Sequence[str],
) -> dict[tuple[Any, ...], dict[str, Any]]:
    """Compute empirical P(Y=1|X) for each feature stratum from true outcomes.

    Args:
        rows: List of data rows (dicts) with features and outcome
        outcome_col: Name of the outcome column (e.g., "PINCP > 50k")
        strata_features: List of feature names to define strata (e.g., ['AGEP', 'SEX'])

    Returns:
        Dictionary mapping stratum key -> {
            'probability': empirical P(Y=1|X) = (# Y=1) / (# total),
            'count': number of samples in stratum,
            'positive_count': number of Y=1 samples,
            'features': dict of feature values for this stratum
        }

    Example:
        >>> rows = [
        ...     {'AGEP': 35, 'SEX': 1, 'PINCP > 50k': True},
        ...     {'AGEP': 35, 'SEX': 1, 'PINCP > 50k': False},
        ...     {'AGEP': 35, 'SEX': 1, 'PINCP > 50k': True},
        ... ]
        >>> strata = compute_empirical_probabilities(rows, 'PINCP > 50k', ['AGEP', 'SEX'])
        >>> strata[(35, 1)]['probability']
        0.6666666666666666
        >>> strata[(35, 1)]['count']
        3
    """
    # Group by strata and count outcomes
    strata_counts: dict[tuple[Any, ...], dict[str, int]] = defaultdict(lambda: {"total": 0, "positive": 0})
    strata_features_map: dict[tuple[Any, ...], dict[str, Any]] = {}

    for row in rows:
        # Create stratum key from selected features
        stratum_key = tuple(row.get(f) for f in strata_features)

        # Count outcomes
        outcome_value = row.get(outcome_col)
        strata_counts[stratum_key]["total"] += 1

        # Convert outcome to boolean (handle "True"/"False" strings, True/False, 1/0, etc.)
        if _is_positive_outcome(outcome_value):
            strata_counts[stratum_key]["positive"] += 1

        # Store feature values for this stratum
        if stratum_key not in strata_features_map:
            strata_features_map[stratum_key] = {f: row.get(f) for f in strata_features}

    # Compute empirical P(Y=1|X) for each stratum
    result = {}
    for stratum_key, counts in strata_counts.items():
        total = counts["total"]
        positive = counts["positive"]

        result[stratum_key] = {
            "probability": positive / total if total > 0 else 0.0,
            "count": total,
            "positive_count": positive,
            "features": strata_features_map[stratum_key],
        }

    return result


def _is_positive_outcome(value: Any) -> bool:
    """Helper to determine if outcome value represents Y=1."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes", "t", "y")
    return False


def generate_simulation_data(
    a: float | None = None,
    b: float | None = None,
    size: int = SIMULATION_SIZE,
    seed: int | None = None,
    point_mass: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic risk scores and binary outcomes.

    Supports two modes:
    - **Beta-Binomial**: risk_scores ~ Beta(a, b), outcomes ~ Binomial(1, risk_scores).
    - **Point mass**: all risk_scores = *point_mass*, outcomes ~ Binomial(1, point_mass).

    Args:
        a: Alpha parameter of the Beta distribution (ignored when *point_mass* is set).
        b: Beta parameter of the Beta distribution (ignored when *point_mass* is set).
        size: Number of samples to generate.
        seed: Random seed for reproducibility.
        point_mass: If provided, every risk score is set to this constant value.

    Returns:
        Tuple of (risk_scores, outcomes).
    """
    rng = np.random.default_rng(seed)
    if point_mass is not None:
        risk_scores = np.full(size, point_mass)
    else:
        risk_scores = rng.beta(a, b, size=size)
    outcomes = rng.binomial(1, risk_scores)
    return risk_scores, outcomes


def compute_optimal_screening_curve(
    rows: list[dict[str, Any]],
    outcome_col: str,
    strata_features: Sequence[str],
    prediction_col: str = "probability",
    beta: float = 0.5,
    alpha_quantiles: Sequence[float] | None = None,
    max_iterations: int = 20,
    tolerance: float = 1e-6,
    seed: int | None = None,
    use_custom_risk_col: str | None = None,
    simulation: str | tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Compute optimal screening curve with treatment budget β and screening budget α.

    Band structure (highest to lowest risk):
    - Band 1: Top (β - α) - Treated, model predictions
    - Band 2: Next (α - avg_risk(Band 3)) - Treated, model predictions
    - Band 3: Next α - Screened (true outcomes)
    - Band 4: Bottom (1 - β - α + avg_risk) - Untreated (predict 0)

    Uses iterative method to resolve circular dependency between Band 2 and Band 3.

    Args:
        rows: List of data rows with features, outcome, and predictions
        outcome_col: Name of outcome column
        strata_features: Features defining strata for computing empirical P(Y=1|X)
        prediction_col: Column name for model predictions
        beta: Treatment budget (proportion who can be treated)
        alpha_quantiles: Screening budget levels to evaluate
        max_iterations: Maximum iterations for convergence
        tolerance: Convergence tolerance for avg_risk
        seed: Random seed for uniform distribution override (for debugging)
        use_custom_risk_col: If provided, use this column for risk instead of computing
            empirical probabilities from strata. Useful for comparing LLM predictions
            with empirical baselines.
        simulation: If provided, generate synthetic data from a Beta distribution instead
            of using real data. Pass a preset name ('uniform', 'bimodal', 'unimodal') or
            a tuple (a, b) of Beta distribution parameters. Uses SIMULATION_SIZE samples.

    Returns:
        Dictionary with screening curves and band information
    """
    if alpha_quantiles is None:
        # Default: 10 equally spaced values from 0 to beta
        alpha_quantiles = [beta * i / 49 for i in range(50)]

    # Assign each row its risk (simulation, custom, or empirical)
    rows_with_risk = []

    if simulation is not None:
        # Generate synthetic data from a Beta distribution
        if isinstance(simulation, str):
            if simulation not in RISK_PRESETS:
                raise ValueError(f"Unknown simulation preset '{simulation}'. Choose from {list(RISK_PRESETS.keys())}.")
            preset = RISK_PRESETS[simulation]
        else:
            preset = simulation

        if isinstance(preset, (int, float)):
            risk_scores, outcomes = generate_simulation_data(size=SIMULATION_SIZE, seed=seed, point_mass=float(preset))
        else:
            a, b = preset
            risk_scores, outcomes = generate_simulation_data(a, b, size=SIMULATION_SIZE, seed=seed)

        for i in range(SIMULATION_SIZE):
            rows_with_risk.append(
                {
                    "row": {"_sim_index": i, "_sim_feature": 0, outcome_col: bool(outcomes[i])},
                    "empirical_risk": float(risk_scores[i]),
                    "true_outcome": bool(outcomes[i]),
                    "model_prediction": float(risk_scores[i]),
                }
            )
    elif use_custom_risk_col is not None:
        # Use custom risk column directly
        for row in rows:
            risk = row.get(use_custom_risk_col, 0.5)
            rows_with_risk.append(
                {
                    "row": row,
                    "empirical_risk": risk,
                    "true_outcome": _is_positive_outcome(row.get(outcome_col)),
                    "model_prediction": row.get(prediction_col, 0.5),
                }
            )
    else:
        # Compute empirical P(Y=1|X) for each stratum
        empirical_probs = compute_empirical_probabilities(rows, outcome_col, strata_features)

        for row in rows:
            stratum_key = tuple(row.get(f) for f in strata_features)
            empirical_risk = empirical_probs.get(stratum_key, {}).get("probability", 0.5)

            rows_with_risk.append(
                {
                    "row": row,
                    "empirical_risk": empirical_risk,
                    "true_outcome": _is_positive_outcome(row.get(outcome_col)),
                    "model_prediction": row.get(prediction_col, 0.5),
                }
            )

    # Sort by risk (highest to lowest)
    rows_with_risk.sort(key=lambda x: x["empirical_risk"], reverse=True)

    total_positive = sum(1 for r in rows_with_risk if r["true_outcome"])
    n = len(rows_with_risk)

    # Results storage
    results = {
        "beta": beta,
        "alpha_values": [],
        "true_positives": [],
        "band_info": [],
        "total_positive": total_positive,
        "total_samples": n,
    }

    for alpha in alpha_quantiles:
        assert alpha <= beta, f"Screening budget α={alpha} exceeds treatment budget β={beta}"

        # Iteratively find Band 3 position.
        prev_avg_risk = 0.0

        for _iteration in range(max_iterations):
            # Compute target mass: ∫ f(risk) d(risk) = target
            # Where f(risk) is the density over risk values
            # For discrete: sum of (count at each risk / total count) = proportion of population at that risk
            band1_target_mass = beta - alpha
            # Band 2 size: ∫ 1 × f(risk) d(risk) over Band 2 = ∫ (1 - risk) × f(risk) d(risk) over Band 3
            # Since Band 3 has mass α and average risk prev_avg_risk:
            # ∫ (1 - risk) × f(risk) d(risk) over Band 3 = α × (1 - prev_avg_risk)
            band2_target_mass = alpha * (1 - prev_avg_risk)
            band3_target_mass = alpha

            # Band 1: Find index where cumulative proportion of population = band1_target_mass
            # This is: ∫ f(risk) d(risk) from risk=1 down to some risk threshold
            cumulative_mass = 0.0
            band1_end_idx = 0
            for i in range(n):
                # Each person contributes 1/n to the density (proportion of population)
                population_contribution = 1.0 / n
                cumulative_mass += population_contribution
                if cumulative_mass >= band1_target_mass:
                    band1_end_idx = i + 1
                    break
            if band1_end_idx == 0 and band1_target_mass > 0:
                band1_end_idx = 1  # At least one person

            # Band 2: Continue from Band 1 end
            target_mass_band1_plus_band2 = band1_target_mass + band2_target_mass
            band2_end_idx = band1_end_idx
            for i in range(band1_end_idx, n):
                population_contribution = 1.0 / n
                cumulative_mass += population_contribution
                if cumulative_mass >= target_mass_band1_plus_band2:
                    band2_end_idx = i + 1
                    break

            # Band 3: Continue from Band 2 end
            target_mass_band1_plus_band2_plus_band3 = band1_target_mass + band2_target_mass + band3_target_mass
            band3_end_idx = band2_end_idx
            for i in range(band2_end_idx, n):
                population_contribution = 1.0 / n
                cumulative_mass += population_contribution
                if cumulative_mass >= target_mass_band1_plus_band2_plus_band3:
                    band3_end_idx = i + 1
                    break

            # Ensure indices are within bounds
            band1_end_idx = min(band1_end_idx, n)
            band2_end_idx = min(band2_end_idx, n)
            band3_end_idx = min(band3_end_idx, n)

            # Compute average risk of Band 3
            if band3_end_idx > band2_end_idx:
                band3_risks = [rows_with_risk[i]["empirical_risk"] for i in range(band2_end_idx, band3_end_idx)]
                current_avg_risk = np.mean(band3_risks) if band3_risks else 0.0
            else:
                current_avg_risk = 0.0

            # Check convergence
            if abs(current_avg_risk - prev_avg_risk) < tolerance:
                break

            prev_avg_risk = current_avg_risk

        # Final band sizes (keep the indices from the last iteration)
        # The indices are already set from the converged iteration above
        avg_risk_band3 = prev_avg_risk

        # Compute integrals: ∫ risk × (1/n) dx for each band (for reporting purposes)
        band1_integral = sum(rows_with_risk[i]["empirical_risk"] / n for i in range(0, band1_end_idx))
        band2_integral = sum(rows_with_risk[i]["empirical_risk"] / n for i in range(band1_end_idx, band2_end_idx))
        band3_integral = sum(rows_with_risk[i]["empirical_risk"] / n for i in range(band2_end_idx, band3_end_idx))
        band4_integral = sum(rows_with_risk[i]["empirical_risk"] / n for i in range(band3_end_idx, n))

        # Population proportions = ∫ f(risk) d(risk) for each band
        # This is the "mass" used for band selection
        band1_pop_prop = band1_end_idx / n
        band2_pop_prop = (band2_end_idx - band1_end_idx) / n
        band3_pop_prop = (band3_end_idx - band2_end_idx) / n
        band4_pop_prop = (n - band3_end_idx) / n

        # Expected negatives in Band 3: ∫ (1 - risk) × f(risk) d(risk) over Band 3
        band3_expected_negatives = sum(
            (1 - rows_with_risk[i]["empirical_risk"]) / n for i in range(band2_end_idx, band3_end_idx)
        )

        # Count true positives in each band
        tp_count = 0

        # Band 1: Treated, empirical predictions
        for i in range(0, band1_end_idx):
            item = rows_with_risk[i]
            if item["true_outcome"]:
                tp_count += 1

        # Band 2: Treated, empirical predictions
        for i in range(band1_end_idx, band2_end_idx):
            item = rows_with_risk[i]
            if item["true_outcome"]:
                tp_count += 1

        # Band 3: Screened, use true outcomes
        for i in range(band2_end_idx, band3_end_idx):
            item = rows_with_risk[i]
            if item["true_outcome"]:
                tp_count += 1

        # Band 4: Untreated, predict 0 (no TPs)
        # (no contribution to tp_count)

        results["alpha_values"].append(alpha)
        # Enforce monotonicity: TP can never decrease as screening budget grows
        tp_count = max(tp_count, results["true_positives"][-1] if results["true_positives"] else 0)
        results["true_positives"].append(tp_count)
        results["band_info"].append(
            {
                "alpha": alpha,
                "band1_integral": band1_integral,
                "band2_integral": band2_integral,
                "band3_integral": band3_integral,
                "band4_integral": band4_integral,
                "band1_pop_prop": band1_pop_prop,
                "band2_pop_prop": band2_pop_prop,
                "band3_pop_prop": band3_pop_prop,
                "band4_pop_prop": band4_pop_prop,
                "band3_expected_negatives": band3_expected_negatives,
                "avg_risk_band3": avg_risk_band3,
                "band1_end_idx": band1_end_idx,
                "band2_end_idx": band2_end_idx,
                "band3_end_idx": band3_end_idx,
            }
        )

    return results


def compute_random_screening_curve(
    rows: list[dict[str, Any]],
    outcome_col: str,
    strata_features: Sequence[str],
    prediction_col: str = "probability",
    beta: float = 0.5,
    alpha_quantiles: Sequence[float] | None = None,
    seed: int = 42,
    use_custom_risk_col: str | None = None,
    simulation: str | tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Compute random screening baseline curve.

    This baseline screens α proportion of the population at random (instead of targeting
    low-risk individuals). It treats:
    1. All screened individuals with Y=1 (true positive outcome)
    2. From unscreened, treats top (β + prop_screened_negatives - prop_screened_positives) by risk

    The intuition: by randomly screening, we identify some negatives and don't waste treatment
    budget on them, allowing us to treat more high-risk unscreened individuals.

    Args:
        rows: List of data rows with features, outcome, and predictions
        outcome_col: Name of outcome column
        strata_features: Features defining strata (used for risk scoring)
        prediction_col: Column name for model predictions
        beta: Treatment budget (proportion who can be treated)
        alpha_quantiles: Screening budget levels to evaluate
        seed: Random seed for reproducible random screening
        use_custom_risk_col: If provided, use this column for risk instead of empirical
        simulation: If provided, generate synthetic data from a Beta distribution instead
            of using real data. Pass a preset name ('uniform', 'bimodal', 'unimodal') or
            a tuple (a, b) of Beta distribution parameters. Uses SIMULATION_SIZE samples.

    Returns:
        Dictionary with screening curves
    """
    if alpha_quantiles is None:
        alpha_quantiles = [beta * i / 49 for i in range(50)]

    # Assign each row its risk (simulation, custom, or empirical)
    rows_with_risk = []

    if simulation is not None:
        # Generate synthetic data from a Beta distribution
        if isinstance(simulation, str):
            if simulation not in RISK_PRESETS:
                raise ValueError(f"Unknown simulation preset '{simulation}'. Choose from {list(RISK_PRESETS.keys())}.")
            preset = RISK_PRESETS[simulation]
        else:
            preset = simulation

        if isinstance(preset, (int, float)):
            risk_scores, outcomes = generate_simulation_data(size=SIMULATION_SIZE, seed=seed, point_mass=float(preset))
        else:
            a, b = preset
            risk_scores, outcomes = generate_simulation_data(a, b, size=SIMULATION_SIZE, seed=seed)

        for i in range(SIMULATION_SIZE):
            rows_with_risk.append(
                {
                    "row": {"_sim_index": i, "_sim_feature": 0, outcome_col: bool(outcomes[i])},
                    "empirical_risk": float(risk_scores[i]),
                    "true_outcome": bool(outcomes[i]),
                    "model_prediction": float(risk_scores[i]),
                }
            )
    elif use_custom_risk_col is not None:
        # Use custom risk column directly
        for row in rows:
            risk = row.get(use_custom_risk_col, 0.5)
            rows_with_risk.append(
                {
                    "row": row,
                    "empirical_risk": risk,
                    "true_outcome": _is_positive_outcome(row.get(outcome_col)),
                    "model_prediction": row.get(prediction_col, 0.5),
                }
            )
    else:
        # Compute empirical P(Y=1|X) for each stratum
        empirical_probs = compute_empirical_probabilities(rows, outcome_col, strata_features)

        for row in rows:
            stratum_key = tuple(row.get(f) for f in strata_features)
            empirical_risk = empirical_probs.get(stratum_key, {}).get("probability", 0.5)

            rows_with_risk.append(
                {
                    "row": row,
                    "empirical_risk": empirical_risk,
                    "true_outcome": _is_positive_outcome(row.get(outcome_col)),
                    "model_prediction": row.get(prediction_col, 0.5),
                }
            )

    total_positive = sum(1 for r in rows_with_risk if r["true_outcome"])
    n = len(rows_with_risk)

    # Results storage
    results = {
        "beta": beta,
        "alpha_values": [],
        "true_positives": [],
        "total_positive": total_positive,
        "total_samples": n,
    }

    # Set random seed for reproducibility — use a single permutation so that
    # screened sets are nested (larger α always includes the smaller α set).
    rng = np.random.RandomState(seed)
    random_order = rng.permutation(n)

    for alpha in alpha_quantiles:
        assert alpha <= beta, f"Screening budget α={alpha} exceeds treatment budget β={beta}"
        # Screen α proportion uniformly at random
        n_screen = min(int(alpha * n), n)
        n_treat = int(beta * n)

        screened_indices = set(random_order[:n_screen])

        # Identify screened positives (gamma mass)
        screened_positive_indices = {idx for idx in screened_indices if rows_with_risk[idx]["true_outcome"]}
        gamma_count = len(screened_positive_indices)

        # Treat screened positives up to budget
        tp_from_screening = min(gamma_count, n_treat)
        remaining_budget = max(0, n_treat - tp_from_screening)

        # Pool for risk-based treatment: everyone except screened positives
        pool = [(idx, rows_with_risk[idx]) for idx in range(n) if idx not in screened_positive_indices]
        pool.sort(key=lambda x: x[1]["empirical_risk"], reverse=True)

        # Treat top (β - γ) mass by risk score
        n_treat_by_risk = min(remaining_budget, len(pool))
        tp_from_risk = sum(1 for i in range(n_treat_by_risk) if pool[i][1]["true_outcome"])

        tp_count = tp_from_screening + tp_from_risk
        results["alpha_values"].append(alpha)
        results["true_positives"].append(tp_count)

    return results


def compute_intuitive_optimal_curve(
    rows: list[dict[str, Any]],
    outcome_col: str,
    strata_features: Sequence[str],
    prediction_col: str = "probability",
    beta: float = 0.5,
    alpha_quantiles: Sequence[float] | None = None,
    seed: int | None = None,
    use_custom_risk_col: str | None = None,
    simulation: str | tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Compute intuitive-optimal screening curve.

    Algorithm (all bands are adjacent slices of the risk-sorted population):
      1. Band A: treat the top (β − α) mass by risk (highest risk, no screening).
      2. Band B: screen the next α mass.  Let γ ≤ α be the mass of screened
         individuals with Y=0.  Screened Y=1 are treated; screened Y=0 are not.
      3. Band C: treat the next γ mass below the screened band (replaces the
         screened negatives, preserving total treatment budget = β).

    Args:
        rows: List of data rows (ignored when *simulation* is set).
        outcome_col: Name of outcome column.
        strata_features: Features defining strata.
        prediction_col: Column name for model predictions.
        beta: Treatment budget (proportion who can be treated).
        alpha_quantiles: Screening budget levels to evaluate.
        seed: Random seed for simulation mode.
        use_custom_risk_col: Use this column for risk instead of empirical.
        simulation: Preset name or (a, b) Beta parameters for synthetic data.

    Returns:
        Dictionary with alpha_values, true_positives, total_positive, total_samples.
    """
    if alpha_quantiles is None:
        alpha_quantiles = [beta * i / 49 for i in range(50)]

    # --- Build rows_with_risk (same logic as compute_optimal_screening_curve) ---
    rows_with_risk = []

    if simulation is not None:
        if isinstance(simulation, str):
            if simulation not in RISK_PRESETS:
                raise ValueError(f"Unknown simulation preset '{simulation}'. Choose from {list(RISK_PRESETS.keys())}.")
            preset = RISK_PRESETS[simulation]
        else:
            preset = simulation

        if isinstance(preset, (int, float)):
            risk_scores, outcomes = generate_simulation_data(size=SIMULATION_SIZE, seed=seed, point_mass=float(preset))
        else:
            a, b = preset
            risk_scores, outcomes = generate_simulation_data(a, b, size=SIMULATION_SIZE, seed=seed)

        for i in range(SIMULATION_SIZE):
            rows_with_risk.append(
                {
                    "row": {"_sim_index": i, "_sim_feature": 0, outcome_col: bool(outcomes[i])},
                    "empirical_risk": float(risk_scores[i]),
                    "true_outcome": bool(outcomes[i]),
                    "model_prediction": float(risk_scores[i]),
                }
            )
    elif use_custom_risk_col is not None:
        for row in rows:
            risk = row.get(use_custom_risk_col, 0.5)
            rows_with_risk.append(
                {
                    "row": row,
                    "empirical_risk": risk,
                    "true_outcome": _is_positive_outcome(row.get(outcome_col)),
                    "model_prediction": row.get(prediction_col, 0.5),
                }
            )
    else:
        empirical_probs = compute_empirical_probabilities(rows, outcome_col, strata_features)
        for row in rows:
            stratum_key = tuple(row.get(f) for f in strata_features)
            empirical_risk = empirical_probs.get(stratum_key, {}).get("probability", 0.5)
            rows_with_risk.append(
                {
                    "row": row,
                    "empirical_risk": empirical_risk,
                    "true_outcome": _is_positive_outcome(row.get(outcome_col)),
                    "model_prediction": row.get(prediction_col, 0.5),
                }
            )

    # Sort by risk (highest to lowest)
    rows_with_risk.sort(key=lambda x: x["empirical_risk"], reverse=True)

    total_positive = sum(1 for r in rows_with_risk if r["true_outcome"])
    n = len(rows_with_risk)

    results = {
        "beta": beta,
        "alpha_values": [],
        "true_positives": [],
        "total_positive": total_positive,
        "total_samples": n,
    }

    for alpha in alpha_quantiles:
        assert alpha <= beta, f"Screening budget α={alpha} exceeds treatment budget β={beta}"

        band_a_end = int((beta - alpha) * n)
        band_b_end = band_a_end + int(alpha * n)

        # Band A: treated by risk
        tp_band_a = 0
        for i in range(band_a_end):
            item = rows_with_risk[i]
            if item["true_outcome"]:
                tp_band_a += 1

        # Band B: screened — Y=1 treated, Y=0 not treated
        tp_band_b = 0
        gamma_count = 0  # number of screened with Y=0
        for i in range(band_a_end, band_b_end):
            item = rows_with_risk[i]
            if item["true_outcome"]:
                tp_band_b += 1
            else:
                gamma_count += 1

        # Band C: next gamma_count individuals treated by risk
        band_c_end = min(band_b_end + gamma_count, n)
        tp_band_c = 0
        for i in range(band_b_end, band_c_end):
            item = rows_with_risk[i]
            if item["true_outcome"]:
                tp_band_c += 1

        tp_count = tp_band_a + tp_band_b + tp_band_c
        results["alpha_values"].append(alpha)
        results["true_positives"].append(tp_count)

    return results
