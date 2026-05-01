#!/usr/bin/env python
"""Simulate optimal screening curves across Beta distribution risk presets and save results to JSON.

For each preset (uniform, bimodal, unimodal, delta_half), runs n_sims independent simulations
and records per-simulation TP and precision arrays for optimal, intuitive, and random screening.
Does NOT produce plots — run plot_figures.py to generate figures from the saved artifact.

Usage:
    # Paper figures 3 & 4 — β=35%, 10 sims per preset:
    python scripts/simulate_distributions.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, "src")
from optimal_screening.analysis import (
    RISK_PRESETS,
    compute_intuitive_optimal_curve,
    compute_optimal_screening_curve,
    compute_random_screening_curve,
)


def main():
    parser = argparse.ArgumentParser(description="Simulate screening curves for Beta distribution presets")
    parser.add_argument("--beta", type=float, default=0.35, help="Treatment budget (default: 0.35)")
    parser.add_argument("--n-sims", type=int, default=10, help="Number of simulations per preset (default: 10)")
    parser.add_argument("--output", type=str, default="runs/simulations.json", help="Output JSON path")
    args = parser.parse_args()

    beta = args.beta
    n_sims = args.n_sims

    print(f"Running {n_sims} simulations per preset with β={beta:.1%}...")
    print(f"Presets: {list(RISK_PRESETS.keys())}")
    print()

    all_preset_results = {}

    for preset in RISK_PRESETS:
        params = RISK_PRESETS[preset]
        if isinstance(params, tuple):
            print(f"  [{preset}] Beta({params[0]}, {params[1]})")
        else:
            print(f"  [{preset}] Point mass at {params}")

        optimal_tp_runs = []
        intuitive_tp_runs = []
        random_tp_runs = []
        optimal_prec_runs = []
        intuitive_prec_runs = []
        random_prec_runs = []
        alpha_values = None

        for sim_idx in range(n_sims):
            optimal = compute_optimal_screening_curve(
                rows=[],
                outcome_col="_outcome",
                strata_features=[],
                beta=beta,
                seed=sim_idx,
                simulation=preset,
            )
            intuitive = compute_intuitive_optimal_curve(
                rows=[],
                outcome_col="_outcome",
                strata_features=[],
                beta=beta,
                seed=sim_idx,
                simulation=preset,
            )
            random_baseline = compute_random_screening_curve(
                rows=[],
                outcome_col="_outcome",
                strata_features=[],
                beta=beta,
                seed=sim_idx,
                simulation=preset,
            )

            if alpha_values is None:
                alpha_values = optimal["alpha_values"]

            total_pos_opt = optimal["total_positive"]
            total_pos_int = intuitive["total_positive"]
            total_pos_rnd = random_baseline["total_positive"]
            treated_opt = beta * optimal["total_samples"]
            treated_int = beta * intuitive["total_samples"]
            treated_rnd = beta * random_baseline["total_samples"]

            optimal_tp_runs.append(
                [tp / total_pos_opt * 100 for tp in optimal["true_positives"]]
                if total_pos_opt > 0
                else [0.0] * len(optimal["true_positives"])
            )
            intuitive_tp_runs.append(
                [tp / total_pos_int * 100 for tp in intuitive["true_positives"]]
                if total_pos_int > 0
                else [0.0] * len(intuitive["true_positives"])
            )
            random_tp_runs.append(
                [tp / total_pos_rnd * 100 for tp in random_baseline["true_positives"]]
                if total_pos_rnd > 0
                else [0.0] * len(random_baseline["true_positives"])
            )
            optimal_prec_runs.append(
                [tp / treated_opt * 100 if treated_opt > 0 else 0.0 for tp in optimal["true_positives"]]
            )
            intuitive_prec_runs.append(
                [tp / treated_int * 100 if treated_int > 0 else 0.0 for tp in intuitive["true_positives"]]
            )
            random_prec_runs.append(
                [tp / treated_rnd * 100 if treated_rnd > 0 else 0.0 for tp in random_baseline["true_positives"]]
            )

            if (sim_idx + 1) % 25 == 0:
                print(f"    Completed {sim_idx + 1}/{n_sims}")

        all_preset_results[preset] = {
            "alpha_values": alpha_values,
            "optimal_tp": np.array(optimal_tp_runs),
            "intuitive_tp": np.array(intuitive_tp_runs),
            "random_tp": np.array(random_tp_runs),
            "optimal_prec": np.array(optimal_prec_runs),
            "intuitive_prec": np.array(intuitive_prec_runs),
            "random_prec": np.array(random_prec_runs),
        }

    # --- Save JSON artifact ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    series_keys = ("optimal_tp", "intuitive_tp", "random_tp", "optimal_prec", "intuitive_prec", "random_prec")

    json_data: dict = {
        "meta": {
            "beta": beta,
            "n_sims": n_sims,
            "presets": {k: list(v) if isinstance(v, tuple) else v for k, v in RISK_PRESETS.items()},
            "alpha_values": [float(a) for a in all_preset_results[next(iter(all_preset_results))]["alpha_values"]],
        },
        "presets": {
            preset: (
                {s: data[s].tolist() for s in series_keys}
                | {f"{s}_mean": data[s].mean(axis=0).tolist() for s in series_keys}
                | {f"{s}_std": data[s].std(axis=0).tolist() for s in series_keys}
            )
            for preset, data in all_preset_results.items()
        },
    }

    output_path.write_text(json.dumps(json_data, indent=2))
    print(f"\nData saved to {output_path}")


if __name__ == "__main__":
    main()
