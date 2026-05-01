#!/usr/bin/env python
"""Simulate utility gap vs Beta(t,t) distribution parameter at a fixed alpha level and save to JSON.

Sweeps the symmetric Beta(t,t) parameter t log-uniformly from near-bimodal (t→0) to
near-point-mass at 1/2 (t→∞). For each t, computes:

    utility gap = U_optimal(alpha) - U_optimal(0)

where U is the fraction of true positives identified (%).
Does NOT produce plots — run plot_figures.py to generate the figure from the saved artifact.

Usage:
    # Paper figure 4 — α=25%, β=35%, 3 sims per point:
    python scripts/simulate_utility_gap.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, "src")
from optimal_screening.analysis import compute_optimal_screening_curve


def main():
    parser = argparse.ArgumentParser(description="Utility gap vs Beta(t,t) parameter at fixed alpha")
    parser.add_argument("--alpha", type=float, default=0.35, help="Fixed screening budget α (default: 0.25)")
    parser.add_argument("--beta", type=float, default=0.35, help="Treatment budget β (default: 0.35)")
    parser.add_argument("--n-sims", type=int, default=10, help="Simulations per parameter value (default: 3)")
    parser.add_argument("--n-params", type=int, default=50, help="Number of t values on the log grid (default: 50)")
    parser.add_argument("--t-min", type=float, default=0.02, help="Minimum Beta parameter (default: 0.02)")
    parser.add_argument("--t-max", type=float, default=200.0, help="Maximum Beta parameter (default: 200.0)")
    parser.add_argument("--output", type=str, default="runs/utility_gap_vs_dist_param.json", help="Output JSON path")
    args = parser.parse_args()

    if args.alpha > args.beta:
        raise ValueError(f"alpha={args.alpha} must be <= beta={args.beta}")

    alpha = args.alpha
    beta = args.beta
    n_sims = args.n_sims

    t_values = np.logspace(np.log10(args.t_min), np.log10(args.t_max), args.n_params)

    print("Utility gap vs Beta(t,t) parameter")
    print(f"  α={alpha:.0%}, β={beta:.0%}, {n_sims} sims per point, {len(t_values)} t values")
    print(f"  t range: [{args.t_min}, {args.t_max}] (log scale)")
    print()

    gaps_mean = []
    gaps_std = []

    for t_idx, t in enumerate(t_values):
        gap_runs = []
        for sim_idx in range(n_sims):
            result = compute_optimal_screening_curve(
                rows=[],
                outcome_col="_outcome",
                strata_features=[],
                beta=beta,
                alpha_quantiles=[0.0, alpha],
                seed=sim_idx,
                simulation=(t, t),
            )
            total_pos = result["total_positive"]
            if total_pos > 0:
                u_no_screen = result["true_positives"][0] / total_pos * 100
                u_screen = result["true_positives"][1] / total_pos * 100
                gap_runs.append(u_screen - u_no_screen)
            else:
                gap_runs.append(0.0)

        gaps_mean.append(float(np.mean(gap_runs)))
        gaps_std.append(float(np.std(gap_runs)))

        if (t_idx + 1) % 10 == 0 or t_idx == len(t_values) - 1:
            print(f"  [{t_idx + 1:3d}/{len(t_values)}]  t={t:.4f}  gap={gaps_mean[-1]:.2f} ± {gaps_std[-1]:.2f} pp")

    # --- Save JSON artifact ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    json_data = {
        "meta": {
            "alpha": alpha,
            "beta": beta,
            "n_sims": n_sims,
            "n_params": args.n_params,
            "t_min": args.t_min,
            "t_max": args.t_max,
        },
        "t_values": t_values.tolist(),
        "gaps_mean": gaps_mean,
        "gaps_std": gaps_std,
    }

    output_path.write_text(json.dumps(json_data, indent=2))
    print(f"\nData saved to {output_path}")


if __name__ == "__main__":
    main()
