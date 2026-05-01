#!/usr/bin/env python
"""Generate all paper figures from pre-computed JSON artifacts.

Reads artifacts produced by the simulation and model scripts and saves one PNG per figure.

Artifact sources:
  - runs/simulations.json             → produced by simulate_distributions.py
  - runs/utility_gap_vs_dist_param.json → produced by simulate_utility_gap.py
  - runs/screening_tabular_*.json     → produced by run_tabular_models.py (one per dataset)

Output PNGs are saved alongside their source JSON (same path, .png extension),
except for the simulation plots which share a single JSON but produce multiple PNGs.

Usage:
    # 1. Generate JSON artifacts (only needed when rerunning experiments):
    python scripts/simulate_distributions.py
    python scripts/simulate_utility_gap.py
    python scripts/run_tabular_models.py --csv data/landmines_raw.csv --outcome mines_outcome \
        --strata Municipio --beta 0.1 --test-municipio GRANADA \
        --output runs/screening_tabular_landmines_granada_test.json
    python scripts/run_tabular_models.py --csv data/acs_income_2018.csv --outcome "PINCP > 50k" \
        --strata AGEP COW SCHL MAR OCCP POBP RELP WKHP SEX RAC1P --beta 0.3 --strata-only \
        --test-size 0.5 --output runs/screening_tabular_acs2018_all_strata_50pct.json

    # 2. Generate all plots from existing artifacts:
    python scripts/plot_figures.py
    python scripts/plot_figures.py --runs-dir runs/
"""

import argparse
import json
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta as beta_dist


# ---------------------------------------------------------------------------
# Shared visual style
# ---------------------------------------------------------------------------

# Style for each Beta distribution preset (used in simulation plots)
PRESET_STYLES = {
    "uniform": {"color": "blue", "color_pastel": "#a0c4ff", "marker": "o"},
    "bimodal": {"color": "orange", "color_pastel": "#ffd6a5", "marker": "s"},
    "unimodal": {"color": "green", "color_pastel": "#94cc86", "marker": "^"},
    "delta_half": {"color": "red", "color_pastel": "#ffadad", "marker": "D"},
}

# Style for each ML model (used in tabular screening plots)
MODEL_STYLES: dict[str, dict[str, str]] = {
    "Logistic Regression": {"color": "tab:blue", "marker": "o"},
    "Random Forest": {"color": "tab:green", "marker": "s"},
    "XGBoost": {"color": "tab:red", "marker": "^"},
    "Random": {"color": "gray", "marker": "x"},
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_preset(preset: str, presets_meta: dict) -> str:
    """Return a short parameter label for a preset, e.g. '(1.0, 1.0)' or 'δ(0.5)'."""
    params = presets_meta[preset]
    return f"({params[0]}, {params[1]})" if isinstance(params, list) else f"δ({params})"


def _style_ax(ax, title: str, beta: float, y_values=None, legend=True, ylabel: str = "True Positives Identified (%)"):
    ax.set_xlabel("Screening Budget α (%)", fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    _ = title
    if legend:
        ax.legend(fontsize=12, loc="best")
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, beta * 100)
    if y_values:
        y_min, y_max = min(y_values), max(y_values)
        margin = (y_max - y_min) * 0.15
        ax.set_ylim(max(0, y_min - margin), min(105, y_max + margin))
    else:
        ax.set_ylim(0, 105)


def _compact_legend(ax, presets, label):
    dist_handles = [
        mlines.Line2D(
            [],
            [],
            color=PRESET_STYLES[p]["color"],
            marker=PRESET_STYLES[p]["marker"],
            markersize=5,
            linewidth=2.5,
            label=p.capitalize(),
        )
        for p in presets
    ]
    method_handles = [
        mlines.Line2D([], [], color="gray", linewidth=2.5, linestyle="-", label="Optimal"),
        mlines.Line2D([], [], color="lightgray", linewidth=2.5, linestyle=":", label=label),
    ]

    dist_title = mlines.Line2D([], [], linestyle="none")
    screen_title = mlines.Line2D([], [], linestyle="none")
    empty = mlines.Line2D([], [], linestyle="none")
    pad = 2
    handles = [dist_title] + dist_handles + [screen_title] + method_handles + [empty] * pad

    labels = (
        ["Distribution"]
        + [h.get_label() for h in dist_handles]
        + ["Screening"]
        + [h.get_label() for h in method_handles]
        + [""] * pad
    )

    leg = ax.legend(
        handles=handles,
        labels=labels,
        loc="lower right",
        frameon=True,
        fontsize=14,
        title_fontsize=15,
        ncol=2,
        columnspacing=1.5,
        handlelength=2.5,
    )

    # Make section titles bold
    for text in leg.get_texts():
        if text.get_text() in ["Distribution", "Screening"]:
            text.set_weight("bold")
    leg._legend_box.align = "center"


def _band(ax, x, mean, std, color, alpha: float = 0.4):
    """Draw a ±1 std dev shaded band around a mean line."""
    ax.fill_between(x, mean - std, mean + std, alpha=alpha, color=color, linewidth=0)


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Simulation plots  (source: runs/simulations.json)
# ---------------------------------------------------------------------------


def plot_simulations(json_path: Path) -> None:
    """Generate all simulation comparison plots from simulate_distributions.py output."""
    data = json.loads(json_path.read_text())
    meta = data["meta"]
    beta = meta["beta"]
    n_sims = meta["n_sims"]
    presets_meta = meta["presets"]
    alpha_pct = [a * 100 for a in meta["alpha_values"]]
    presets = data["presets"]

    runs_dir = json_path.parent

    # --- Figure: Risk score distributions (Beta PDFs, no simulation data needed) ---
    # Artifact: runs/simulations.json (meta.presets contains Beta parameters)
    print("  [risk_distributions] Beta PDF shapes for each preset")
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.linspace(0, 1, 500)
    for preset, params in presets_meta.items():
        style = PRESET_STYLES[preset]
        if isinstance(params, list):
            a, b = params
            ax.plot(
                x,
                beta_dist.pdf(x, a, b),
                linewidth=2.5,
                color=style["color"],
                label=f"{preset.capitalize()} — Beta({a}, {b})",
            )
        else:
            ax.axvline(x=params, color=style["color"], linewidth=2.5, label=f"{preset.capitalize()} — δ({params})")
    ax.set_xlabel("Risk score", fontsize=18)
    ax.set_ylabel("Density", fontsize=18)
    # ax.set_title("Risk Score Distributions", fontsize=15, fontweight="bold")
    ax.legend(fontsize=15)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 5)
    _save(fig, runs_dir / "risk_distributions.png")

    # --- Figure: Optimal vs Intuitive screening (TP recall) ---
    # Artifact fields used: optimal_tp_mean, optimal_tp_std, intuitive_tp_mean, intuitive_tp_std
    print("  [optimal_vs_intuitive] Mean TP recall ± 1 std — optimal vs intuitive")
    fig, ax = plt.subplots(figsize=(10, 6))
    y_vals = []
    for preset, pdata in presets.items():
        style = PRESET_STYLES[preset]
        int_mean = np.array(pdata["intuitive_tp_mean"])
        int_std = np.array(pdata["intuitive_tp_std"])
        y_vals.extend([(int_mean - int_std).min(), (int_mean + int_std).max()])
        _band(ax, alpha_pct, int_mean, int_std, style["color_pastel"])
        ax.plot(
            alpha_pct,
            int_mean,
            marker=style["marker"],
            linewidth=2.5,
            markersize=4,
            linestyle=":",
            color=style["color_pastel"],
            label=f"Intuitive – {preset}",
            zorder=1,
        )
    for preset, pdata in presets.items():
        style = PRESET_STYLES[preset]
        opt_mean = np.array(pdata["optimal_tp_mean"])
        opt_std = np.array(pdata["optimal_tp_std"])
        label = _format_preset(preset, presets_meta)
        y_vals.extend([(opt_mean - opt_std).min(), (opt_mean + opt_std).max()])
        _band(ax, alpha_pct, opt_mean, opt_std, style["color_pastel"])
        ax.plot(
            alpha_pct,
            opt_mean,
            marker=style["marker"],
            linewidth=2.5,
            markersize=5,
            color=style["color"],
            label=f"Optimal – {preset} {label}",
            zorder=2,
        )
    _style_ax(ax, f"Optimal vs Heuristic Screening ({n_sims} sims, β={beta:.0%})", beta, y_vals, legend=False)
    _compact_legend(ax, presets, label="Heuristic Top")
    _save(fig, runs_dir / "recall_optimal_vs_heuristic.png")

    # --- Figure: Optimal vs Random screening (TP recall) ---
    # Artifact fields used: optimal_tp_mean/std, random_tp_mean/std
    print("  [optimal_vs_random] Mean TP recall ± 1 std — optimal vs random")
    fig, ax = plt.subplots(figsize=(10, 6))
    y_vals = []
    for preset, pdata in presets.items():
        style = PRESET_STYLES[preset]
        rnd_mean = np.array(pdata["random_tp_mean"])
        rnd_std = np.array(pdata["random_tp_std"])
        y_vals.extend([(rnd_mean - rnd_std).min(), (rnd_mean + rnd_std).max()])
        _band(ax, alpha_pct, rnd_mean, rnd_std, style["color_pastel"])
        ax.plot(
            alpha_pct,
            rnd_mean,
            marker=style["marker"],
            linewidth=2.5,
            markersize=4,
            linestyle=":",
            color=style["color_pastel"],
            label=f"Random – {preset}",
            zorder=1,
        )
    for preset, pdata in presets.items():
        style = PRESET_STYLES[preset]
        opt_mean = np.array(pdata["optimal_tp_mean"])
        opt_std = np.array(pdata["optimal_tp_std"])
        label = _format_preset(preset, presets_meta)
        y_vals.extend([(opt_mean - opt_std).min(), (opt_mean + opt_std).max()])
        _band(ax, alpha_pct, opt_mean, opt_std, style["color"])
        ax.plot(
            alpha_pct,
            opt_mean,
            marker=style["marker"],
            linewidth=2.5,
            markersize=5,
            color=style["color"],
            label=f"Optimal – {preset} {label}",
            zorder=2,
        )
    _style_ax(ax, f"Optimal vs Random Screening ({n_sims} sims, β={beta:.0%})", beta, y_vals, legend=False)
    _compact_legend(ax, presets, label="Random")
    _save(fig, runs_dir / "recall_optimal_vs_random.png")

    # --- Figure: Optimal vs No Screening (paper Figure 3) ---
    # Artifact fields used: optimal_tp_mean/std (no-screening baseline = value at α=0)
    print("  [optimal_vs_no_screening] Mean TP recall ± 1 std — optimal vs no-screening baseline (paper Fig 3)")
    fig, ax = plt.subplots(figsize=(10, 6))
    y_vals = []
    for preset, pdata in presets.items():
        style = PRESET_STYLES[preset]
        no_screen_mean = np.array(pdata["optimal_tp_mean"])[0]
        no_screen_std = np.array(pdata["optimal_tp_std"])[0]
        y_vals.extend([no_screen_mean - no_screen_std, no_screen_mean + no_screen_std])
        _band(ax, alpha_pct, no_screen_mean, no_screen_std, style["color_pastel"])
        ax.plot(
            alpha_pct,
            [no_screen_mean] * len(alpha_pct),
            marker=style["marker"],
            linewidth=2.5,
            markersize=4,
            linestyle=":",
            color=style["color_pastel"],
            label=f"No screening – {preset} ({no_screen_mean:.1f}%)",
            zorder=1,
        )
    for preset, pdata in presets.items():
        style = PRESET_STYLES[preset]
        opt_mean = np.array(pdata["optimal_tp_mean"])
        opt_std = np.array(pdata["optimal_tp_std"])
        label = _format_preset(preset, presets_meta)
        y_vals.extend([(opt_mean - opt_std).min(), (opt_mean + opt_std).max()])
        _band(ax, alpha_pct, opt_mean, opt_std, style["color_pastel"])
        ax.plot(
            alpha_pct,
            opt_mean,
            marker=style["marker"],
            linewidth=2.5,
            markersize=5,
            color=style["color"],
            label=f"Optimal – {preset} {label}",
            zorder=2,
        )
    _style_ax(ax, f"Optimal vs No Screening ({n_sims} sims, β={beta:.0%})", beta, y_vals, legend=False)
    _compact_legend(ax, presets, label="No Screening")
    _save(fig, runs_dir / "recall_optimal_vs_no_screening.png")

    prec_ylabel = "Allocation Precision (%)"

    # --- Figure: Precision — Optimal vs Intuitive ---
    # Artifact fields used: optimal_prec_mean/std, intuitive_prec_mean/std
    print("  [precision_optimal_vs_heuristic] Precision ± 1 std — optimal vs heuristic")
    fig, ax = plt.subplots(figsize=(10, 6))
    y_vals = []
    for preset, pdata in presets.items():
        style = PRESET_STYLES[preset]
        int_mean = np.array(pdata["intuitive_prec_mean"])
        int_std = np.array(pdata["intuitive_prec_std"])
        y_vals.extend([(int_mean - int_std).min(), (int_mean + int_std).max()])
        _band(ax, alpha_pct, int_mean, int_std, style["color_pastel"])
        ax.plot(
            alpha_pct,
            int_mean,
            marker=style["marker"],
            linewidth=2.5,
            markersize=4,
            linestyle=":",
            color=style["color_pastel"],
            label=f"Heuristic – {preset}",
            zorder=1,
        )
    for preset, pdata in presets.items():
        style = PRESET_STYLES[preset]
        opt_mean = np.array(pdata["optimal_prec_mean"])
        opt_std = np.array(pdata["optimal_prec_std"])
        label = _format_preset(preset, presets_meta)
        y_vals.extend([(opt_mean - opt_std).min(), (opt_mean + opt_std).max()])
        _band(ax, alpha_pct, opt_mean, opt_std, style["color"])
        ax.plot(
            alpha_pct,
            opt_mean,
            marker=style["marker"],
            linewidth=2.5,
            markersize=5,
            color=style["color"],
            label=f"Optimal – {preset} {label}",
            zorder=2,
        )
    _style_ax(
        ax,
        f"Precision: Optimal vs Heuristic ({n_sims} sims, β={beta:.0%})",
        beta,
        y_vals,
        ylabel=prec_ylabel,
        legend=False,
    )
    _compact_legend(ax, presets, label="Heuristic Top")
    _save(fig, runs_dir / "precision_optimal_vs_heuristic.png")

    # --- Figure: Precision — Optimal vs Random ---
    # Artifact fields used: optimal_prec_mean/std, random_prec_mean/std
    print("  [precision_optimal_vs_random] Precision ± 1 std — optimal vs random")
    fig, ax = plt.subplots(figsize=(10, 6))
    y_vals = []
    for preset, pdata in presets.items():
        style = PRESET_STYLES[preset]
        rnd_mean = np.array(pdata["random_prec_mean"])
        rnd_std = np.array(pdata["random_prec_std"])
        y_vals.extend([(rnd_mean - rnd_std).min(), (rnd_mean + rnd_std).max()])
        _band(ax, alpha_pct, rnd_mean, rnd_std, style["color_pastel"])
        ax.plot(
            alpha_pct,
            rnd_mean,
            marker=style["marker"],
            linewidth=2.5,
            markersize=4,
            linestyle=":",
            color=style["color_pastel"],
            label=f"Random – {preset}",
            zorder=1,
        )
    for preset, pdata in presets.items():
        style = PRESET_STYLES[preset]
        opt_mean = np.array(pdata["optimal_prec_mean"])
        opt_std = np.array(pdata["optimal_prec_std"])
        label = _format_preset(preset, presets_meta)
        y_vals.extend([(opt_mean - opt_std).min(), (opt_mean + opt_std).max()])
        _band(ax, alpha_pct, opt_mean, opt_std, style["color"])
        ax.plot(
            alpha_pct,
            opt_mean,
            marker=style["marker"],
            linewidth=2.5,
            markersize=5,
            color=style["color"],
            label=f"Optimal – {preset} {label}",
            zorder=2,
        )
    _style_ax(
        ax,
        f"Precision: Optimal vs Random ({n_sims} sims, β={beta:.0%})",
        beta,
        y_vals,
        ylabel=prec_ylabel,
        legend=False,
    )
    _compact_legend(ax, presets, label="Random")
    _save(fig, runs_dir / "precision_optimal_vs_random.png")

    # --- Figure: Precision — Optimal vs No Screening ---
    # Artifact fields used: optimal_prec_mean/std (no-screening baseline = value at α=0)
    print("  [precision_optimal_vs_no_screening] Precision ± 1 std — optimal vs no-screening baseline")
    fig, ax = plt.subplots(figsize=(10, 6))
    y_vals = []
    for preset, pdata in presets.items():
        style = PRESET_STYLES[preset]
        no_screen_mean = np.array(pdata["optimal_prec_mean"])[0]
        no_screen_std = np.array(pdata["optimal_prec_std"])[0]
        y_vals.extend([no_screen_mean - no_screen_std, no_screen_mean + no_screen_std])
        _band(ax, alpha_pct, no_screen_mean, no_screen_std, style["color_pastel"])
        ax.plot(
            alpha_pct,
            [no_screen_mean] * len(alpha_pct),
            marker=style["marker"],
            linewidth=2.5,
            markersize=4,
            linestyle=":",
            color=style["color_pastel"],
            label=f"No screening – {preset} ({no_screen_mean:.1f}%)",
            zorder=1,
        )
    for preset, pdata in presets.items():
        style = PRESET_STYLES[preset]
        opt_mean = np.array(pdata["optimal_prec_mean"])
        opt_std = np.array(pdata["optimal_prec_std"])
        label = _format_preset(preset, presets_meta)
        y_vals.extend([(opt_mean - opt_std).min(), (opt_mean + opt_std).max()])
        _band(ax, alpha_pct, opt_mean, opt_std, style["color"])
        ax.plot(
            alpha_pct, opt_mean, marker=style["marker"], linewidth=2.5, markersize=5, color=style["color"], zorder=2
        )
    _style_ax(
        ax,
        f"Precision: Optimal vs No Screening ({n_sims} sims, β={beta:.0%})",
        beta,
        y_vals,
        ylabel=prec_ylabel,
        legend=False,
    )
    _compact_legend(ax, presets, label="No Screening")
    _save(fig, runs_dir / "precision_optimal_vs_no_screening.png")

    # --- Figure: Utility gap — Optimal vs No Screening ---
    # Artifact fields used: optimal_tp (raw runs, to compute gap std correctly)
    # gap_i = opt_tp_i - opt_tp_i[α=0], so variance propagates from per-run differences
    print("  [utility_gap_optimal_vs_no_screening] Utility gap (TP gain over no-screening) ± 1 std")
    fig, ax = plt.subplots(figsize=(10, 6))
    y_vals = []
    for preset, pdata in presets.items():
        style = PRESET_STYLES[preset]
        opt_runs = np.array(pdata["optimal_tp"])  # shape: (n_sims, n_alphas)
        gap_runs = opt_runs - opt_runs[:, 0:1]  # subtract no-screening value per run
        gap_mean = gap_runs.mean(axis=0)
        gap_std = gap_runs.std(axis=0)
        label = _format_preset(preset, presets_meta)
        y_vals.extend([(gap_mean - gap_std).min(), (gap_mean + gap_std).max()])
        _band(ax, alpha_pct, gap_mean, gap_std, style["color"])
        ax.plot(
            alpha_pct,
            gap_mean,
            marker=style["marker"],
            linewidth=2.5,
            markersize=5,
            color=style["color"],
            label=f"{preset} {label}",
        )
    ax.set_xlabel("Screening Budget α (%)", fontsize=13)
    ax.set_ylabel("Utility Gap (percentage points)", fontsize=13)
    ax.set_title(f"Utility Gap: Optimal vs No Screening ({n_sims} sims, β={beta:.0%})", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, beta * 100)
    ax.set_ylim(bottom=0)
    _save(fig, runs_dir / "utility_gap_optimal_vs_no_screening.png")


# ---------------------------------------------------------------------------
# Utility gap vs Beta parameter  (source: runs/utility_gap_vs_dist_param.json)
# ---------------------------------------------------------------------------


def plot_utility_gap_vs_dist_param(json_path: Path) -> None:
    """Generate utility gap vs Beta(t,t) parameter plot (paper Figure 4).

    Artifact fields used: t_values, gaps_mean, gaps_std, meta (alpha, beta, n_sims, t_min, t_max)
    """
    data = json.loads(json_path.read_text())
    meta = data["meta"]
    beta = 35
    prevalence = 50

    t_values = np.array(data["t_values"])
    gaps_mean = np.array(data["gaps_mean"]) * prevalence / beta
    gaps_std = np.array(data["gaps_std"]) * prevalence / beta

    print("  [utility_gap_vs_dist_param] Utility gap vs Beta(t,t) parameter (paper Fig 4)")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Mean curve with ±1 std band
    ax.plot(t_values, gaps_mean, linewidth=3, color="blue", marker="o", markersize=4, zorder=1)
    ax.fill_between(t_values, gaps_mean - gaps_std, gaps_mean + gaps_std, alpha=0.4, color="steelblue", zorder=1)

    # Reference vertical lines for the named simulation presets
    preset_refs = [
        (0.1, "Bimodal", "tab:orange"),
        (1.0, "Uniform", "tab:blue"),
        (10.0, "Unimodal", "tab:green"),
    ]
    for t_ref, label, color in preset_refs:
        ax.axvline(x=t_ref, color=color, linestyle="--", linewidth=3, alpha=1, label=label)

    ax.set_xscale("log")
    ax.set_xlabel("Beta(t,t) distribution parameter t", fontsize=22)
    ax.set_ylabel("Utility Gap (percentage points)", fontsize=22)
    ax.tick_params(axis="both", labelsize=18)
    # ax.set_title(
    #    f"Utility Gap: Optimal vs No Screening\n"
    #    f"α={meta['alpha']:.0%}, β={meta['beta']:.0%}, "
    #    f"{meta['n_sims']} sim{'s' if meta['n_sims'] > 1 else ''} per point",
    #    fontsize=14, fontweight="bold",
    # )
    leg = ax.legend(loc="lower right", fontsize=18)
    leg.get_title().set_weight("bold")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(meta["t_min"], meta["t_max"])
    ax.set_ylim(bottom=0)

    ax.text(meta["t_min"] * 1.3, ax.get_ylim()[1] * 0.9, "← bimodal\n(0 / 1 risk)", fontsize=16, va="top", color="k")
    ax.text(
        meta["t_max"] * 0.7,
        ax.get_ylim()[1] * 0.9,
        "point mass\nat 0.5 →",
        fontsize=16,
        va="top",
        ha="right",
        color="k",
    )

    _save(fig, json_path.with_suffix(".png"))


# ---------------------------------------------------------------------------
# Tabular ML screening curves  (source: runs/screening_tabular_*.json)
# ---------------------------------------------------------------------------


def plot_tabular_screening_curve(json_path: Path) -> None:
    """Generate a screening curve plot for one tabular model run.

    Artifact fields used: per-model alpha_values and true_positives, meta (beta, total_positive, dataset)
    Each model gets its own line; Random baseline uses a dashed style.
    """
    data = json.loads(json_path.read_text())
    meta = data["meta"]
    total_pos = meta["total_positive"]
    beta = meta["beta"]
    dataset_name = Path(meta["dataset"]).stem
    metric = "Landmines" if dataset_name == "landmines_raw" else ""

    # All keys except "meta" and "risk_scores" are model results
    model_keys = [k for k in data if k not in ("meta", "risk_scores")]

    print(f"  [{json_path.stem}] Screening curves for {dataset_name} (β={beta:.0%})")

    fig, ax = plt.subplots(figsize=(10, 6))

    for label in model_keys:
        res = data[label]
        style = MODEL_STYLES.get(label, {"color": "black", "marker": "."})
        alpha_pct = [a * 100 for a in res["alpha_values"]]
        tp_pct = [tp / total_pos * 100 for tp in res["true_positives"]]
        linestyle = "--" if label == "Random" else "-"
        line_alpha = 0.7 if label == "Random" else 1.0
        ax.plot(
            alpha_pct,
            tp_pct,
            marker=style["marker"],
            color=style["color"],
            linewidth=3,
            markersize=6,
            label=label,
            linestyle=linestyle,
            alpha=line_alpha,
        )

    ax.set_xlabel("Screening Budget α (%)", fontsize=22)
    ax.set_ylabel(f"Allocation Efficiency (%{metric})", fontsize=21)
    # ax.set_title(
    #    f"Optimal Screening: True Positive Rate vs Screening Budget\n"
    #    f"Dataset: {dataset_name} | β = {beta:.0%}",
    #    fontsize=14, fontweight="bold",
    # )
    ax.legend(fontsize=18, loc="best")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=18)
    ax.set_xlim(0, beta * 100)

    # ax.set_ylim(max(0, min(all_tp_pcts) - 2), min(105, max(all_tp_pcts) + 2))
    ax.set_ylim(0, 100)

    _save(fig, json_path.with_suffix(".png"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all paper figures from JSON artifacts")
    parser.add_argument("--runs-dir", default="runs/", help="Directory containing JSON artifacts (default: runs/)")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)

    # --- Simulation distribution plots ---
    # Source: simulate_distributions.py → runs/simulations.json
    sim_json = runs_dir / "simulations.json"
    if sim_json.exists():
        print(f"\nSimulation plots  ({sim_json})")
        plot_simulations(sim_json)
    else:
        print(f"[skip] {sim_json} not found — run simulate_distributions.py first")

    # --- Utility gap vs Beta parameter plot (paper Figure 4) ---
    # Source: simulate_utility_gap.py → runs/utility_gap_vs_dist_param.json
    ug_json = runs_dir / "utility_gap_vs_dist_param.json"
    if ug_json.exists():
        print(f"\nUtility gap plot  ({ug_json})")
        plot_utility_gap_vs_dist_param(ug_json)
    else:
        print(f"[skip] {ug_json} not found — run simulate_utility_gap.py first")

    # --- Tabular ML screening curve plots (one per dataset) ---
    # Source: run_tabular_models.py → runs/screening_tabular_*.json
    tabular_jsons = sorted(runs_dir.glob("screening_tabular_*.json"))
    if tabular_jsons:
        print(f"\nTabular screening plots  ({len(tabular_jsons)} datasets)")
        for json_path in tabular_jsons:
            plot_tabular_screening_curve(json_path)
    else:
        print("\n[skip] No screening_tabular_*.json found — run run_tabular_models.py first")

    print("\nDone.")


if __name__ == "__main__":
    main()
