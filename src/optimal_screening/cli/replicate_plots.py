from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _cleanup_previous_outputs(runs_dir: Path) -> None:
    patterns = [
        "simulations.json",
        "utility_gap_vs_dist_param.json",
        "screening_tabular_*.json",
        "*.png",
    ]
    for pattern in patterns:
        for path in runs_dir.glob(pattern):
            path.unlink()


def _run(args: list[str], env: dict[str, str]) -> None:
    print("\n$ " + " ".join(args))
    subprocess.run(args, cwd=PROJECT_ROOT, env=env, check=True)


def main() -> None:
    runs_dir = PROJECT_ROOT / "runs"
    cache_dir = PROJECT_ROOT / ".cache"
    runs_dir.mkdir(exist_ok=True)
    cache_dir.mkdir(exist_ok=True)
    _cleanup_previous_outputs(runs_dir)

    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    env.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))
    env.setdefault("HF_HOME", str(cache_dir / "huggingface"))
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(env["HF_HOME"]).mkdir(parents=True, exist_ok=True)

    python = sys.executable
    _run([python, "scripts/simulate_distributions.py", "--output", "runs/simulations.json"], env)
    _run(
        [
            python,
            "scripts/simulate_utility_gap.py",
            "--output",
            "runs/utility_gap_vs_dist_param.json",
        ],
        env,
    )
    _run(
        [
            python,
            "scripts/run_tabular_models.py",
            "--hf-dataset",
            "cmpatino/landmine-detection",
            "--outcome",
            "mines_outcome",
            "--strata",
            "Municipio",
            "--beta",
            "0.1",
            "--test-municipio",
            "GRANADA",
            "--output",
            "runs/screening_tabular_landmines_granada_test.json",
        ],
        env,
    )
    _run(
        [
            python,
            "scripts/run_tabular_models.py",
            "--hf-dataset",
            "cmpatino/acs-income-2018",
            "--outcome",
            "PINCP > 50k",
            "--strata",
            "AGEP",
            "COW",
            "SCHL",
            "MAR",
            "OCCP",
            "POBP",
            "RELP",
            "WKHP",
            "SEX",
            "RAC1P",
            "--beta",
            "0.3",
            "--strata-only",
            "--test-size",
            "0.5",
            "--output",
            "runs/screening_tabular_acs2018_all_strata_50pct.json",
        ],
        env,
    )
    _run([python, "scripts/plot_figures.py", "--runs-dir", "runs/"], env)

    print("\nPaper artifacts and plots written to runs/")


if __name__ == "__main__":
    main()
