# Optimal Screening Decisions

[![arXiv](https://img.shields.io/badge/arXiv-2605.07979v1-b31b1b.svg)](https://arxiv.org/abs/2605.07979v1)

This directory contains a small, runnable implementation for turning risk predictions into optimal screening decisions.
Given a table of cases, a risk score, a treatment budget, and a screening budget, the code writes one action per row:

- `0`: ignore
- `1`: treat directly
- `2`: screen

The implementation accompanies the paper [The Limits of AI-Driven Allocation: Optimal Screening under Aleatoric Uncertainty](https://arxiv.org/abs/2605.07979v1), but the main entry point here is the decision calculator, not the replication pipeline.

## Install

Install dependencies with `uv`:

```bash
uv sync
```

The project targets Python 3.11.

## Quickstart: Compute Decisions From Risk Predictions

Create a YAML or JSON config that points to either a local CSV file or a Hugging Face dataset.
When you already have risk predictions, set `risk_col` to the column containing predicted risk, where larger values mean higher risk.

```yaml
csv: data/my-risk-predictions.csv
outcome: outcome
strata:
  - group
risk_col: predicted_risk
beta: 0.50
alpha: 0.25
output: runs/my-screening-decisions.csv
```

Run the decision calculator:

```bash
uv run get-optimal-screening configs/my-risk-config.yaml
```

The output CSV contains the original rows plus a `screening_decision` column.
For example:

```csv
predicted_risk,outcome,group,screening_decision
0.91,1,a,1
0.76,1,a,1
0.34,0,b,2
0.08,0,b,0
```

Budgets are population proportions:

- `beta`: treatment budget. For example, `0.20` means at most 20% of rows can receive treatment.
- `alpha`: screening budget. For example, `0.05` means at most 5% of rows can be screened.
- `alpha` must be between `0` and `beta`.

The current CLI expects an `outcome` column because the same code path is also used for labeled evaluation and paper artifacts.
When `risk_col` is provided, the action assignment is ranked by that risk column.

## Config Fields

- `csv`: local CSV path for custom data.
- `hf_dataset`: Hugging Face dataset repository ID.
- `hf_split`: optional Hugging Face split, default `train`.
- `hf_revision`: optional Hugging Face dataset revision.
- `outcome`: required binary outcome column.
- `strata`: required non-empty list of columns. Used to estimate empirical risk strata when `risk_col` is not provided.
- `risk_col`: optional column containing precomputed risk predictions.
- `prediction_col`: optional model prediction column, default `probability`.
- `beta`: treatment budget, in `(0, 1]`.
- `alpha`: screening budget, in `[0, beta]`.
- `action_col`: optional output decision column name, default `screening_decision`.
- `output`: optional output CSV path, default `runs/optimal_screening.csv`.

Provide exactly one data source: `csv` or `hf_dataset`.

## Hugging Face Dataset Example

The included example uses the landmine dataset from Hugging Face and writes decisions to `runs/example-risk-output.csv`:

```bash
uv run get-optimal-screening configs/example-risk.yaml
```

Its config is:

```yaml
hf_dataset: cmpatino/landmine-detection
hf_split: train
outcome: mines_outcome
strata:
  - Municipio
beta: 0.1
alpha: 0.05
output: runs/example-risk-output.csv
```

Because this example does not specify `risk_col`, risks are estimated empirically from the listed strata.

## Python API

Use the core function directly when you already have rows in memory:

```python
from optimal_screening.analysis import compute_optimal_screening_actions

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
```

For screening curves across multiple `alpha` values, use `compute_optimal_screening_curve`.

## Citation

If you use this code or the optimal screening method, please cite:

```bibtex
@misc{cortesgomez2026limitsaidrivenallocationoptimal,
      title={The Limits of AI-Driven Allocation: Optimal Screening under Aleatoric Uncertainty},
      author={Santiago Cortes-Gomez and Mateo Dulce Rubio and Carlos Patino and Bryan Wilder},
      year={2026},
      eprint={2605.07979},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2605.07979},
}
```

## Reproduce Paper Results

The camera-ready replication pipeline is still available:

```bash
uv run replicate-results
```

This writes JSON artifacts and PNG figures to `runs/`.
Input tabular data is downloaded from Hugging Face with the `datasets` library and cached locally by Hugging Face.
The replication command sets `HF_HOME=.cache/huggingface` so the cache stays inside this project directory.

Runtime and requirements:

- Expect the full command to take about 15-25 minutes on a recent laptop or desktop. Most time is spent in the simulation loop and fitting the ACS 2018 tabular models.
- The command needs internet access on the first run to download `cmpatino/landmine-detection` and `cmpatino/acs-income-2018` from Hugging Face. Later runs reuse the local `.cache/huggingface` cache unless it is removed.
- Use Python 3.11 through `uv`. The ACS step trains on roughly 1.6 million rows, so run it on a machine with several GB of free RAM and enough disk space for the Hugging Face cache, virtual environment, and `runs/` outputs.

The command generates:

- `runs/simulations.json`
- `runs/utility_gap_vs_dist_param.json`
- `runs/screening_tabular_landmines_granada_test.json`
- `runs/screening_tabular_acs2018_all_strata_50pct.json`
- all figures generated by `scripts/plot_figures.py`

The one-command pipeline runs these steps:

```bash
uv run python scripts/simulate_distributions.py
uv run python scripts/simulate_utility_gap.py
uv run python scripts/run_tabular_models.py --hf-dataset cmpatino/landmine-detection --outcome mines_outcome \
  --strata Municipio --beta 0.1 --test-municipio GRANADA \
  --output runs/screening_tabular_landmines_granada_test.json
uv run python scripts/run_tabular_models.py --hf-dataset cmpatino/acs-income-2018 --outcome "PINCP > 50k" \
  --strata AGEP COW SCHL MAR OCCP POBP RELP WKHP SEX RAC1P --beta 0.3 --strata-only \
  --test-size 0.5 --output runs/screening_tabular_acs2018_all_strata_50pct.json
uv run python scripts/plot_figures.py --runs-dir runs/
```

## Repository Map

- `src/optimal_screening/analysis/stratified.py`: core screening algorithms.
- `src/optimal_screening/cli/get_optimal_screening.py`: decision-calculator CLI.
- `src/optimal_screening/cli/replicate_results.py`: paper replication CLI.
- `scripts/`: standalone artifact and plotting scripts.
- `configs/`: example optimal-screening configuration.
- `tests/`: focused unit tests for the core algorithm and screening CLI.
