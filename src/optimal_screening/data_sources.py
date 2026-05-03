from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import load_dataset


def load_hf_dataframe(dataset: str, split: str = "train", revision: str | None = None) -> pd.DataFrame:
    """Load a tabular Hugging Face dataset split as a pandas DataFrame."""
    kwargs = {"path": dataset, "split": split}
    if revision is not None:
        kwargs["revision"] = revision
    return load_dataset(**kwargs).to_pandas()


def load_dataframe(
    *,
    csv_path: str | None = None,
    hf_dataset: str | None = None,
    hf_split: str = "train",
    hf_revision: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """Load a DataFrame from exactly one supported source and return a source label."""
    sources = [source is not None for source in (csv_path, hf_dataset)]
    if sum(sources) != 1:
        raise ValueError("Provide exactly one data source: csv_path or hf_dataset")

    if hf_dataset is not None:
        return load_hf_dataframe(hf_dataset, split=hf_split, revision=hf_revision), hf_dataset

    assert csv_path is not None
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path), str(path)
