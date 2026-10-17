"""Reusable severity-dataset validation and grouped splitting."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


CLASSES = ("low", "medium", "high")
REQUIRED_COLUMNS = {"detection_id", "image_name", "crop_path", "manual_label"}


def load_clean_dataset(csv_path: Path, project_root: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Severity dataset CSV not found: {csv_path}")

    dataframe = pd.read_csv(csv_path)
    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)
    if missing_columns:
        raise ValueError(
            "Severity dataset is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    dataframe["manual_label"] = (
        dataframe["manual_label"].fillna("").str.lower().str.strip()
    )
    invalid_labels = sorted(set(dataframe["manual_label"]) - set(CLASSES))
    if invalid_labels:
        raise ValueError(f"Invalid severity labels: {invalid_labels}")
    if dataframe["detection_id"].duplicated().any():
        raise ValueError("Severity dataset contains duplicate detection_id values")

    missing_crops = [
        str(path)
        for path in dataframe["crop_path"]
        if not (project_root / str(path)).is_file()
    ]
    if missing_crops:
        raise FileNotFoundError(
            f"{len(missing_crops)} severity crops are missing; "
            f"first missing crop: {missing_crops[0]}"
        )
    return dataframe


def create_grouped_splits(
    dataframe: pd.DataFrame,
    *,
    seed: int = 42,
    test_size: float = 0.20,
    validation_size_within_train: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by source image so related crops cannot leak across partitions."""
    if dataframe["image_name"].nunique() < 3:
        raise ValueError("At least three unique source images are required")

    test_splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=seed
    )
    train_indices, test_indices = next(
        test_splitter.split(
            dataframe,
            dataframe["manual_label"],
            groups=dataframe["image_name"],
        )
    )
    train_pool = dataframe.iloc[train_indices].copy()
    test = dataframe.iloc[test_indices].copy()

    validation_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=validation_size_within_train,
        random_state=seed + 1,
    )
    train_sub_indices, validation_indices = next(
        validation_splitter.split(
            train_pool,
            train_pool["manual_label"],
            groups=train_pool["image_name"],
        )
    )
    train = train_pool.iloc[train_sub_indices].copy()
    validation = train_pool.iloc[validation_indices].copy()
    validate_group_isolation(train, validation, test)
    return train, validation, test


def validate_group_isolation(*dataframes: pd.DataFrame) -> None:
    group_sets = [set(frame["image_name"]) for frame in dataframes]
    for first_index, first in enumerate(group_sets):
        for second in group_sets[first_index + 1 :]:
            overlap = first & second
            if overlap:
                raise ValueError(
                    f"Source-image leakage across severity splits: {sorted(overlap)[:3]}"
                )


def save_splits(
    output_dir: Path,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    seed: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = {"train": train, "validation": validation, "test": test}
    summary = {"seed": seed, "classes": list(CLASSES), "splits": {}}

    for name, frame in frames.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
        summary["splits"][name] = {
            "examples": len(frame),
            "source_images": int(frame["image_name"].nunique()),
            "class_counts": {
                label: int((frame["manual_label"] == label).sum())
                for label in CLASSES
            },
        }

    (output_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
