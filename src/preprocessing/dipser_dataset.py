#!/usr/bin/env python3
"""Finalize the validated DIPSER data and export the final dataset.

Input:
    Data/dipser_data_valid.parquet

Output:
    Data/dipser_dataset.parquet
"""

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger


RACE_COLS = [
    "race_asian",
    "race_indian",
    "race_black",
    "race_white",
    "race_middle_eastern",
    "race_latino_hispanic",
]

HEART_RATE_COL = "samsung_hr_none_wakeup_sensor value0_mean"
HEART_RATE_STD_COL = "samsung_hr_none_wakeup_sensor value0_std"
ROTATION_COL = "samsung_rotation_vector value0_mean"
MISSING_SENSORS_THRESHOLD = 0.3
EXCLUDED_SUBJECT_EXPERIMENT = "group01_experiment04_subject_15"
AGE_GROUP_BINS = [13, 20, 22, 26, 44]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("Data/dipser_data_valid.parquet"))
    parser.add_argument("--output", type=Path, default=Path("Data/dipser_dataset.parquet"))
    return parser.parse_args()


def finalize_dataset(df):
    """Apply the final fast cleaning steps before experiments consume the dataset."""
    require_columns(
        df,
        [
            "valid",
            "invalid_reason",
            "group",
            "experiment",
            "subject",
            "time",
            HEART_RATE_COL,
            ROTATION_COL,
            "age",
            "gender_name",
            "race",
            *RACE_COLS,
        ],
    )

    df = df.copy()
    df["valid"] = df["valid"].astype("boolean").fillna(True)
    df = df[df["valid"] == True].copy()

    # DeepFace estimates demographics per frame. Collapse them to one stable
    # subject-level value so fairness groups do not change inside a sequence.
    subject_probs = df.groupby(["group", "subject"])[RACE_COLS].mean().reset_index()
    subject_probs["race"] = subject_probs.apply(get_dominant_race, axis=1)
    subject_probs = subject_probs.drop(columns=RACE_COLS)

    subject_metadata = (
        df.groupby(["group", "subject"])
        .agg({"gender_name": get_mode, "age": "mean"})
        .reset_index()
    )
    subject_metadata = subject_metadata.merge(
        subject_probs,
        on=["group", "subject"],
        how="left",
    )

    df = df.drop(columns=["race", "gender_name", "age"])
    df = df.merge(subject_metadata, on=["group", "subject"], how="left")
    df["age"] = pd.to_numeric(df["age"], errors="coerce").round()
    df["age_group"] = pd.cut(df["age"], bins=AGE_GROUP_BINS).astype(str)
    df = df.drop(columns=RACE_COLS)

    # Subject-experiment IDs are the unit used for sequence construction and
    # for dropping sessions with too much missing sensor information.
    df["subject_experiment_id"] = (
        df["group"].astype(str)
        + "_"
        + df["experiment"].astype(str)
        + "_"
        + df["subject"].astype(str)
    )

    missing_ratio = (
        df.groupby("subject_experiment_id")[HEART_RATE_COL]
        .apply(lambda values: values.isna().mean())
    )
    valid_subjects = missing_ratio[missing_ratio < MISSING_SENSORS_THRESHOLD].index
    df = df[df["subject_experiment_id"].isin(valid_subjects)].copy()
    df = df.dropna(subset=[HEART_RATE_COL, ROTATION_COL])
    df = df.drop(columns=["valid"])

    # Use the mean labeler attention as the regression target; self
    # reports are kept as features/context but not included in this target.
    attention_cols = [
        col for col in df.columns if "attentionfilled" in col and "self" not in col
    ]
    if not attention_cols:
        raise ValueError("No non-self attentionfilled columns found.")
    df["attention"] = df[attention_cols].mean(axis=1)

    df["subject_id"] = df["group"].astype(str) + "_" + df["subject"].astype(str)
    df = df.rename(
        columns={
            HEART_RATE_COL: "heart_rate",
            HEART_RATE_STD_COL: "heart_rate_std",
            "gender_name": "gender",
        }
    )

    # Convert absolute or timedelta-like times into elapsed seconds per session.
    df["time"] = normalize_time(df["time"])
    df["time_sec"] = (
        df.groupby("subject_experiment_id")["time"]
        .transform(lambda values: (values - values.min()).dt.total_seconds().astype(int))
    )

    columns = df.columns.tolist()
    columns.remove("time_sec")
    columns.insert(2, "time_sec")
    df = df[columns]

    df = df[df["subject_experiment_id"] != EXCLUDED_SUBJECT_EXPERIMENT]
    return df


def require_columns(df, columns):
    """Fail early with a clear message if a previous preprocessing step was missed."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def get_mode(series):
    """Return the most frequent non-null value for a subject-level label."""
    values = series.dropna().mode()
    return values.iloc[0] if not values.empty else None


def get_dominant_race(row):
    """Choose the race category with the highest average DeepFace probability."""
    values = row[RACE_COLS]
    if values.isna().all():
        return None
    return values.idxmax().replace("race_", "")


def normalize_time(series):
    """Normalize both fresh timedelta outputs and older datetime parquet files."""
    if pd.api.types.is_timedelta64_dtype(series):
        return pd.Timestamp("1970-01-01") + series

    try:
        return pd.to_datetime(series)
    except (TypeError, ValueError):
        return pd.Timestamp("1970-01-01") + pd.to_timedelta(series.astype(str))


def main():
    args = parse_args()
    valid_df = pd.read_parquet(args.input, engine="pyarrow")
    final_df = finalize_dataset(valid_df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(args.output, engine="pyarrow", index=False)
    logger.info(f"[SAVED] {args.output} with shape {final_df.shape}")


if __name__ == "__main__":
    main()
