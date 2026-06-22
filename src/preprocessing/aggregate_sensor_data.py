#!/usr/bin/env python3
"""Step 2: aggregate sensor readings around image frames.

Inputs:
    Data/extracted_data_extended/group*/experiment*/subject*.csv

Outputs:
    Data/extracted_data_aggregated/group*/experiment*/subject*.parquet
    Data/initial_data.parquet
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from tqdm.auto import tqdm


ACCEL_COLS = [
    "samsung_linear_acceleration_sensor value0",
    "samsung_linear_acceleration_sensor value1",
    "samsung_linear_acceleration_sensor value2",
]

GYRO_COLS = [
    "lsm6dso_gyroscope value0",
    "lsm6dso_gyroscope value1",
    "lsm6dso_gyroscope value2",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("Data/extracted_data_extended"))
    parser.add_argument("--output-dir", type=Path, default=Path("Data/extracted_data_aggregated"))
    parser.add_argument("--initial-data", type=Path, default=Path("Data/initial_data.parquet"))
    parser.add_argument("--window-ms", type=int, default=500)
    parser.add_argument("--group")
    parser.add_argument("--experiment")
    parser.add_argument("--subject")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of subject CSV files to aggregate in parallel.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip subject parquet files that already exist in the output directory.",
    )
    return parser.parse_args()


def find_csv_files(input_dir, group=None, experiment=None, subject=None):
    """Find subject CSV files, optionally restricted to one split of the raw tree."""
    files = sorted(input_dir.glob("group*/experiment*/subject*.csv"))
    selected = []

    for path in files:
        file_group = path.parent.parent.name
        file_experiment = path.parent.name
        file_subject = path.stem

        if group and file_group != group:
            continue
        if experiment and file_experiment != experiment:
            continue
        if subject and file_subject != subject:
            continue

        selected.append(path)

    return selected


def get_output_path(csv_path, input_dir, output_dir):
    relative_path = Path(csv_path).relative_to(input_dir).with_suffix(".parquet")
    return Path(output_dir) / relative_path


def aggregate_file(
    csv_path,
    input_dir,
    output_dir,
    window,
    show_frame_progress=True,
    skip_existing=False,
):
    """Align sensor rows to image frames and write one aggregated subject parquet."""
    csv_path = Path(csv_path)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_path = get_output_path(csv_path, input_dir, output_dir)

    if skip_existing and output_path.exists():
        logger.info(f"[SKIP] Existing output: {output_path}")
        return output_path

    logger.info(f"Processing: {csv_path}")
    df = pd.read_csv(csv_path)

    # Step 1 output contains image rows and raw sensor rows in the same table.
    # Image rows become the final frame-level records; sensor rows are summarized.
    images_df = df[df["image_path"].notna()].copy()
    sensors_df = df[df["image_path"].isna()].copy()

    if images_df.empty:
        logger.warning(f"Skipping {csv_path}: no image rows.")
        return None

    images_df["time"] = pd.to_timedelta(images_df["time"].astype(str))
    sensors_df["time"] = pd.to_timedelta(sensors_df["time"].astype(str))

    images_df = images_df.sort_values("time")
    sensors_df = sensors_df.sort_values("time")

    sensor_columns = [
        col
        for col in sensors_df.columns
        if "value" in col and sensors_df[col].notna().any()
    ]

    aggregated_rows = []

    for _, img_row in tqdm(
        images_df.iterrows(),
        total=len(images_df),
        leave=False,
        disable=not show_frame_progress,
        desc="Aggregating",
    ):
        t = img_row["time"]
        # Summarize sensor values close to this frame so each image has one
        # synchronized sensor feature vector.
        nearby = sensors_df[
            (sensors_df["time"] >= t - window)
            & (sensors_df["time"] <= t + window)
        ]

        new_row = img_row.to_dict()

        for col in sensor_columns:
            values = nearby[col].dropna()
            if len(values) == 0:
                new_row[f"{col}_mean"] = np.nan
                new_row[f"{col}_std"] = np.nan
            else:
                new_row[f"{col}_mean"] = values.mean()
                new_row[f"{col}_std"] = values.std()

        add_magnitude(new_row, nearby, ACCEL_COLS, "accel")
        add_magnitude(new_row, nearby, GYRO_COLS, "gyro")

        aggregated_rows.append(new_row)

    final_df = pd.DataFrame(aggregated_rows)

    # Raw per-sample sensor values are no longer needed after mean/std features
    # have been computed around each image timestamp.
    raw_sensor_cols = [
        col
        for col in final_df.columns
        if "value" in col and not col.endswith("_mean") and not col.endswith("_std")
    ]
    final_df = final_df.drop(columns=raw_sensor_cols)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    final_df.to_parquet(output_path, engine="pyarrow", index=False)
    logger.info(f"[SAVED] {output_path}")
    return output_path


def aggregate_file_worker(args):
    csv_path, input_dir, output_dir, window_ms, skip_existing = args
    window = pd.Timedelta(milliseconds=window_ms)
    return aggregate_file(
        csv_path,
        input_dir,
        output_dir,
        window,
        show_frame_progress=False,
        skip_existing=skip_existing,
    )


def add_magnitude(row, nearby, columns, prefix):
    """Add vector magnitude features for 3-axis motion sensors."""
    if not all(col in nearby.columns for col in columns):
        return

    valid = nearby[columns].dropna()
    if len(valid) == 0:
        row[f"{prefix}_magnitude_mean"] = np.nan
        row[f"{prefix}_magnitude_std"] = np.nan
        return

    magnitude = np.sqrt(
        valid[columns[0]] ** 2
        + valid[columns[1]] ** 2
        + valid[columns[2]] ** 2
    )
    row[f"{prefix}_magnitude_mean"] = magnitude.mean()
    row[f"{prefix}_magnitude_std"] = magnitude.std()


def concatenate_outputs(parquet_files, output_path):
    """Create the single parquet consumed by the metadata extraction step."""
    if not parquet_files:
        raise RuntimeError("No aggregated parquet files were created.")

    logger.info(f"Concatenating {len(parquet_files)} files into {output_path}")
    df = pd.concat((pd.read_parquet(path) for path in parquet_files), ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow", index=False)
    logger.info(f"[SAVED] {output_path}")


def main():
    args = parse_args()
    if args.num_workers < 1:
        raise ValueError("--num-workers must be at least 1")

    files = find_csv_files(args.input_dir, args.group, args.experiment, args.subject)

    if not files:
        raise FileNotFoundError(f"No subject CSV files found in {args.input_dir}")

    window = pd.Timedelta(milliseconds=args.window_ms)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.num_workers == 1:
        output_files = []
        for file in tqdm(files, desc="Processing files"):
            output_path = aggregate_file(
                file,
                args.input_dir,
                args.output_dir,
                window,
                skip_existing=args.skip_existing,
            )
            if output_path is not None:
                output_files.append(output_path)
    else:
        logger.info(f"Aggregating {len(files)} subject files with {args.num_workers} workers.")
        output_files = []
        worker_args = [
            (file, args.input_dir, args.output_dir, args.window_ms, args.skip_existing)
            for file in files
        ]
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(aggregate_file_worker, item) for item in worker_args]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
                output_path = future.result()
                if output_path is not None:
                    output_files.append(output_path)

        output_files = sorted(output_files)

    concatenate_outputs(output_files, args.initial_data)


if __name__ == "__main__":
    main()
