#!/usr/bin/env python3
"""Run the full DIPSER preprocessing pipeline.

Input:
    Data/DIPSER/group*/experiment*/subject*/

Output:
    Data/dipser_dataset.parquet
"""

import argparse
import time
from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm.auto import tqdm

from ..classes.experiment_processor import ExperimentProcessor
from .aggregate_sensor_data import (
    aggregate_file,
    aggregate_file_worker,
    concatenate_outputs,
    find_csv_files,
)
from .dipser_dataset import finalize_dataset
from .extract_metadata import process_dataframe
from .validate_frame_metadata import validate_dataframe


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("Data/DIPSER"))
    parser.add_argument("--output-dir", type=Path, default=Path("Data"))
    parser.add_argument("--window-ms", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--group")
    parser.add_argument("--experiment")
    parser.add_argument("--subject")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing step 1 CSVs and step 2 per-subject parquet files.",
    )
    return parser.parse_args()


def paths(output_dir):
    return {
        "extracted_dir": output_dir / "extracted_data_extended",
        "aggregated_dir": output_dir / "extracted_data_aggregated",
        "initial_data": output_dir / "initial_data.parquet",
        "initial_metadata": output_dir / "initial_metadata.parquet",
        "valid_data": output_dir / "dipser_data_valid.parquet",
        "final_data": output_dir / "dipser_dataset.parquet",
    }


def run_step_1(args, pipeline_paths):
    logger.info("Step 1/5: extracting initial subject CSVs.")
    ExperimentProcessor.process_experiments(
        str(args.raw_root),
        group=args.group,
        experiment=args.experiment,
        subject=args.subject,
        output_path=str(pipeline_paths["extracted_dir"]),
        skip_existing=args.skip_existing,
    )


def run_step_2(args, pipeline_paths):
    logger.info("Step 2/5: aggregating sensor data around image frames.")
    files = find_csv_files(
        pipeline_paths["extracted_dir"],
        args.group,
        args.experiment,
        args.subject,
    )
    if not files:
        raise FileNotFoundError(
            f"No subject CSV files found in {pipeline_paths['extracted_dir']}"
        )

    pipeline_paths["aggregated_dir"].mkdir(parents=True, exist_ok=True)

    if args.num_workers < 1:
        raise ValueError("--num-workers must be at least 1")

    if args.num_workers == 1:
        window = pd.Timedelta(milliseconds=args.window_ms)
        output_files = []
        for file in tqdm(files, desc="Processing files"):
            output_path = aggregate_file(
                file,
                pipeline_paths["extracted_dir"],
                pipeline_paths["aggregated_dir"],
                window,
                skip_existing=args.skip_existing,
            )
            if output_path is not None:
                output_files.append(output_path)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        logger.info(f"Aggregating {len(files)} subject files with {args.num_workers} workers.")
        worker_args = [
            (
                file,
                pipeline_paths["extracted_dir"],
                pipeline_paths["aggregated_dir"],
                args.window_ms,
                args.skip_existing,
            )
            for file in files
        ]
        output_files = []
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(aggregate_file_worker, item) for item in worker_args]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
                output_path = future.result()
                if output_path is not None:
                    output_files.append(output_path)
        output_files = sorted(output_files)

    concatenate_outputs(output_files, pipeline_paths["initial_data"])


def run_step_3(args, pipeline_paths):
    logger.info("Step 3/5: extracting demographic metadata.")
    df = pd.read_parquet(pipeline_paths["initial_data"], engine="pyarrow")
    df = process_dataframe(df, args.n_jobs)
    df.to_parquet(pipeline_paths["initial_metadata"], engine="pyarrow", index=False)
    logger.info(f"[SAVED] {pipeline_paths['initial_metadata']}")


def run_step_4(pipeline_paths):
    logger.info("Step 4/5: validating frame metadata.")
    df = pd.read_parquet(pipeline_paths["initial_metadata"], engine="pyarrow")
    df = validate_dataframe(df)
    df.to_parquet(pipeline_paths["valid_data"], engine="pyarrow", index=False)
    logger.info(f"[SAVED] {pipeline_paths['valid_data']}")


def run_step_5(pipeline_paths):
    logger.info("Step 5/5: building final DIPSER dataset.")
    df = pd.read_parquet(pipeline_paths["valid_data"], engine="pyarrow")
    df = finalize_dataset(df)
    df.to_parquet(pipeline_paths["final_data"], engine="pyarrow", index=False)
    logger.info(f"[SAVED] {pipeline_paths['final_data']} with shape {df.shape}")


def main():
    start = time.time()
    args = parse_args()
    pipeline_paths = paths(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_step_1(args, pipeline_paths)
    run_step_2(args, pipeline_paths)
    run_step_3(args, pipeline_paths)
    run_step_4(pipeline_paths)
    run_step_5(pipeline_paths)

    logger.info(f"Full preprocessing completed in {time.time() - start:.2f}s.")


if __name__ == "__main__":
    main()
