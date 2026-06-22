#!/usr/bin/env python3
"""Step 1: extract initial per-subject CSV files from the raw DIPSER folder.

Input:
    Data/DIPSER/group*/experiment*/subject*/

Output:
    Data/extracted_data_extended/group*/experiment*/subject*.csv
"""

import argparse

from ..classes.experiment_processor import ExperimentProcessor
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_path", required=True, help="Base path of the experiments")
    parser.add_argument("--group", required=False, help="Specify group to process")
    parser.add_argument("--experiment", required=False, help="Specify experiment to process")
    parser.add_argument("--subject", required=False, help="Specify subject to process")
    parser.add_argument("--output_path", required=True, help="Where to save processed data")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip subject CSV files that already exist in the output path.",
    )
    
    args = parser.parse_args()

    ExperimentProcessor.process_experiments(
    args.base_path,
    args.group,
    args.experiment,
    args.subject,
    args.output_path,
    args.skip_existing,
    )
