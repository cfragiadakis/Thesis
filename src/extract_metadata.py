#!/usr/bin/env python3

import pandas as pd
import json
from multiprocessing import Pool
import argparse
import time
from loguru import logger

# Metadata extraction function

def extract_metadata(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        face = data.get("person", {}).get("face", {})

        age = face.get("age", None)
        gender = face.get("gender", {}).get("gender_name", None)
        race = face.get("race", {}).get("dominant_race", None)

        return age, gender, race

    except Exception:
        return None, None, None



# Parallel processing

def process_dataframe(df, n_jobs):
    paths = df["metadata"].tolist()
    results = []

    for i, p in enumerate(paths):
        results.append(extract_metadata(p))

        if i % 5000 == 0:
            logger.info(f"Processed {i}/{len(paths)} rows")

    df[["age", "gender_name", "race"]] = pd.DataFrame(results, index=df.index)
    return df


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input dataframe path (parquet)")
    parser.add_argument("--output", required=True, help="Output dataframe path")
    parser.add_argument("--n_jobs", type=int, default=16)

    args = parser.parse_args()

    logger.info("Loading dataframe...")
    df = pd.read_parquet(args.input)

    logger.info(f"Processing {len(df)} rows with {args.n_jobs} workers...")

    df = process_dataframe(df, args.n_jobs)

    logger.info("Saving output...")
    df.to_parquet(args.output)

    logger.info("Done.")
    logger.info(f"Elapsed: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
