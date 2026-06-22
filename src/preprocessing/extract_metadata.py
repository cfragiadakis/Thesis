#!/usr/bin/env python3
"""Step 3: extract demographic metadata from frame metadata JSON paths.

Input:
    Data/initial_data.parquet

Output:
    Data/initial_metadata.parquet
"""

import pandas as pd
import json
import argparse
import tarfile
import time
from pathlib import Path
from loguru import logger


def read_metadata(json_path, tar_cache):
    """Read metadata from a plain JSON file or from the subject metadata.tar."""
    path = Path(str(json_path))

    if path.is_file():
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    tar_path = path.parent.with_suffix(".tar")
    filename = path.name

    if tar_path not in tar_cache:
        tar = tarfile.open(tar_path, "r")
        index = {
            Path(member.name).name: member
            for member in tar.getmembers()
            if member.isfile() and member.name.endswith(".json")
        }
        tar_cache[tar_path] = (tar, index)

    tar, index = tar_cache[tar_path]
    member = index.get(filename)
    if member is None:
        return None

    file = tar.extractfile(member)
    if file is None:
        return None
    return json.load(file)


def extract_metadata(json_path, tar_cache):
    """Extract frame-level DeepFace age, gender, race, and race probabilities."""
    try:
        data = read_metadata(json_path, tar_cache)
        if not data:
            return (None,) * 9

        face = data.get("person", {}).get("face", {})

        age = face.get("age", None)
        gender = face.get("gender", {}).get("gender_name", None)
        race = face.get("race", {}).get("dominant_race", None)

        # race probabilities
        prob_race = face.get("race", {}).get("probability_race", {})

        return (
            age,
            gender,
            race,
            prob_race.get("asian", None),
            prob_race.get("indian", None),
            prob_race.get("black", None),
            prob_race.get("white", None),
            prob_race.get("middle eastern", None),
            prob_race.get("latino hispanic", None),
        )

    except Exception:
        return (None,) * 9

def process_dataframe(df, n_jobs):
    paths = df["metadata"].tolist()
    results = []
    tar_cache = {}

    try:
        for i, p in enumerate(paths):
            results.append(extract_metadata(p, tar_cache))

            if i % 5000 == 0:
                logger.info(f"Processed {i}/{len(paths)} rows")
    finally:
        for tar, _ in tar_cache.values():
            tar.close()

    df[[
    "age",
    "gender_name",
    "race",
    "race_asian",
    "race_indian",
    "race_black",
    "race_white",
    "race_middle_eastern",
    "race_latino_hispanic"
    ]] = pd.DataFrame(results, index=df.index)
    return df



def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
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
