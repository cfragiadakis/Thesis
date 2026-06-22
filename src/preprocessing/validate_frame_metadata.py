#!/usr/bin/env python3
"""Step 4: Validate frame metadata and mark noisy image records.

Input:
    Data/initial_metadata.parquet

Output:
    Data/dipser_data_valid.parquet
"""

import argparse
import json
import tarfile
from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm.auto import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("Data/initial_metadata.parquet"))
    parser.add_argument("--output", type=Path, default=Path("Data/dipser_data_valid.parquet"))
    return parser.parse_args()


def get_tar_and_filename(metadata_path):
    """Map a frame metadata path to its subject-level metadata.tar archive."""
    path = Path(str(metadata_path))

    try:
        metadata_idx = path.parts.index("metadata")
    except ValueError:
        return None, None

    subject_dir = Path(*path.parts[:metadata_idx])
    tar_path = subject_dir / "metadata.tar"
    return tar_path, path.name


def load_tar_index(tar_path):
    """Open one metadata archive and index JSON members by filename."""
    tar = tarfile.open(tar_path, "r")
    index = {}

    for member in tar.getmembers():
        if member.isfile() and member.name.endswith(".json"):
            index[Path(member.name).name] = member

    return tar, index


def read_metadata_from_tar(tar, member):
    file = tar.extractfile(member)
    if file is None:
        return None
    return json.load(file)


def is_valid_frame(metadata_path, metadata):
    """Reject frames where the face metadata is missing or geometrically invalid."""
    if metadata is None:
        return False, "metadata_none"

    person = metadata.get("person")
    face = person.get("face")
    if face is None:
        return False, "missing_face"

    bbox = face.get("bounding_box")
    if bbox is None:
        return False, "missing_bbox"

    if isinstance(bbox, dict):
        x0 = bbox.get("x0")
        y0 = bbox.get("y0")
        x1 = bbox.get("x1")
        y1 = bbox.get("y1")
    elif isinstance(bbox, list):
        if len(bbox) != 4:
            return False, "malformed_bbox"
        x0, y0, x1, y1 = bbox
    else:
        return False, "unknown_bbox_format"

    if None in (x0, y0, x1, y1):
        return False, "incomplete_bbox"

    width = x1 - x0
    height = y1 - y0

    if width <= 0 or height <= 0:
        return False, "invalid_bbox"

    return True, "valid"


def validate_dataframe(df):
    results = []
    reasons = []
    # Each subject has many frame rows but one metadata.tar archive, so keep
    # opened archives and filename indexes around for the full pass.
    tar_cache = {}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Validating frames"):
        metadata_path = row["metadata"]
        tar_path, filename = get_tar_and_filename(metadata_path)

        if tar_path is None:
            results.append(None)
            reasons.append("invalid_tar_path")
            continue

        try:
            if tar_path not in tar_cache:
                tar, index = load_tar_index(tar_path)
                tar_cache[tar_path] = (tar, index)

            tar, index = tar_cache[tar_path]
        except Exception:
            results.append(None)
            reasons.append("tar_open_failure")
            continue

        member = index.get(filename)
        if member is None:
            results.append(None)
            reasons.append("missing_json_in_tar")
            continue

        try:
            metadata = read_metadata_from_tar(tar, member)
        except Exception:
            results.append(None)
            reasons.append("metadata_read_failure")
            continue

        if metadata is None or metadata == {}:
            results.append(None)
            reasons.append("empty_metadata")
            continue

        # Store both the boolean flag used by the final filter and the exact
        # reason, which is useful for auditing how many frames were removed.
        try:
            valid, reason = is_valid_frame(metadata_path, metadata)
            results.append(valid)
            reasons.append(reason)
        except Exception:
            results.append(None)
            reasons.append("validation_failure")

    for tar, _ in tar_cache.values():
        tar.close()

    df = df.copy()
    df["valid"] = results
    df["invalid_reason"] = reasons
    logger.info("Validation reason counts:")
    logger.info(df["invalid_reason"].value_counts(dropna=False).to_string())
    return df


def main():
    args = parse_args()
    df = pd.read_parquet(args.input, engine="pyarrow")
    df = validate_dataframe(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, engine="pyarrow", index=False)
    logger.info(f"[SAVED] {args.output}")


if __name__ == "__main__":
    main()
