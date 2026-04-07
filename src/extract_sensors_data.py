import pandas as pd
import numpy as np
import glob
import json
import os
from datetime import datetime
import argparse
from loguru import logger

# each sensor file has a time interval of some seconds. there is no overlap between intervals for the files

def build_sensor_intervals(sensor_folder):
    sensor_files = glob.glob(sensor_folder)

    intervals = []
    data_cache = {}

    for f in sensor_files:
        with open(f, "r") as fp:
            data = json.load(fp)

        times = []

        for sensor in data["data"].values():
            for entry in sensor:
                times.append(datetime.strptime(entry["timestamp"], "%H:%M:%S:%f"))

        if not times:
            continue

        start = min(times)
        end = max(times)

        intervals.append((f, start, end))
        data_cache[f] = data

    intervals.sort(key=lambda x: x[1])

    return intervals, data_cache



#def find_sensor_interval(target_time, intervals, idx):
#    n = len(intervals)
#
#    while idx < n - 1 and target_time > intervals[idx][2]:
#        idx += 1
#
#    f, start, end = intervals[idx]
#
#    if start <= target_time <= end:
#        return f, idx
#
#    return None, idx


def find_sensor_interval(target_time, intervals, idx, max_diff_sec=0.5):
    n = len(intervals)

    # Move pointer forward
    while idx < n - 1 and target_time > intervals[idx][2]:
        idx += 1

    candidates = []

    # current
    f, start, end = intervals[idx]
    dist_current = min(
        abs((target_time - start).total_seconds()),
        abs((target_time - end).total_seconds())
    )
    candidates.append((dist_current, idx))

    # previous
    if idx > 0:
        f_prev, s_prev, e_prev = intervals[idx - 1]
        dist_prev = min(
            abs((target_time - s_prev).total_seconds()),
            abs((target_time - e_prev).total_seconds())
        )
        candidates.append((dist_prev, idx - 1))

    # next
    if idx < n - 1:
        f_next, s_next, e_next = intervals[idx + 1]
        dist_next = min(
            abs((target_time - s_next).total_seconds()),
            abs((target_time - e_next).total_seconds())
        )
        candidates.append((dist_next, idx + 1))

    # pick closest
    best_dist, best_idx = min(candidates, key=lambda x: x[0])

    # if the frame does not correspond to any of the intervals, assign it to the closest one 
    # as long as the difference is not higher than threshold (1 second)
    if best_dist > max_diff_sec:
        return None, idx

    f_best = intervals[best_idx][0]

    return f_best, best_idx



def compute_magnitude(x, y, z):
    return np.sqrt(x**2 + y**2 + z**2)


def aggregate_sensor(sensor_data, sensor_name):
    features = {}

    if len(sensor_data) == 0:
        return features

    values = {}
    for key in sensor_data[0].keys():
        if key.startswith("value"):
            values[key] = np.array([d[key] for d in sensor_data])

    if sensor_name in ["samsung_linear_acceleration_sensor", "lsm6dso_gyroscope"]:
        x, y, z = values["value0"], values["value1"], values["value2"]
        mag = compute_magnitude(x, y, z)

        features.update({
            f"{sensor_name}_mean_x": x.mean(),
            f"{sensor_name}_std_x": x.std(),
            f"{sensor_name}_mean_y": y.mean(),
            f"{sensor_name}_std_y": y.std(),
            f"{sensor_name}_mean_z": z.mean(),
            f"{sensor_name}_std_z": z.std(),
            f"{sensor_name}_mean_mag": mag.mean(),
            f"{sensor_name}_std_mag": mag.std(),
        })

    elif sensor_name == "samsung_rotation_vector":
        for k, v in values.items():
            features[f"{sensor_name}_mean_{k}"] = v.mean()
            features[f"{sensor_name}_std_{k}"] = v.std()

    elif sensor_name == "opt3007_light":
        v = values["value0"]
        features[f"{sensor_name}_mean"] = v.mean()
        features[f"{sensor_name}_std"] = v.std()

    elif sensor_name == "samsung_hr_none_wakeup_sensor":
        v = values["value0"]
        features[f"{sensor_name}_value"] = v.mean()

    return features


def aggregate_all_sensors(data):
    all_features = {}
    for sensor_name, sensor_data in data["data"].items():
        feats = aggregate_sensor(sensor_data, sensor_name)
        all_features.update(feats)

    return all_features

def process_sensors(df, sensor_folder):
    intervals, data_cache = build_sensor_intervals(sensor_folder)

    sensor_rows = []
    idx = 0

    for _, row in df.iterrows():
        target_time = datetime.strptime(row["time"], "%H:%M:%S.%f")

        sensor_file, idx = find_sensor_interval(target_time, intervals, idx)

        if sensor_file is None:
            sensor_rows.append({})
            continue

        sensor_json = data_cache[sensor_file]
        features = aggregate_all_sensors(sensor_json)
        sensor_rows.append(features)

    sensor_df = pd.DataFrame(sensor_rows)

    return pd.concat([df.reset_index(drop=True), sensor_df], axis=1)


# loop over all subjects

def add_sensors_to_df(df, base_path):

    sensor_parts = []

    grouped = df.groupby(["group", "experiment", "subject"])

    for (group, experiment, subject), sub_df in grouped:
        logger.info(f"Processing {group} | {experiment} | {subject}")

        sensor_folder = os.path.join(
            base_path,
            group,
            experiment,
            subject,
            "watch_sensors",
            "*.json"
        )

        if len(glob.glob(sensor_folder)) == 0:
            logger.warning(f"No sensor data for {subject}")
            sensor_parts.append(sub_df)
            continue

        sensor_sub = process_sensors(sub_df, sensor_folder)
        sensor_parts.append(sensor_sub)

    added_sensors_df = pd.concat(sensor_parts, ignore_index=True)

    return added_sensors_df



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--base_path", required=True)
    parser.add_argument("--output_file", required=True)

    args = parser.parse_args()

    df = pd.read_parquet(args.input_file)

    sensors_df = add_sensors_to_df(df, args.base_path)

    sensors_df.to_parquet(args.output_file, engine='pyarrow', index=False)

    logger.info("Finished enrichment")
