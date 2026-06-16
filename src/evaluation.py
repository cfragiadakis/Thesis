from __future__ import annotations

import json
import os
import time
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    r2_score,
    roc_auc_score,
)


DEFAULT_BINARY_THRESHOLD = 3.0
DEFAULT_TOLERANCE = 0.15
DEFAULT_ROBUSTNESS_SUBSETS = {
    "at_least_2_missing_images": "at_least_2_missing_images",
    "at_least_4_missing_images": "at_least_4_missing_images",
}


def build_robustness_metadata(
    sequence_df: pd.DataFrame,
    visual_missing_col: str,
) -> pd.DataFrame:
    """Build per-sequence difficulty labels for robustness analysis.

    The labels are derived from the already-created temporal sequence dataframe,
    so they stay aligned with the exact test samples used for prediction.
    """
    required = {"subject_experiment_id", "time_sec", visual_missing_col}
    missing = required.difference(sequence_df.columns)
    if missing:
        raise KeyError(f"sequence_df is missing required columns: {sorted(missing)}")

    key_cols = ["subject_experiment_id", "time_sec"]
    for optional_col in ["subject_id", "gender", "age"]:
        if optional_col in sequence_df.columns and optional_col not in key_cols:
            key_cols.append(optional_col)

    metadata = sequence_df[key_cols].copy()
    visual_missing = sequence_df[visual_missing_col]
    sequence_length = visual_missing.apply(len).replace(0, np.nan)

    metadata["visual_missing_count"] = visual_missing.apply(lambda x: int(np.sum(x)))
    metadata["visual_missing_rate"] = metadata["visual_missing_count"] / sequence_length
    metadata["at_least_2_missing_images"] = metadata["visual_missing_count"] >= 2
    metadata["at_least_4_missing_images"] = metadata["visual_missing_count"] >= 4

    return metadata


def add_robustness_metadata(
    results_df: pd.DataFrame,
    sequence_df: pd.DataFrame,
    visual_missing_col: str,
) -> pd.DataFrame:
    metadata = build_robustness_metadata(
        sequence_df=sequence_df,
        visual_missing_col=visual_missing_col,
    )

    merge_keys = ["subject_experiment_id", "time_sec"]
    return results_df.merge(metadata, on=merge_keys, how="left", suffixes=("", "_sequence"))


def robustness_subset_table(
    predictions_by_model: Mapping[str, pd.DataFrame],
    subsets: Mapping[str, str | None] | None = None,
) -> pd.DataFrame:
    subsets = subsets or DEFAULT_ROBUSTNESS_SUBSETS
    rows = []

    for model_name, df in predictions_by_model.items():
        working = df.copy()
        working["abs_error"] = np.abs(
            working["true"].astype(float) - working["pred"].astype(float)
        )

        for subset_name, subset_col in subsets.items():
            if subset_col is None:
                subset = working
            elif subset_col not in working.columns:
                continue
            else:
                subset = working[working[subset_col].fillna(False).astype(bool)]

            if subset.empty:
                rows.append(
                    {
                        "model": model_name,
                        "subset": subset_name,
                        "n_samples": 0,
                        "mae": np.nan,
                        "rmse": np.nan,
                    }
                )
                continue

            errors = subset["abs_error"]
            rows.append(
                {
                    "model": model_name,
                    "subset": subset_name,
                    "n_samples": int(len(subset)),
                    "mae": float(errors.mean()),
                    "rmse": float(np.sqrt(np.mean(errors**2))),
                    "true_mean": float(subset["true"].mean()),
                    "pred_mean": float(subset["pred"].mean()),
                }
            )

    return pd.DataFrame(rows)


def paired_robustness_comparison(
    predictions_by_model: Mapping[str, pd.DataFrame],
    baseline_model: str,
    candidate_model: str,
    subsets: Mapping[str, str | None] | None = None,
    keys: Iterable[str] = ("subject_experiment_id", "time_sec"),
) -> pd.DataFrame:
    """Compare two models on exactly matched prediction rows.

    Positive candidate_mae_gain means the candidate model reduced absolute error
    relative to the baseline model.
    """
    subsets = subsets or DEFAULT_ROBUSTNESS_SUBSETS
    keys = list(keys)

    baseline = predictions_by_model[baseline_model].copy()
    candidate = predictions_by_model[candidate_model].copy()

    keep_cols = keys + ["true", "pred"] + [
        col
        for col in DEFAULT_ROBUSTNESS_SUBSETS.values()
        if col is not None and col in baseline.columns
    ]

    paired = baseline[keep_cols].merge(
        candidate[keys + ["pred"]],
        on=keys,
        how="inner",
        suffixes=(f"_{baseline_model}", f"_{candidate_model}"),
    )
    paired = paired.rename(
        columns={
            f"pred_{baseline_model}": "baseline_pred",
            f"pred_{candidate_model}": "candidate_pred",
        }
    )
    paired["baseline_abs_error"] = np.abs(paired["baseline_pred"] - paired["true"])
    paired["candidate_abs_error"] = np.abs(paired["candidate_pred"] - paired["true"])
    paired["candidate_mae_gain"] = (
        paired["baseline_abs_error"] - paired["candidate_abs_error"]
    )

    rows = []
    for subset_name, subset_col in subsets.items():
        if subset_col is None:
            subset = paired
        elif subset_col not in paired.columns:
            continue
        else:
            subset = paired[paired[subset_col].fillna(False).astype(bool)]

        rows.append(
            {
                "baseline_model": baseline_model,
                "candidate_model": candidate_model,
                "subset": subset_name,
                "n_samples": int(len(subset)),
                "baseline_mae": float(subset["baseline_abs_error"].mean()) if len(subset) else np.nan,
                "candidate_mae": float(subset["candidate_abs_error"].mean()) if len(subset) else np.nan,
                "candidate_mae_gain": float(subset["candidate_mae_gain"].mean()) if len(subset) else np.nan,
                "candidate_better_rate": float((subset["candidate_mae_gain"] > 0).mean()) if len(subset) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def collect_predictions(
    model,
    df: pd.DataFrame,
    loader,
    device,
    mode: str,
) -> Tuple[pd.DataFrame, float, float, float, float]:
    """Collect predictions from the three notebook model families.

    Parameters
    ----------
    mode:
        One of:
        - "visual": batch is visual_features, missing_flags, labels, weights, idx
        - "fusion": batch is visual_features, motion, heart_rate, visual_missing_flags, labels, weights, idx
        - "sensor": batch is motion, heart_rate, labels, weights, idx
    """
    model.eval()
    rows = []
    total_time = 0.0
    total_samples = 0

    import torch

    with torch.no_grad():
        for batch in loader:
            if mode == "visual":
                visual_features, missing_flags, labels, _sample_weights, idx = batch
                visual_features = visual_features.to(device)
                missing_flags = missing_flags.to(device)

                start = time.time()
                preds = model(visual_features, missing_flags)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end = time.time()

                batch_size = visual_features.size(0)

            elif mode == "fusion":
                (
                    visual_features,
                    motion,
                    heart_rate,
                    visual_missing_flags,
                    labels,
                    _sample_weights,
                    idx,
                ) = batch
                visual_features = visual_features.to(device)
                motion = motion.to(device)
                heart_rate = heart_rate.to(device)
                visual_missing_flags = visual_missing_flags.to(device)

                start = time.time()
                preds = model(visual_features, motion, heart_rate, visual_missing_flags)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end = time.time()

                batch_size = visual_features.size(0)

            elif mode == "sensor":
                motion, heart_rate, labels, _sample_weights, idx = batch
                motion = motion.to(device)
                heart_rate = heart_rate.to(device)

                start = time.time()
                preds = model(motion, heart_rate)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end = time.time()

                batch_size = motion.size(0)

            else:
                raise ValueError("mode must be one of: 'visual', 'fusion', 'sensor'.")

            total_time += end - start
            total_samples += int(batch_size)

            preds_np = preds.detach().cpu().numpy().reshape(-1)
            labels_np = labels.detach().cpu().numpy().reshape(-1)
            idx_np = idx.detach().cpu().numpy().reshape(-1)

            for pred, true, row_idx in zip(preds_np, labels_np, idx_np):
                source_row = df.iloc[int(row_idx)]
                row = {
                    "pred": float(pred),
                    "true": float(true),
                    "subject_experiment_id": source_row["subject_experiment_id"],
                    "subject_id": source_row["subject_id"],
                    "time_sec": source_row["time_sec"],
                    "attention_bin": source_row["attention_bin"],
                    "gender": source_row["gender"],
                    "age": source_row["age"],
                }
                if "age_group" in df.columns:
                    row["age_group"] = source_row["age_group"]
                rows.append(row)

    results_df = pd.DataFrame(rows)
    metrics = compute_prediction_metrics(results_df)
    latency_per_batch = total_time / max(len(loader), 1)
    latency_per_sample = total_time / max(total_samples, 1)

    return (
        results_df,
        metrics["mae"],
        metrics["rmse"],
        latency_per_batch,
        latency_per_sample,
    )


def make_mean_baseline_predictions(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
) -> pd.DataFrame:
    pred_value = float(train_df["target"].mean())

    out = eval_df.copy()
    out["true"] = out["target"].astype(float)
    out["pred"] = pred_value

    keep_cols = [
        "subject_experiment_id",
        "subject_id",
        "time_sec",
        "true",
        "pred",
        "attention_bin",
        "gender",
        "age",
        "age_group",
    ]
    keep_cols = [col for col in keep_cols if col in out.columns]
    return out[keep_cols].copy()


def compute_prediction_metrics(
    results_df: pd.DataFrame,
    tolerance: float = DEFAULT_TOLERANCE,
    binary_threshold: float = DEFAULT_BINARY_THRESHOLD,
) -> dict:
    y_true = results_df["true"].to_numpy(dtype=float)
    y_pred = results_df["pred"].to_numpy(dtype=float)

    binary_true = (y_true >= binary_threshold).astype(int)
    binary_pred = (y_pred >= binary_threshold).astype(int)

    try:
        binary_auc = roc_auc_score(binary_true, y_pred)
    except ValueError:
        binary_auc = np.nan

    return {
        "n_samples": int(len(results_df)),
        "mae": float(np.mean(np.abs(y_pred - y_true))),
        "rmse": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "r2": float(r2_score(y_true, y_pred)),
        "tolerant_accuracy": float(np.mean(np.abs(y_pred - y_true) <= tolerance)),
        "tolerance": float(tolerance),
        "one_off_accuracy": float(
            np.mean(np.abs(np.rint(y_pred) - np.rint(y_true)) <= 1)
        ),
        "binary_threshold": float(binary_threshold),
        "binary_accuracy": float(accuracy_score(binary_true, binary_pred)),
        "binary_f1": float(f1_score(binary_true, binary_pred, zero_division=0)),
        "binary_roc_auc": None if np.isnan(binary_auc) else float(binary_auc),
        "binary_confusion_matrix": confusion_matrix(binary_true, binary_pred).tolist(),
        "true_mean": float(np.mean(y_true)),
        "pred_mean": float(np.mean(y_pred)),
    }


def compute_group_mae(
    results_df: pd.DataFrame,
    group_col: str,
) -> Tuple[pd.Series, float, float]:
    df = results_df.copy()
    df["abs_error"] = np.abs(df["true"].astype(float) - df["pred"].astype(float))

    group_mae = (
        df.dropna(subset=[group_col])
        .groupby(group_col, observed=True)["abs_error"]
        .mean()
        .dropna()
    )

    if group_mae.empty:
        return group_mae, np.nan, np.nan

    worst_group = float(group_mae.max())
    gap = float(group_mae.max() - group_mae.min())
    return group_mae, worst_group, gap


def compute_fairness_metrics(results_df: pd.DataFrame) -> dict:
    metrics = {}

    if "gender" in results_df.columns:
        gender_mae, gender_worst, gender_gap = compute_group_mae(results_df, "gender")
        metrics["gender"] = {
            "mae_per_group": {
                str(k): round(float(v), 3) for k, v in gender_mae.items()
            },
            "worst_group_mae": round(float(gender_worst), 3),
            "gap": round(float(gender_gap), 3),
            "group_counts": {
                str(k): int(v)
                for k, v in results_df["gender"].value_counts(dropna=False).sort_index().items()
            },
        }

    if "age_group" in results_df.columns:
        age_mae, age_worst, age_gap = compute_group_mae(results_df, "age_group")
        metrics["age"] = {
            "mae_per_group": {
                str(k): round(float(v), 3) for k, v in age_mae.items()
            },
            "worst_group_mae": round(float(age_worst), 3),
            "gap": round(float(age_gap), 3),
            "group_counts": {
                str(k): int(v)
                for k, v in results_df["age_group"]
                .value_counts(sort=False, dropna=False)
                .items()
            },
        }

    return metrics


def compute_attention_bin_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
    if "attention_bin" not in results_df.columns:
        raise KeyError("results_df must include an 'attention_bin' column.")

    df = results_df.copy()
    df["abs_error"] = np.abs(df["true"].astype(float) - df["pred"].astype(float))

    return (
        df.groupby("attention_bin", observed=True)
        .agg(count=("abs_error", "size"), mae=("abs_error", "mean"))
        .reset_index()
        .sort_values("attention_bin")
    )


def evaluate_predictions(
    results_df: pd.DataFrame,
    model_name: str,
    tolerance: float = DEFAULT_TOLERANCE,
    binary_threshold: float = DEFAULT_BINARY_THRESHOLD,
) -> dict:
    prediction_metrics = compute_prediction_metrics(
        results_df,
        tolerance=tolerance,
        binary_threshold=binary_threshold,
    )
    fairness_metrics = compute_fairness_metrics(results_df)
    attention_bin_metrics = compute_attention_bin_metrics(results_df)

    return {
        "model": model_name,
        **prediction_metrics,
        "fairness": fairness_metrics,
        "attention_bin_mae": {
            str(row["attention_bin"]): {
                "count": int(row["count"]),
                "mae": round(float(row["mae"]), 3),
            }
            for _, row in attention_bin_metrics.iterrows()
        },
    }


def overall_table(
    predictions_by_model,
    tolerance: float = DEFAULT_TOLERANCE,
    binary_threshold: float = DEFAULT_BINARY_THRESHOLD,
) -> pd.DataFrame:
    rows = []
    for model_name, df in predictions_by_model.items():
        rows.append(
            {
                "model": model_name,
                **compute_prediction_metrics(
                    df,
                    tolerance=tolerance,
                    binary_threshold=binary_threshold,
                ),
            }
        )
    return pd.DataFrame(rows)


def fairness_table(predictions_by_model) -> pd.DataFrame:
    rows = []

    for model_name, df in predictions_by_model.items():
        if "gender" in df.columns:
            gender_mae, gender_worst, gender_gap = compute_group_mae(df, "gender")
            rows.append(
                {
                    "model": model_name,
                    "attribute": "gender",
                    "worst_group_mae": gender_worst,
                    "best_group_mae": float(gender_mae.min()) if not gender_mae.empty else np.nan,
                    "gap": gender_gap,
                    "mae_per_group": {
                        str(k): round(float(v), 3) for k, v in gender_mae.items()
                    },
                    "group_counts": {
                        str(k): int(v)
                        for k, v in df["gender"].value_counts(dropna=False).sort_index().items()
                    },
                }
            )

        if "age_group" in df.columns:
            age_mae, age_worst, age_gap = compute_group_mae(df, "age_group")
            rows.append(
                {
                    "model": model_name,
                    "attribute": "age_group",
                    "worst_group_mae": age_worst,
                    "best_group_mae": float(age_mae.min()) if not age_mae.empty else np.nan,
                    "gap": age_gap,
                    "mae_per_group": {
                        str(k): round(float(v), 3) for k, v in age_mae.items()
                    },
                    "group_counts": {
                        str(k): int(v)
                        for k, v in df["age_group"]
                        .value_counts(sort=False, dropna=False)
                        .items()
                    },
                }
            )

    return pd.DataFrame(rows)


def attention_bin_table(
    predictions_by_model,
) -> pd.DataFrame:
    frames = []
    for model_name, df in predictions_by_model.items():
        table = compute_attention_bin_metrics(df).copy()
        table.insert(0, "model", model_name)
        frames.append(table)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _bootstrap_metric_row(df: pd.DataFrame) -> dict:
    out = {}
    prediction_metrics = compute_prediction_metrics(df)
    out["mae"] = prediction_metrics["mae"]
    out["rmse"] = prediction_metrics["rmse"]
    out["r2"] = prediction_metrics["r2"]

    if "gender" in df.columns:
        _, gender_worst, gender_gap = compute_group_mae(df, "gender")
        out["gender_gap"] = gender_gap
        out["gender_worst_group_mae"] = gender_worst

    if "age_group" in df.columns:
        _, age_worst, age_gap = compute_group_mae(df, "age_group")
        out["age_gap"] = age_gap
        out["age_worst_group_mae"] = age_worst

    return out


def cluster_bootstrap_ci(
    results_df: pd.DataFrame,
    cluster_col: str = "subject_experiment_id",
    n_boot: int = 1000,
    ci: float = 95,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if cluster_col not in results_df.columns:
        raise KeyError(
            f"results_df must include '{cluster_col}' for cluster bootstrap."
        )

    rng = np.random.default_rng(seed)
    valid = results_df.dropna(subset=[cluster_col]).copy()
    clusters = valid[cluster_col].unique()

    grouped = {
        cluster: group
        for cluster, group in valid.groupby(cluster_col, sort=False)
    }

    boot_rows = []
    for _ in range(n_boot):
        sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        sampled_df = pd.concat(
            [grouped[cluster] for cluster in sampled_clusters],
            ignore_index=True,
        )
        boot_rows.append(_bootstrap_metric_row(sampled_df))

    boot_df = pd.DataFrame(boot_rows)
    alpha = (100 - ci) / 2

    summary = pd.DataFrame(
        {
            "mean": boot_df.mean(numeric_only=True),
            "ci_low": boot_df.quantile(alpha / 100, numeric_only=True),
            "ci_high": boot_df.quantile(1 - alpha / 100, numeric_only=True),
        }
    )

    return summary, boot_df


def bootstrap_table(
    predictions_by_model,
    cluster_col: str = "subject_experiment_id",
    n_boot: int = 1000,
    ci: float = 95,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    summaries = []
    samples = {}

    for i, (model_name, df) in enumerate(predictions_by_model.items()):
        summary, boot_df = cluster_bootstrap_ci(
            df,
            cluster_col=cluster_col,
            n_boot=n_boot,
            ci=ci,
            seed=seed + i,
        )
        summary = summary.reset_index(names="metric")
        summary.insert(0, "model", model_name)
        summaries.append(summary)
        samples[model_name] = boot_df

    return pd.concat(summaries, ignore_index=True), samples


def build_evaluation_tables(
    predictions_by_model,
    n_boot: int = 1000,
    cluster_col: str = "subject_experiment_id",
    seed: int = 42,
) -> dict:
    boot_summary, boot_samples = bootstrap_table(
        predictions_by_model,
        cluster_col=cluster_col,
        n_boot=n_boot,
        seed=seed,
    )

    return {
        "overall": overall_table(predictions_by_model),
        "fairness": fairness_table(predictions_by_model),
        "attention_bins": attention_bin_table(predictions_by_model),
        "bootstrap": boot_summary,
        "bootstrap_samples": boot_samples,
    }


def save_prediction_frame(
    results_df: pd.DataFrame,
    path: str,
) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    results_df.to_csv(path, index=False)


def flatten_results_for_csv(results: dict) -> dict:
    row = {}

    for key, value in results.items():
        if key in {"fairness", "attention_bin_mae", "bootstrap_ci"}:
            continue

        if isinstance(value, (dict, list, tuple)):
            row[key] = json.dumps(value, sort_keys=True, default=str)
        else:
            row[key] = value

    fairness = results.get("fairness", {})
    for attribute, metrics in fairness.items():
        prefix = attribute
        row[f"{prefix}_worst_group_mae"] = metrics.get("worst_group_mae")
        row[f"{prefix}_gap"] = metrics.get("gap")
        row[f"{prefix}_mae_per_group"] = json.dumps(
            metrics.get("mae_per_group", {}),
            sort_keys=True,
            default=str,
        )
        row[f"{prefix}_group_counts"] = json.dumps(
            metrics.get("group_counts", {}),
            sort_keys=True,
            default=str,
        )

    attention_bin_mae = results.get("attention_bin_mae", {})
    row["attention_bin_mae"] = json.dumps(
        attention_bin_mae,
        sort_keys=True,
        default=str,
    )

    bootstrap_ci = results.get("bootstrap_ci", {})
    row["bootstrap_ci"] = json.dumps(bootstrap_ci, sort_keys=True, default=str)

    for metric_name, metric_values in bootstrap_ci.items():
        for bound_name, bound_value in metric_values.items():
            row[f"bootstrap_{metric_name}_{bound_name}"] = bound_value

    return row


def append_experiment_results(
    results: dict,
    csv_path: str,
    jsonl_path: str,
) -> None:
    csv_dir = os.path.dirname(csv_path)
    jsonl_dir = os.path.dirname(jsonl_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    if jsonl_dir:
        os.makedirs(jsonl_dir, exist_ok=True)

    new_row = pd.DataFrame([flatten_results_for_csv(results)])

    if os.path.exists(csv_path):
        experiment_log = pd.read_csv(csv_path)
        experiment_log = pd.concat([experiment_log, new_row], ignore_index=True, sort=False)
    else:
        experiment_log = new_row

    experiment_log.to_csv(csv_path, index=False)

    with open(jsonl_path, "a") as f:
        f.write(json.dumps(results, sort_keys=True, default=str) + "\n")
