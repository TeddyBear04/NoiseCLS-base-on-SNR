"""
ML experiments runner with CatBoost and repeated group aware train and test splits

What this script does
Loads task datasets created in step 2 from ML_tasks/<Dataset>/exp_<add>/tasks/*.csv
Runs multiple train and test splits by signal_id with no leakage
Trains an out of the box CatBoostClassifier for each split
Saves per split outputs for evaluation and reproducibility
Prints per split results to the console
  confusion matrix without plot
  accuracy precision recall and f1
  classification report as text
Saves aggregated evaluation outputs with mean and standard deviation over splits together with a human readable TXT report
Optionally saves each trained model by default set to True
Saves exact row indices before and after optional balancing so posthoc XAI analysis can reconstruct the exact same datasets later

Classification tasks expected from step 2
Task 1 uses noise_label_task1 and is multi class
Task 2 uses noise_label_task2 and is multi class
Task 3 uses noise_present and is binary with 0 and 1

Notes
Only feature columns with complexity metrics are used. Identifiers such as signal_id and run_id are excluded.
Train and test splitting is performed on unique signal_id values at group level using a pure run_id split
  signal_id equals str(run_id)
  each split samples a random subset of run_ids for the test set with size floor(TEST_SIZE * n_runs)
  the training set is the complement
  all window rows belonging to a selected test run_id go to test and all others go to train with no leakage
Unique combination rule
  the test set is a combination of n_test run_ids chosen from n_runs total
  the maximum number of unique splits is C(n_runs, n_test). If N_SPLITS is larger an error is raised.
RNG control
  seeds are reset for each split with reseed_all(BASE_RANDOM_SEED + split_idx)
  the CatBoost random_seed is also set to BASE_RANDOM_SEED + split_idx
"""

from __future__ import annotations

import os
import json
import random
from dataclasses import dataclass
from math import comb
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from catboost import CatBoostClassifier



# =============================================================================
# User settings
# =============================================================================

# Must match step-2 outputs
DATASET_NAME = "Real"  # "ECG", "Lorenz", "AR1", ...
add = "ed10_td1_mc300"

ML_TASKS_ROOT = "ML_tasks"
ML_EXPERIMENTS_ROOT = "ML_experiments"

# repeated splits
N_SPLITS = 50
TEST_SIZE = 0.20
BASE_RANDOM_SEED = 42

# model saving
SAVE_MODELS = True

# CatBoost "out of the box"
CATBOOST_KWARGS: Dict[str, Any] = {
    "verbose": False,
    "random_seed": 42,
}

# =============================================================================
# Balancing (NEW: optional downsampling on TRAIN + TEST)
# =============================================================================
BALANCE_DATASETS = True  # True => downsample each non-minority class to minority count (TRAIN and TEST)

# =============================================================================
# Feature selection (explicit list of complexity metrics to keep)
# =============================================================================
FEATURES_TO_USE: List[str] = [
    "var1der",
    "var2der",
    "std_dev",
    "mad",
    "cv",
    "approximate_entropy",
    "permutation_entropy",
    "dfa",
    "hurst",
    "fisher_info",
    "sample_entropy",
    "lempel_ziv_complexity",
    "fisher_info_nk",
    "svd_entropy",
    "rel_decay",
    "svd_energy",
    "condition_number",
    "spectral_skewness",
]


# =============================================================================
# RNG helper (seed control)
# =============================================================================
def reseed_all(seed: int) -> None:
    """
    Reset all relevant RNGs to avoid leakage across splits.
    Note: PYTHONHASHSEED is only fully effective if set before Python starts.
    """
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed))


# =============================================================================
# Utilities
# =============================================================================
def ensure_dirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_task_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing task dataset: {path}")
    return pd.read_csv(path)


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Use FEATURES_TO_USE as the explicit feature list.
    Enforces existence and numeric dtype (coerces to numeric if needed).
    """
    missing = [c for c in FEATURES_TO_USE if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns in CSV: {missing}")

    # Coerce to numeric where needed (keeps NaN if conversion fails)
    for c in FEATURES_TO_USE:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    out = [c for c in FEATURES_TO_USE if c in df.columns]
    if not out:
        raise ValueError("No feature columns available after applying FEATURES_TO_USE.")
    return out


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, is_binary: bool) -> Dict[str, float]:
    avg = "binary" if is_binary else "macro"
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
    }


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def classification_report_table(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    rep = classification_report(y_true, y_pred, output_dict=True, digits=6, zero_division=0)
    return pd.DataFrame(rep).transpose()


def pick_unique_test_ids(
    all_ids: List[str],
    n_test: int,
    rng: np.random.Generator,
    seen: set,
) -> List[str]:
    """
    Sample a test-id subset (size n_test) without replacement, and enforce that this exact
    combination has not been used before.
    """
    if n_test <= 0:
        return []

    while True:
        test_ids = rng.choice(all_ids, size=n_test, replace=False).tolist()
        key = frozenset(test_ids)
        if key not in seen:
            seen.add(key)
            return sorted(test_ids)


def downsample_to_minority(
    X: pd.DataFrame,
    y: pd.Series,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """
    Downsample so every class has exactly the minority class count.

    Returns
    - Xb
    - yb
    - keep_idx as original row indices from the source dataframe

    Notes
    - If multiple classes tie for the minimum, they are all kept fully.
    - Every other class is randomly sampled down to that same minimum count.
    """
    counts = y.value_counts(dropna=False)
    if counts.empty:
        return X, y, X.index.to_numpy()

    min_count = int(counts.min())
    if min_count <= 0:
        return X, y, X.index.to_numpy()

    keep_idx_parts: List[np.ndarray] = []
    for cls, cnt in counts.items():
        idx = y.index[y == cls].to_numpy()
        if int(cnt) <= min_count:
            keep_idx_parts.append(idx)
        else:
            chosen = rng.choice(idx, size=min_count, replace=False)
            keep_idx_parts.append(chosen)

    keep_idx = np.concatenate(keep_idx_parts, axis=0)
    rng.shuffle(keep_idx)

    Xb = X.loc[keep_idx]
    yb = y.loc[keep_idx]
    return Xb, yb, keep_idx


# =============================================================================
# Experiment definitions
# =============================================================================
@dataclass(frozen=True)
class TaskSpec:
    task_key: str
    csv_filename: str
    label_col: str
    is_binary: bool


def get_tasks() -> List[TaskSpec]:
    """Only Task 1 (6-class noise type classification) for Real dataset."""
    return [
        TaskSpec(
            task_key="task1_noise_type",
            csv_filename=f"task1_noise_type_{add}.csv",
            label_col="noise_label_task1",
            is_binary=False,
        ),
    ]


# =============================================================================
# Core runner
# =============================================================================
def run_task_experiment(
    df: pd.DataFrame,
    feature_cols: List[str],
    spec: TaskSpec,
    out_dir: str,
    n_splits: int,
    test_size: float,
    base_seed: int,
    save_models: bool,
) -> None:
    ensure_dirs(out_dir)
    ensure_dirs(os.path.join(out_dir, "splits"))
    ensure_dirs(os.path.join(out_dir, "aggregate"))

    print("------------------------------------------------------------")
    print(f"Task: {spec.task_key}")
    print(f"Label: {spec.label_col}")
    print(f"Rows:  {len(df)}")
    print(f"Runs:  {df['signal_id'].nunique()}")
    print(f"Feats: {len(feature_cols)}")
    print(f"Splits: {n_splits} | test_size={test_size} | base_seed={base_seed}")
    print(f"Save models: {save_models}")
    print("CatBoost params:")
    for k, v in CATBOOST_KWARGS.items():
        print(f"  - {k}: {v}")
    print("------------------------------------------------------------")

    sid = df["signal_id"].astype(str)

    if spec.is_binary:
        labels_all = [0, 1]
    else:
        labels_all = sorted(df[spec.label_col].dropna().astype(str).unique().tolist())

    print(f"Labels ({len(labels_all)}): {labels_all}")

    all_ids = sorted(sid.unique().tolist())
    n_runs = len(all_ids)

    n_test = int(np.floor(float(test_size) * float(n_runs)))
    if n_runs >= 2:
        n_test = max(1, min(n_runs - 1, n_test))
    else:
        n_test = 0

    if n_test == 0:
        raise ValueError(f"Cannot create a test split with n_runs={n_runs} and TEST_SIZE={test_size}")

    max_unique = comb(n_runs, n_test)
    if n_splits > max_unique:
        raise ValueError(
            f"N_SPLITS={n_splits} exceeds max unique combinations C({n_runs},{n_test})={max_unique}."
        )

    split_metrics_rows: List[Dict[str, Any]] = []
    cm_list: List[np.ndarray] = []

    seen_test_sets: set = set()

    for split_idx in range(1, n_splits + 1):
        reseed_all(base_seed + split_idx)
        rng = np.random.default_rng(base_seed + split_idx)

        test_ids = pick_unique_test_ids(all_ids=all_ids, n_test=n_test, rng=rng, seen=seen_test_sets)
        test_set = set(test_ids)
        train_ids = [x for x in all_ids if x not in test_set]

        train_mask = sid.isin(train_ids)
        test_mask = sid.isin(test_ids)

        X_train = df.loc[train_mask, feature_cols]
        y_train = df.loc[train_mask, spec.label_col]

        X_test = df.loc[test_mask, feature_cols]
        y_test = df.loc[test_mask, spec.label_col]

        train_row_indices_before_balance = X_train.index.to_numpy()
        test_row_indices_before_balance = X_test.index.to_numpy()

        if BALANCE_DATASETS:
            X_train, y_train, train_row_indices_after_balance = downsample_to_minority(X_train, y_train, rng=rng)
            X_test, y_test, test_row_indices_after_balance = downsample_to_minority(X_test, y_test, rng=rng)
        else:
            train_row_indices_after_balance = train_row_indices_before_balance
            test_row_indices_after_balance = test_row_indices_before_balance

        split_folder = os.path.join(out_dir, "splits", f"split_{split_idx:03d}")
        ensure_dirs(split_folder)

        save_json(
            os.path.join(split_folder, "split_ids.json"),
            {"split": split_idx, "train_ids": train_ids, "test_ids": test_ids},
        )

        save_json(
            os.path.join(split_folder, "balanced_row_indices.json"),
            {
                "split": split_idx,
                "balanced_datasets": bool(BALANCE_DATASETS),
                "train_row_indices_before_balance": train_row_indices_before_balance.tolist(),
                "test_row_indices_before_balance": test_row_indices_before_balance.tolist(),
                "train_row_indices_after_balance": train_row_indices_after_balance.tolist(),
                "test_row_indices_after_balance": test_row_indices_after_balance.tolist(),
            },
        )

        print(
            f"[{split_idx:03d}/{n_splits}] train_runs={len(train_ids)} test_runs={len(test_ids)} "
            f"train_rows={len(X_train)} test_rows={len(X_test)}"
        )

        if len(X_train) == 0 or len(X_test) == 0:
            raise RuntimeError(
                "Empty train/test rows after split.\n"
                f"  unique signal_id (as strings): {all_ids}\n"
                f"  train_ids: {train_ids}\n"
                f"  test_ids: {test_ids}\n"
                f"  train_rows: {len(X_train)}  test_rows: {len(X_test)}"
            )

        cb_kwargs = dict(CATBOOST_KWARGS)
        cb_kwargs["random_seed"] = int(base_seed + split_idx)

        model = CatBoostClassifier(**cb_kwargs)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        if isinstance(y_pred, np.ndarray) and y_pred.ndim > 1:
            y_pred = y_pred.reshape(-1)

        cm = confusion_matrix(y_test, y_pred, labels=labels_all)
        cm_list.append(cm)

        cm_df = pd.DataFrame(
            cm,
            index=[f"true_{l}" for l in labels_all],
            columns=[f"pred_{l}" for l in labels_all],
        )
        cm_df.to_csv(os.path.join(split_folder, "confusion_matrix.csv"), index=True)

        rep_df = classification_report_table(y_test, y_pred)
        rep_df.to_csv(os.path.join(split_folder, "classification_report.csv"), index=True)
        save_json(os.path.join(split_folder, "classification_report.json"), rep_df.to_dict(orient="index"))

        m = compute_metrics(np.asarray(y_test), np.asarray(y_pred), is_binary=spec.is_binary)
        m_row = {
            "split": split_idx,
            "train_runs": len(train_ids),
            "test_runs": len(test_ids),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            **m,
        }
        split_metrics_rows.append(m_row)

        if save_models:
            model_path = os.path.join(split_folder, "catboost_model.cbm")
            model.save_model(model_path)

        print("Confusion matrix (rows=true, cols=pred):")
        print(cm_df.to_string())
        print("")
        print(
            f"Metrics (macro for multi-class; binary for Task3): "
            f"accuracy={m['accuracy']:.6f} precision={m['precision']:.6f} "
            f"recall={m['recall']:.6f} f1={m['f1']:.6f}"
        )
        print("")
        print("Classification report:")
        print(classification_report(y_test, y_pred, digits=6, zero_division=0))
        print("------------------------------------------------------------")

    metrics_df = pd.DataFrame(split_metrics_rows)
    metrics_df.to_csv(os.path.join(out_dir, "aggregate", "split_metrics.csv"), index=False)

    agg_metrics = {
        "n_splits": int(n_splits),
        "test_size": float(test_size),
        "accuracy_mean": float(metrics_df["accuracy"].mean()),
        "accuracy_std": float(metrics_df["accuracy"].std(ddof=1)),
        "precision_mean": float(metrics_df["precision"].mean()),
        "precision_std": float(metrics_df["precision"].std(ddof=1)),
        "recall_mean": float(metrics_df["recall"].mean()),
        "recall_std": float(metrics_df["recall"].std(ddof=1)),
        "f1_mean": float(metrics_df["f1"].mean()),
        "f1_std": float(metrics_df["f1"].std(ddof=1)),
    }
    save_json(os.path.join(out_dir, "aggregate", "metrics_summary.json"), agg_metrics)

    cm_stack = np.stack(cm_list, axis=0).astype(np.float64)
    cm_mean = cm_stack.mean(axis=0)
    cm_sum_rows = cm_mean.sum(axis=1, keepdims=True) + 1e-12
    cm_norm = cm_mean / cm_sum_rows

    cm_mean_df = pd.DataFrame(
        cm_mean,
        index=[f"true_{l}" for l in labels_all],
        columns=[f"pred_{l}" for l in labels_all],
    )
    cm_norm_df = pd.DataFrame(
        cm_norm,
        index=[f"true_{l}" for l in labels_all],
        columns=[f"pred_{l}" for l in labels_all],
    )

    cm_mean_df.to_csv(os.path.join(out_dir, "aggregate", "confusion_matrix_mean.csv"), index=True)
    cm_norm_df.to_csv(os.path.join(out_dir, "aggregate", "confusion_matrix_mean_normalized.csv"), index=True)

    report_lines = []
    report_lines.append(f"Task: {spec.task_key}")
    report_lines.append(f"Label column: {spec.label_col}")
    report_lines.append(f"Splits: {n_splits} (test_size={test_size})")
    report_lines.append(f"Runs: {df['signal_id'].nunique()} | Rows: {len(df)} | Features: {len(feature_cols)}")
    report_lines.append("")
    report_lines.append("Metrics (mean Â± std across splits)")
    report_lines.append(f"  Accuracy : {agg_metrics['accuracy_mean']:.6f} Â± {agg_metrics['accuracy_std']:.6f}")
    report_lines.append(f"  Precision: {agg_metrics['precision_mean']:.6f} Â± {agg_metrics['precision_std']:.6f}")
    report_lines.append(f"  Recall   : {agg_metrics['recall_mean']:.6f} Â± {agg_metrics['recall_std']:.6f}")
    report_lines.append(f"  F1       : {agg_metrics['f1_mean']:.6f} Â± {agg_metrics['f1_std']:.6f}")
    report_lines.append("")
    report_lines.append("Labels order (confusion matrices):")
    report_lines.append("  " + ", ".join(str(x) for x in labels_all))
    report_lines.append("")
    report_lines.append("Files written:")
    report_lines.append("  aggregate/split_metrics.csv")
    report_lines.append("  aggregate/metrics_summary.json")
    report_lines.append("  aggregate/confusion_matrix_mean.csv")
    report_lines.append("  aggregate/confusion_matrix_mean_normalized.csv")
    report_lines.append("  splits/split_XXX/split_ids.json")
    report_lines.append("  splits/split_XXX/balanced_row_indices.json")
    report_lines.append("  splits/split_XXX/catboost_model.cbm")
    report_lines.append("  splits/split_XXX/confusion_matrix.csv")
    report_lines.append("  splits/split_XXX/classification_report.csv")
    report_lines.append("  splits/split_XXX/classification_report.json")

    report_path = os.path.join(out_dir, "aggregate", "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("Task done")
    print(f"  Wrote: {report_path}")
    print("============================================================")


def main() -> None:
    tasks_dir = os.path.join(ML_TASKS_ROOT, DATASET_NAME, f"exp_{add}", "tasks")
    if not os.path.isdir(tasks_dir):
        raise FileNotFoundError(
            f"Tasks directory not found: {tasks_dir}\n"
            f"Run step 2 first to generate the task CSVs."
        )

    exp_root = os.path.join(ML_EXPERIMENTS_ROOT, DATASET_NAME, f"exp_{add}")
    ensure_dirs(exp_root)

    print("============================================================")
    print("ML experiments (CatBoost, repeated group-aware train/test splits)")
    print("------------------------------------------------------------")
    print(f"Dataset:      {DATASET_NAME}")
    print(f"add tag:       {add}")
    print(f"Tasks folder:  {tasks_dir}")
    print(f"Output folder: {exp_root}")
    print("")
    print(f"N_SPLITS:      {N_SPLITS}")
    print(f"TEST_SIZE:     {TEST_SIZE} (group-aware; run_id subset, no leakage)")
    print(f"BASE_SEED:     {BASE_RANDOM_SEED}")
    print(f"SAVE_MODELS:   {SAVE_MODELS}")
    print("")
    print("Tasks to run (sequential):")
    for t in get_tasks():
        print(f"  - {t.task_key}: {t.csv_filename} | label={t.label_col} | binary={t.is_binary}")
    print("============================================================")

    for spec in get_tasks():
        in_csv = os.path.join(tasks_dir, spec.csv_filename)
        df = load_task_csv(in_csv)

        feature_cols = get_feature_columns(df)

        task_out = os.path.join(exp_root, spec.task_key)
        run_task_experiment(
            df=df,
            feature_cols=feature_cols,
            spec=spec,
            out_dir=task_out,
            n_splits=N_SPLITS,
            test_size=TEST_SIZE,
            base_seed=BASE_RANDOM_SEED,
            save_models=SAVE_MODELS,
        )


if __name__ == "__main__":
    main()
