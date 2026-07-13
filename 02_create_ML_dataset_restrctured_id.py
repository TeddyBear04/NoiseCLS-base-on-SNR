"""
ML dataset construction (step 2)

Loads step-1 generated runs (.npz + metadata/index.csv), slices each run into rolling windows
(memory_complexity=300), computes complexity-metric features per window, adds labels and a
single run identifier, and writes:
- full feature dataset
- task-specific datasets (Task1/Task2/Task3)

Important identifier rule
-------------------------
For every ML sample (i.e., every rolling window row), the ONLY identifier kept is the run id.
To preserve downstream compatibility, this identifier is stored in the column `signal_id`
(as a string). No separate `run_id` column is written.

Embedding / feature-type separation (DYN vs SVD)
------------------------------------------------
Some features are computed from a delay embedding and are conceptually split into:
- DYN (dynamical/geometry-of-trajectory) features: computed on an embedding with
  dimension `embedding_dimension_DYN` and time delay `time_delay_DYN`
- SVD (singular-value / spectrum-of-embedding) features: computed on an embedding with
  dimension `embedding_dimension_SVD` and time delay `time_delay_SVD`

This script therefore computes two embeddings per window segment:
- embedded_DYN = delay_embed(segment, embedding_dimension=embedding_dimension_DYN, time_delay=time_delay_DYN)
- embedded_SVD = delay_embed(segment, embedding_dimension=embedding_dimension_SVD, time_delay=time_delay_SVD)

Segment-only features (entropy, DFA, Hurst, etc.) remain computed directly on the raw segment.

This script does NOT train models. It only builds datasets.
"""

from __future__ import annotations

import os
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import neurokit2 as nk

# your metrics file must be on PYTHONPATH or in the same folder
from metrics import *  # delay_embed, Fisher, PE, SampEn, LZ, etc.


# =============================================================================
# User settings
# =============================================================================

# -----------------------------
# Use-case selection (0..4)
# 0: Roessler, 1: ECG, 2: Lorenz, 3: Henon, 4: AR1
# -----------------------------
USE_CASE = 5  # 5 = Real-world noise dataset (6 environments)

USE_CASES: Dict[int, Dict[str, str]] = {
    0: {
        "name": "Roessler",
        "data_base_folder": "Roessler_Noise_Exp",
        "folder_suffix": "_roessler_data",
        "file_prefix": "roessler",
    },
    1: {
        "name": "ECG",
        "data_base_folder": "ECG_Noise_Exp",
        "folder_suffix": "_ecg_data",
        "file_prefix": "ecg",
    },
    2: {
        "name": "Lorenz",
        "data_base_folder": "Lorenz_Noise_Exp",
        "folder_suffix": "_lorenz_data",
        "file_prefix": "lorenz",
    },
    3: {
        "name": "Henon",
        "data_base_folder": "Henon_Noise_Exp",
        "folder_suffix": "_henon_data",
        "file_prefix": "henon",
    },
    4: {
        "name": "AR1",
        "data_base_folder": "AR1_Noise_Exp",
        "folder_suffix": "_ar1_data",
        "file_prefix": "ar1",
    },
    5: {
        "name": "Real",
        "data_base_folder": "Real_Noise_Exp",
        "folder_suffix": "_real_data",
        "file_prefix": "real",
    },
}

if USE_CASE not in USE_CASES:
    raise ValueError(f"USE_CASE must be one of {sorted(USE_CASES.keys())}, got {USE_CASE}")

DATASET_NAME = USE_CASES[USE_CASE]["name"]
DATA_BASE_FOLDER = USE_CASES[USE_CASE]["data_base_folder"]
DATASET_FOLDER_SUFFIX = USE_CASES[USE_CASE]["folder_suffix"]
FILE_PREFIX = USE_CASES[USE_CASE]["file_prefix"]

# This string is appended to output folder and filenames (e.g., embedding dimension version tag)
add = "ed10_td1_mc300"

# Output folder root for ML tasks
ML_TASKS_ROOT = "ML_tasks"

# Feature extraction settings (match the old script)
embedding_dimension_SVD = 10
embedding_dimension_DYN = 3

time_delay_SVD = 1
time_delay_DYN = 1

memory_complexity = 300

# Sliding-window step size (stride between successive window end indices)
window_step = 300  # default 100, increased for speed on large real dataset

# If you want to explicitly point to a specific dataset root folder, set it here.
# Otherwise, the script auto-selects the latest dataset folder under DATA_BASE_FOLDER.
DATASET_ROOT_OVERRIDE = ""  # e.g., r"ECG_Noise_Exp\20260101_120000_ecg_data"


# =============================================================================
# Folder utilities
# =============================================================================

def find_latest_dataset_root(base_folder: str) -> str:
    """
    Pick the newest dataset folder matching the selected use-case suffix under base_folder.
    Assumes folder naming starts with sortable timestamp 'YYYYmmdd_HHMMSS'.
    """
    candidates: List[str] = []
    for name in os.listdir(base_folder):
        full = os.path.join(base_folder, name)
        if os.path.isdir(full) and name.endswith(DATASET_FOLDER_SUFFIX):
            candidates.append(full)

    if not candidates:
        raise FileNotFoundError(
            f"No '*{DATASET_FOLDER_SUFFIX}' folders found in '{base_folder}'. "
            f"Run step-1 data generation first for dataset '{DATASET_NAME}'."
        )

    # sort by folder name (timestamp prefix)
    candidates_sorted = sorted(candidates, key=lambda p: os.path.basename(p))
    return candidates_sorted[-1]


def ensure_ml_dirs(root: str) -> Dict[str, str]:
    paths = {
        "root": root,
        "datasets": os.path.join(root, "datasets"),
        "tasks": os.path.join(root, "tasks"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


# =============================================================================
# Metric helpers
# =============================================================================

def calculate_fisher_information_nk(time_series: np.ndarray, delay: int = 1, dimension: int = 3) -> float:
    fi, _ = nk.fisher_information(time_series, delay=delay, dimension=dimension)
    return fi


def safe_scalar(val: Any) -> float:
    if isinstance(val, (list, tuple, np.ndarray)):
        try:
            return float(np.ravel(val)[0])
        except Exception:
            return np.nan
    try:
        return float(val)
    except Exception:
        return np.nan


def extract_complexity_metrics(
    signal: np.ndarray,
    t_eval: np.ndarray,
    emb_dim_svd: int = embedding_dimension_SVD,
    emb_dim_dyn: int = embedding_dimension_DYN,
    delay_svd: int = time_delay_SVD,
    delay_dyn: int = time_delay_DYN,
) -> pd.DataFrame:
    """
    Rolling-window feature extraction.
    For each i in [memory_complexity, len(signal)-1]:
      segment = signal[i-memory_complexity : i]
      compute features on segment and embeddings(segment)
    """
    rows: List[Dict[str, Any]] = []
    fail_counts: Dict[str, int] = {}

    n = len(signal)
    if n != len(t_eval):
        raise ValueError(f"signal and t_eval length mismatch: {n} vs {len(t_eval)}")

    print(
        f"    feature extraction: n={n}, memory_complexity={memory_complexity}, "
        f"emb_dim_dyn={emb_dim_dyn}, delay_dyn={delay_dyn}, "
        f"emb_dim_svd={emb_dim_svd}, delay_svd={delay_svd}"
    )

    for i in range(memory_complexity, n, window_step):
        segment = np.asarray(signal[i - memory_complexity : i], dtype=float)
        timestamp = float(t_eval[i])

        if not np.isfinite(segment).all():
            continue

        # embeddings (separate for DYN and SVD)
        try:
            embedded_DYN = delay_embed(segment, embedding_dimension=emb_dim_dyn, time_delay=delay_dyn)
            embedded_SVD = delay_embed(segment, embedding_dimension=emb_dim_svd, time_delay=delay_svd)
        except Exception as e:
            fail_counts[type(e).__name__] = fail_counts.get(type(e).__name__, 0) + 1
            continue

        try:
            vals = {
                "timestep": timestamp,
                "window_end_idx": i,
                "window_start_idx": i - memory_complexity,

                "var1der": safe_scalar(calculate_variance_1st_derivative(embedded_DYN)),  # DYN
                "var2der": safe_scalar(calculate_variance_2nd_derivative(embedded_DYN)),  # DYN
                "std_dev": safe_scalar(np.std(segment)),
                "mad": safe_scalar(np.mean(np.abs(segment - np.mean(segment)))),
                "cv": safe_scalar(np.std(segment) / (np.mean(np.abs(segment)) + 1e-12)),

                "approximate_entropy": safe_scalar(nk.entropy_approximate(segment)),
                "permutation_entropy": safe_scalar(permutation_entropy_metric(segment, dimension=embedding_dimension_DYN, delay=delay_dyn)),
                "dfa": safe_scalar(nk.fractal_dfa(segment)),
                "hurst": safe_scalar(calculate_hurst(segment)),
                "fisher_info": safe_scalar(calculate_fisher_information(segment)),
                "sample_entropy": safe_scalar(sample_entropy_metric(segment)),
                "lempel_ziv_complexity": safe_scalar(lempel_ziv_complexity_metric(segment)),

                "fisher_info_nk": safe_scalar(calculate_fisher_information_nk(segment, delay=delay_svd, dimension=emb_dim_svd)),  # SVD
                "svd_entropy": safe_scalar(nk.entropy_svd(segment, dimension=emb_dim_svd)),
                "rel_decay": safe_scalar(relative_decay_singular_values(embedded_SVD)),  # SVD
                "svd_energy": safe_scalar(svd_energy(embedded_SVD, k=3)),  # SVD
                "condition_number": safe_scalar(condition_number(embedded_SVD)),  # SVD
                "spectral_skewness": safe_scalar(spectral_skewness(embedded_SVD)),  # SVD
            }
            rows.append(vals)

        except Exception as e:
            fail_counts[type(e).__name__] = fail_counts.get(type(e).__name__, 0) + 1
            continue

    df = pd.DataFrame(rows)

    if fail_counts:
        top = sorted(fail_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
        print(f"    metric failures (top): {top}")

    print(f"    extracted rows: {len(df)}")
    return df


# =============================================================================
# Dataset construction
# =============================================================================

def load_index_csv(dataset_root: str) -> pd.DataFrame:
    index_path = os.path.join(dataset_root, "metadata", "index.csv")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Missing metadata index.csv at: {index_path}")
    return pd.read_csv(index_path)


def load_signal_from_npz(dataset_root: str, rel_npz_path: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    For 'noisy' split, uses stored 'noisy' array.
    For 'clean' split, uses stored 'clean_norm' array.
    """
    npz_path = os.path.join(dataset_root, rel_npz_path)
    data = np.load(npz_path)

    t_eval = data["t_eval"]

    if split == "noisy":
        signal = data["noisy"]
    elif split == "clean":
        signal = data["clean_norm"]
    else:
        raise ValueError(f"Unknown split: {split}")

    return t_eval.astype(np.float64), signal.astype(np.float64)


def build_full_feature_dataset(dataset_root: str) -> pd.DataFrame:
    """
    Loads all entries from index.csv and builds one row per rolling window.

    Adds:
      - split, noise_type, noise_intensity
      - noise_label_task1, noise_label_task2, noise_present
      - signal_id  (THIS IS THE ONLY IDENTIFIER; equals str(run_id))
    """
    idx = load_index_csv(dataset_root)

    required_cols = {"split", "noise_type", "noise_intensity", "run_id", "npz_path"}
    missing = required_cols.difference(set(idx.columns))
    if missing:
        raise ValueError(f"index.csv missing columns: {sorted(missing)}")

    all_rows: List[pd.DataFrame] = []

    print(f"[1/4] Load index.csv: {len(idx)} entries")
    print(f"      dataset_root={dataset_root}")

    for k, row in enumerate(idx.itertuples(index=False), start=1):
        split = str(row.split)
        noise_type = str(row.noise_type)
        noise_intensity = float(row.noise_intensity)
        run_id = int(row.run_id)
        rel_npz = str(row.npz_path)

        # ONLY identifier used downstream
        signal_id = str(run_id)

        print(f"[2/4] Entry {k}/{len(idx)}: run_id={run_id:04d} split={split} label={noise_type} (Task 1: 6-class. SNR={noise_intensity:.2f}dB is metadata only)")

        t_eval, signal = load_signal_from_npz(dataset_root, rel_npz, split=split)

        feats = extract_complexity_metrics(
            signal=signal,
            t_eval=t_eval,
            emb_dim_svd=embedding_dimension_SVD,
            emb_dim_dyn=embedding_dimension_DYN,
            delay_svd=time_delay_SVD,
            delay_dyn=time_delay_DYN,
        )

        # labels and identifier (one per window row)
        feats["split"] = split
        feats["noise_type"] = noise_type
        feats["noise_intensity"] = noise_intensity
        feats["noise_label_task1"] = noise_type
        feats["noise_label_task2"] = f"{noise_type}_{noise_intensity}" if noise_type != "none" else "none_0"
        feats["noise_present"] = 0 if noise_type == "none" else 1

        # single identifier
        feats["signal_id"] = signal_id

        all_rows.append(feats)

    print("[3/4] Concatenate feature tables")
    full_df = pd.concat(all_rows, ignore_index=True)

    # stable ordering (useful for later group-splitting)
    full_df = full_df.sort_values(by=["noise_label_task2", "signal_id", "timestep"], ignore_index=True)

    print(f"[4/4] Full dataset shape: {full_df.shape}")
    return full_df


def write_task_datasets(full_df: pd.DataFrame, out_root: str, add_tag: str) -> None:
    """
    Writes:
      - full dataset
      - Task 1 dataset
      - Task 2 dataset (skipped for USE_CASE == 5 / Real)
      - Task 3 dataset (skipped for USE_CASE == 5 / Real)
    """
    paths = ensure_ml_dirs(out_root)

    # full dataset
    full_path = os.path.join(paths["datasets"], f"{FILE_PREFIX}_full_features_{add_tag}.csv")
    print(f"Save full dataset: {full_path}")
    full_df.to_csv(full_path, index=False)

    # Task 1
    task1 = full_df.copy()
    task1_path = os.path.join(paths["tasks"], f"task1_noise_type_{add_tag}.csv")
    print(f"Save Task 1 dataset: {task1_path}")
    task1.to_csv(task1_path, index=False)

    # Task 2 and Task 3 are skipped for Real dataset
    if USE_CASE == 5:
        print("Skipping Task 2 and Task 3 (not applicable for real-world dataset)")
        return

    # Task 2
    task2 = full_df.copy()
    task2_path = os.path.join(paths["tasks"], f"task2_noise_type_intensity_{add_tag}.csv")
    print(f"Save Task 2 dataset: {task2_path}")
    task2.to_csv(task2_path, index=False)

    # Task 3
    task3 = full_df.copy()
    task3_path = os.path.join(paths["tasks"], f"task3_noise_present_{add_tag}.csv")
    print(f"Save Task 3 dataset: {task3_path}")
    task3.to_csv(task3_path, index=False)


def main() -> None:
    # select dataset root
    if DATASET_ROOT_OVERRIDE.strip():
        dataset_root = DATASET_ROOT_OVERRIDE.strip()
        if not os.path.isdir(dataset_root):
            raise FileNotFoundError(f"DATASET_ROOT_OVERRIDE not found: {dataset_root}")
    else:
        dataset_root = find_latest_dataset_root(DATA_BASE_FOLDER)

    # output folder: ML_tasks/<DatasetName>/exp_{add}
    out_root = os.path.join(ML_TASKS_ROOT, DATASET_NAME, f"exp_{add}")
    os.makedirs(out_root, exist_ok=True)

    print("============================================================")
    print(f"Step 2: Build ML datasets ({DATASET_NAME})")
    print("------------------------------------------------------------")
    print(f"USE_CASE: {USE_CASE} ({DATASET_NAME})")
    print(f"add tag: {add}")
    print(f"dataset_root: {dataset_root}")
    print(f"output_root:  {out_root}")
    print("============================================================")

    full_df = build_full_feature_dataset(dataset_root)
    write_task_datasets(full_df, out_root, add_tag=add)

    print("Done")


if __name__ == "__main__":
    main()
