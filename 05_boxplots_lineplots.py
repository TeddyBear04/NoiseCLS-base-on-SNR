# =============================================================================
# Feature selection (explicit list of complexity metrics to keep)
# =============================================================================
FEATURES_TO_USE = [
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

"""
Global comparison of complexity-metric datasets (Step-2 ML task CSVs)
====================================================================

Purpose
-------
This script performs a cross-system, global-comparable analysis of the *complexity metrics*
that were computed during your Step-2 ML dataset creation (task CSV exports).

In your pipeline, Step 2 produces per-dataset task CSV files in:

    ML_tasks/<DATASET_NAME>/exp_<add>/tasks/*.csv

These CSVs already contain:
- extracted complexity metrics (feature columns)
- identifiers such as signal_id (group/run identifier)
- noise descriptors such as noise_type and noise_intensity (depending on task)
- task labels such as:
    * noise_label_task1  (noise category)
    * noise_label_task2  (noise type + intensity category)
    * noise_present      (binary noise/no-noise)
- and possibly extra bookkeeping columns (depending on your Step-2 export)

This script:
1) Discovers one "best" Step-2 task CSV per system/dataset (Henon, Lorenz, Roessler, ECG, AR1, ...).
   It prefers a CSV that contains the richest set of columns needed for the three analyses below.
2) Loads all systems into a single combined DataFrame and identifies numeric complexity metrics
   (by excluding known non-feature columns).
3) Applies a *global min-max scaling* (same scaling across all systems) for comparability.
4) Produces per-system visualizations under:
       Results/Global_Comparison/exp_<add>/<System>/
   including:
   - Task 1 (if available): boxplots per noise category (noise_label_task1)
   - Task 2 (if available): mean metric vs noise intensity, per noise type
   - Task 3 (if available): boxplots: noise present vs no noise
   All plots are saved as BOTH PNG and EPS.
5) Saves the globally scaled combined dataset to:
       Results/Global_Comparison/exp_<add>/all_systems_global_minmax_scaled_complexity.csv

Important notes / assumptions
-----------------------------
- This script does NOT recompute complexity metrics. It uses the Step-2 exported CSVs.
- The selection of the "best" CSV per system is heuristic:
  it searches within ML_tasks/<System>/exp_<add>/tasks/ and prefers:
    1) task2_noise_type_intensity_<add>.csv   (usually has noise_type + noise_intensity)
    2) task1_noise_type_<add>.csv
    3) task3_noise_present_<add>.csv
  If your filenames differ, adjust the scoring rules in score_dataset_candidate().
- The "Task 2" plot requires both noise_type and noise_intensity columns.
  If missing for a system, that part is skipped for that system.
- The script excludes non-feature columns via exclude_cols and exclude_metrics, then keeps numeric columns.

Outputs / folder structure
--------------------------
Results/Global_Comparison/exp_<add>/
  all_systems_global_minmax_scaled_complexity.csv
  <System>/
    <System>_boxplot_task1_<category>.png/.eps           (if task1 label exists)
    <System>_lines_task2_<noise_type>.png/.eps           (if noise_type & noise_intensity exist)
    <System>_boxplot_task3_noise.png/.eps
    <System>_boxplot_task3_no_noise.png/.eps
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# Plot styling switches (new)
# --------------------------------------------------------------
DPI = 500
FONT_SCALE = 1.2          # 1.0 keeps current sizes; >1 larger; <1 smaller
SHOW_TITLES = False        # if False: no plot titles anywhere
USE_CUSTOM_COLORS = False  # if True: apply palette to boxplots only (line plots stay unchanged)

CUSTOM_PALETTE = ["#188FA7", "#769FB6", "#9DBBAE", "#D5D6AA", "#E2DBBE"]


def _scaled(v: float) -> float:
    return float(v) * float(FONT_SCALE)


# --------------------------------------------------------------
# Global matplotlib font scaling (adapted)
# --------------------------------------------------------------
plt.rcParams.update({
    "font.size": _scaled(12),
    "axes.titlesize": _scaled(12),
    "axes.labelsize": _scaled(12),
    "xtick.labelsize": _scaled(10),
    "ytick.labelsize": _scaled(10),
    "legend.fontsize": _scaled(14),
})

# --------------------------------------------------------------
# 1. Configuration
# --------------------------------------------------------------
# Step-2 ML task exports live here:
ML_TASKS_ROOT = "ML_tasks"

# Must match your step-2 "exp_<add>" tag
add = "ed10_td1_mc300"

# Systems to include. Adjust freely.
base_folders = {
    "Henon": os.path.join(ML_TASKS_ROOT, "Henon", f"exp_{add}", "tasks"),
    "Lorenz": os.path.join(ML_TASKS_ROOT, "Lorenz", f"exp_{add}", "tasks"),
    "Roessler": os.path.join(ML_TASKS_ROOT, "Roessler", f"exp_{add}", "tasks"),
    "ECG": os.path.join(ML_TASKS_ROOT, "ECG", f"exp_{add}", "tasks"),
    "AR1": os.path.join(ML_TASKS_ROOT, "AR1", f"exp_{add}", "tasks"),
}

# Output root aligned with your existing analysis convention
RESULTS_ROOT = "Results"
results_folder = os.path.join(RESULTS_ROOT, "Global_Comparison", f"exp_{add}")
os.makedirs(results_folder, exist_ok=True)

# --------------------------------------------------------------
# 2. Dataset finder (generic, recursive) adapted for Step-2 tasks/
# --------------------------------------------------------------
def iter_csv_files(root_folder, max_depth=2):
    """
    Yield CSV file paths within root_folder up to max_depth subfolder levels.
    max_depth=0 means only the root folder.
    """
    root_folder = os.path.abspath(root_folder)
    root_depth = root_folder.rstrip(os.sep).count(os.sep)

    for dirpath, dirnames, filenames in os.walk(root_folder):
        depth = dirpath.rstrip(os.sep).count(os.sep) - root_depth
        if depth > max_depth:
            dirnames[:] = []
            continue

        for fn in filenames:
            if fn.lower().endswith(".csv"):
                yield os.path.join(dirpath, fn)


def score_dataset_candidate(path, system_name):
    """
    Score a CSV path as a dataset candidate.
    Higher score = more likely the "right" Step-2 complexity dataset.

    For your Step-2 exports, we prefer task2 > task1 > task3.
    We also prefer files that contain the add-tag.
    """
    name = os.path.basename(path).lower()
    sys = system_name.lower()

    score = 0

    # Strong positives: task CSVs by convention
    if "task2_noise_type_intensity" in name:
        score += 50
    if "task1_noise_type" in name:
        score += 30
    if "task3_noise_present" in name:
        score += 20

    # Prefer matching exp-tag
    if add.lower() in name:
        score += 10

    # Mild preference if system name appears
    if sys in name:
        score += 2

    # Size tie-breaker nudge
    try:
        score += min(os.path.getsize(path) / 1_000_000, 5)
    except OSError:
        pass

    return score


def find_complexity_dataset(folder, system_name, max_depth=2):
    """
    Find the most likely Step-2 task CSV for a system.
    Searches root folder and subfolders up to max_depth.
    """
    if not os.path.exists(folder):
        print(f"Folder not found: {folder}")
        return None

    candidates = []
    for p in iter_csv_files(folder, max_depth=max_depth):
        fname = os.path.basename(p).lower()
        # in Step-2 tasks folder: accept task*.csv primarily
        if not fname.endswith(".csv"):
            continue
        candidates.append(p)

    if not candidates:
        print(f"No CSV found in {folder}")
        return None

    scored = [(score_dataset_candidate(p, system_name), p) for p in candidates]
    scored.sort(
        key=lambda x: (x[0], os.path.getsize(x[1]) if os.path.exists(x[1]) else 0),
        reverse=True
    )

    best_score, best_path = scored[0]
    print(f"Using Step-2 ML task dataset for {system_name}: {best_path} (score={best_score:.2f})")
    return best_path


datasets = {
    name: find_complexity_dataset(folder, name, max_depth=1)
    for name, folder in base_folders.items()
}
datasets = {k: v for k, v in datasets.items() if v is not None}

if not datasets:
    raise FileNotFoundError(
        "No Step-2 task CSVs found. Check ML_tasks/<System>/exp_<add>/tasks/ paths and filenames."
    )

# --------------------------------------------------------------
# 3. Load all datasets
# --------------------------------------------------------------
dfs = []
for system, path in datasets.items():
    df = pd.read_csv(path)
    df["system"] = system
    dfs.append(df)
    print(f"Loaded {system}: {df.shape} from {path}")

all_df = pd.concat(dfs, ignore_index=True)
print(f"Combined dataset shape: {all_df.shape}")

# --------------------------------------------------------------
# 4. Identify complexity metrics
# --------------------------------------------------------------
exclude_cols = [
    # common bookkeeping / identifiers / labels
    "timestep", "timestep_size", "dt", "time", "t_eval",
    "noise_type", "noise_intensity",
    "noise_label_task1", "noise_label_task2",
    "signal_id", "run_id", "window_id",
    "noise_present", "system",
    # sometimes present in your exports
    "split", "rep", "npz_path"
]

exclude_metrics = {"corr_dim", "higuchi_fd"}

complexity_metrics = [
    c for c in all_df.columns
    if c not in exclude_cols
    and c not in exclude_metrics
    and pd.api.types.is_numeric_dtype(all_df[c])
]

# --- NEW: explicit feature selection (keep only FEATURES_TO_USE that are present & numeric) ---
features_to_use_present = [f for f in FEATURES_TO_USE if f in complexity_metrics]
missing_features = [f for f in FEATURES_TO_USE if f not in complexity_metrics]

complexity_metrics = features_to_use_present

print(f"Complexity metrics ({len(complexity_metrics)}):")
print(", ".join(complexity_metrics))

if missing_features:
    print("Missing requested features (not found / not numeric in combined CSVs):")
    print(", ".join(missing_features))

if not complexity_metrics:
    raise ValueError(
        "After applying FEATURES_TO_USE, no numeric complexity metrics remain. "
        "Check FEATURES_TO_USE names vs CSV columns."
    )

# --------------------------------------------------------------
# 5. Global min–max scaling
# --------------------------------------------------------------
global_min = all_df[complexity_metrics].min()
global_max = all_df[complexity_metrics].max()

scaled_df = all_df.copy()
scaled_df[complexity_metrics] = (
    scaled_df[complexity_metrics] - global_min
) / (global_max - global_min + 1e-12)

print("Applied global min–max scaling.")

# --------------------------------------------------------------
# 6. Boxplot helper (PNG + EPS)
# --------------------------------------------------------------
def save_boxplot(sub_df, metrics, title, out_base_no_ext):
    if sub_df.empty:
        return

    plt.figure(figsize=(max(10, len(metrics) * 0.6), 6))
    bp = plt.boxplot(
        [sub_df[m].dropna() for m in metrics],
        labels=metrics,
        patch_artist=True,
        medianprops=dict(color="black")
    )

    # Apply custom palette to boxplots only (new)
    if USE_CUSTOM_COLORS:
        for i, box in enumerate(bp.get("boxes", [])):
            box.set_facecolor(CUSTOM_PALETTE[i % len(CUSTOM_PALETTE)])
            box.set_alpha(0.85)

    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Global scaled value (0–1)")

    if SHOW_TITLES:
        plt.title(title)

    plt.grid(alpha=0.3)

    plt.savefig(out_base_no_ext + ".png", dpi=DPI, bbox_inches="tight")
    plt.savefig(out_base_no_ext + ".eps", bbox_inches="tight")
    plt.close()

    print(f"Saved boxplot: {out_base_no_ext}.png/.eps")


# --------------------------------------------------------------
# 7. Per-system visualizations
# --------------------------------------------------------------
# Line-plot colors remain unchanged by design (do NOT apply custom palette here).
colors = list(plt.cm.tab20.colors)
linestyles = ["-", "--", "-.", ":", (0, (5, 2)), (0, (3, 5, 1, 5))]
markers = ["o", "s", "D", "^", "v", "p", "*", "X", "P", "h"]

for system, sys_df in scaled_df.groupby("system"):
    sys_folder = os.path.join(results_folder, system)
    os.makedirs(sys_folder, exist_ok=True)

    print(f"Processing system: {system}")

    # Task 1: Noise category boxplots
    if "noise_label_task1" in sys_df.columns:
        for cat in sorted(sys_df["noise_label_task1"].dropna().unique()):
            sub = sys_df[sys_df["noise_label_task1"] == cat]
            outb = os.path.join(sys_folder, f"{system}_boxplot_task1_{cat}")
            save_boxplot(sub, complexity_metrics, f"{system}_task1_noise_type_{cat}", outb)
    else:
        print(f"Task 1 skipped for {system} (missing noise_label_task1).")

    # Task 2: Mean metric vs noise intensity (per noise type)
    # NOTE: this plot intentionally does NOT use the custom palette.
    if {"noise_type", "noise_intensity"}.issubset(sys_df.columns):
        noise_types = [n for n in sys_df["noise_type"].dropna().unique() if n != "none"]
        baseline = sys_df[sys_df["noise_type"] == "none"][complexity_metrics].mean().to_dict()

        for nt in noise_types:
            sub = sys_df[sys_df["noise_type"] == nt]
            if sub.empty:
                continue

            means = (
                sub.groupby("noise_intensity")[complexity_metrics]
                .mean()
                .reset_index()
                .sort_values("noise_intensity")
            )

            base_row = {"noise_intensity": 0.0}
            for m in complexity_metrics:
                base_row[m] = baseline.get(m, np.nan)

            means = pd.concat([pd.DataFrame([base_row]), means], ignore_index=True).sort_values("noise_intensity")

            # --- main line plot (NO legend) ---
            fig, ax = plt.subplots(figsize=(10, 6))
            lines = []
            labels = []
            for i, m in enumerate(complexity_metrics):
                (ln,) = ax.plot(
                    means["noise_intensity"], means[m],
                    label=m,
                    color=colors[i % len(colors)],
                    linestyle=linestyles[i % len(linestyles)],
                    marker=markers[i % len(markers)],
                    linewidth=1.3,
                    markersize=4,
                    alpha=0.9
                )
                lines.append(ln)
                labels.append(m)

            ax.set_xlabel("Noise intensity (0 = no noise)")
            ax.set_ylabel("Global scaled value (0-1)")
            if SHOW_TITLES:
                ax.set_title(f"{system}_task2_noise_type_intensity_{nt}")
            ax.grid(alpha=0.35)

            outb = os.path.join(sys_folder, f"{system}_lines_task2_{nt}")
            fig.savefig(outb + ".png", dpi=DPI, bbox_inches="tight")
            fig.savefig(outb + ".eps", bbox_inches="tight")
            plt.close(fig)
            print(f"Saved line plot: {outb}.png/.eps")

            # --- legend-only plot (separate PNG + EPS) ---
            fig_leg = plt.figure(figsize=(10, 2.5))
            fig_leg.legend(
                handles=lines,
                labels=labels,
                fontsize=_scaled(10),
                ncol=4,
                frameon=False,
                loc="center"
            )
            fig_leg.tight_layout()

            outb_leg = os.path.join(sys_folder, f"{system}_lines_task2_{nt}_legend")
            fig_leg.savefig(outb_leg + ".png", dpi=DPI, bbox_inches="tight")
            fig_leg.savefig(outb_leg + ".eps", bbox_inches="tight")
            plt.close(fig_leg)
            print(f"Saved legend: {outb_leg}.png/.eps")
    else:
        print(f"Task 2 skipped for {system} (missing noise_type or noise_intensity).")

    # Task 3: Noise vs no-noise boxplots
    if "noise_type" in sys_df.columns:
        tmp = sys_df.copy()
        tmp["noise_present"] = (tmp["noise_type"] != "none").astype(int)

        noise_df = tmp[tmp["noise_present"] == 1]
        clean_df = tmp[tmp["noise_present"] == 0]

        save_boxplot(
            noise_df,
            complexity_metrics,
            f"{system}_task3_noise_present",
            os.path.join(sys_folder, f"{system}_boxplot_task3_noise")
        )

        save_boxplot(
            clean_df,
            complexity_metrics,
            f"{system}_task3_noise_present",
            os.path.join(sys_folder, f"{system}_boxplot_task3_no_noise")
        )
    else:
        print(f"Task 3 skipped for {system} (missing noise_type).")

# --------------------------------------------------------------
# 8. Save globally scaled dataset
# --------------------------------------------------------------
scaled_path = os.path.join(results_folder, "all_systems_global_minmax_scaled_complexity.csv")
scaled_df.to_csv(scaled_path, index=False)

print(f"Saved globally scaled dataset to {scaled_path}")
print("Global-comparable complexity analysis complete.")