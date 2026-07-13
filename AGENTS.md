# Repository Guidelines

## Project Structure & Module Organization
This repository is a script-driven Python experiment pipeline for noise detection with complexity metrics. Root-level numbered scripts define the workflow:

- `01a_...` through `01e_...`: generate ECG, Lorenz, Roessler, Henon, and AR1 noisy signal datasets.
- `02_create_ML_dataset_restrctured_id.py`: builds rolling-window feature CSVs under `ML_tasks/`.
- `03_Run_ML.py`: trains CatBoost models and writes results under `ML_experiments/`.
- `04_plots_and_analysis.py` and `05_boxplots_lineplots.py`: create per-dataset and global plots under `Results/`.
- `06_XAI_Analysis.py`: post-hoc explainability analysis.
- `metrics.py`: shared complexity metric and utility functions.

Treat `catboost_info/`, `__pycache__/`, `ML_tasks/`, `ML_experiments/`, and `Results/` as generated outputs unless a change explicitly requires artifacts.

## Build, Test, and Development Commands
Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate Predictability_Estimation
```

Alternatively, run the command listed in `pip_install` inside a compatible Python environment.

Run the pipeline step by step from the repository root:

```bash
python 01e_create_AR1_data_restructured_seeded.py
python 02_create_ML_dataset_restrctured_id.py
python 03_Run_ML.py
python 04_plots_and_analysis.py
python 05_boxplots_lineplots.py
python 06_XAI_Analysis.py
```

Before running, update each script's `User settings` block, especially `USE_CASE`, `DATASET_NAME`, and `add`.

## Coding Style & Naming Conventions
Use standard Python style: 4-space indentation, `snake_case` for functions and variables, and `UPPER_CASE` for configuration constants. Keep the numbered filename convention. Preserve existing spellings such as `restrctured` and `restrcutured`, because scripts and documentation may reference them.

Prefer small helper functions in `metrics.py` for shared calculations. Avoid moving configuration blocks unless dependent scripts are updated together.

## Testing Guidelines
There is no formal test suite or pytest configuration. Validate changes by running the affected pipeline step and checking expected output files. For ML or analysis changes, run `02_create_ML_dataset_restrctured_id.py` and `03_Run_ML.py` for one dataset, then inspect CSV summaries and console metrics.

## Commit & Pull Request Guidelines
Recent commits use short descriptive summaries, for example `XAI Analyses added, plot saving unified` and `Update README.md`. Follow that concise, scoped style.

Pull requests should describe the affected pipeline step, dataset/use case tested, configuration values changed, and artifacts intentionally included. Link related issues or paper sections when relevant.

## Security & Configuration Tips
Do not commit local IDE settings, caches, trained model files, or large generated datasets unless they are required review artifacts. Keep machine-specific paths out of scripts and prefer repository-relative output paths.
