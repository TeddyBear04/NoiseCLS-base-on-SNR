# Repository Guidelines

## Project Structure & Module Organization

The repository implements a 36-label single-label BEATs fine-tuning pipeline. Its primary entrypoint is `main.py`; source is organized in `config/`, `models/`, `noise_pipeline/`, `tasks/`, and `utils/`. Edit `config/train_config.json` to control training defaults. The dataset is `dataset/mix-dataset/` with `manifest.csv` and `labels.txt`; pretrained and fine-tuned weights live in `checkpoint/`; generated embedding caches live in `artifacts/`.

## Build, Test, and Development Commands

Create and activate a local environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Use `python main.py train --limit 1000 --epochs 3 --batch-size 8` for a bounded fine-tuning run. Run validation calibration with `python main.py evaluate`, test with `python main.py test`, and inference with `python main.py predict <path-to.wav>`.

## Coding Style & Naming Conventions

Follow standard Python conventions: four-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and `UPPER_CASE` for constants. Add type hints to public data-loading and model interfaces where practical. Use `pathlib.Path` for paths and keep CLI behavior behind `main()` and an `if __name__ == "__main__"` guard. No formatter or linter is configured, so keep imports grouped and changes PEP 8-compatible.

## Testing Guidelines

There is currently no dedicated automated test suite or coverage threshold. Before submitting changes, run a bounded BEATs training command and confirm it completes without non-finite losses. For loader changes, exercise at least one file from each split. For inference changes, verify output against a known test WAV and fine-tuned checkpoint. Add future tests under `tests/` using names such as `test_data.py` and `test_model.py`.

## Commit & Pull Request Guidelines

Git history is not available in this directory, so use concise, imperative commit subjects (for example, `Fix PCM16 padding for short clips`). Keep commits focused. Pull requests should explain the motivation, list validation commands and results, note dataset or checkpoint impacts, and include metric comparisons for model changes. Attach screenshots only when changing diagrams or other visual artifacts; link related issues when applicable.

## Data & Artifact Safety

Do not rewrite manifest paths unnecessarily: the loader derives local WAV paths from `sample_id`. Avoid committing new large audio files, virtual environments, or experimental checkpoints unless they are explicitly required deliverables.
