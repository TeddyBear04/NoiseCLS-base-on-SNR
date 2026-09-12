# Repository Guidelines

## Project Structure & Module Organization

The repository implements a 21-label audio-noise classification pipeline. `train.py` is the training and validation entry point, while `infer.py` runs prediction for one PCM16 WAV file. Reusable code lives in `noise_pipeline/`: `data.py` loads manifests and audio, and `model.py` defines `NoiseNet`. Dataset manifests, label maps, and split audio are under `21_labels_dataset/`. Store generated model weights in `checkpoints/`; `checkpoints/best.pt` is the default inference checkpoint. `README_IMPLEMENTATION.md` contains the baseline workflow, and `noise_classification_pipeline.png` documents the design visually.

## Build, Test, and Development Commands

Create and activate a local environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Use `python train.py --smoke-test --batch-size 2` for a quick end-to-end check. Run a bounded experiment with `python train.py --limit 1000 --epochs 3 --batch-size 8`, or full training with `python train.py --epochs 30 --batch-size 8 --workers 4`. For inference, run `python infer.py <path-to.wav>`; add `--threshold 0.3` or `--top-k 5` when needed.

## Coding Style & Naming Conventions

Follow standard Python conventions: four-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and `UPPER_CASE` for constants. Add type hints to public data-loading and model interfaces where practical. Use `pathlib.Path` for paths and keep CLI behavior behind `main()` and an `if __name__ == "__main__"` guard. No formatter or linter is configured, so keep imports grouped and changes PEP 8-compatible.

## Testing Guidelines

There is currently no dedicated automated test suite or coverage threshold. Before submitting changes, run the smoke-test command and confirm it completes training and validation without non-finite losses. For loader changes, exercise at least one file from each split. For inference changes, verify output against a known test WAV and the default checkpoint. Add future tests under `tests/` using names such as `test_data.py` and `test_model.py`.

## Commit & Pull Request Guidelines

Git history is not available in this directory, so use concise, imperative commit subjects (for example, `Fix PCM16 padding for short clips`). Keep commits focused. Pull requests should explain the motivation, list validation commands and results, note dataset or checkpoint impacts, and include metric comparisons for model changes. Attach screenshots only when changing diagrams or other visual artifacts; link related issues when applicable.

## Data & Artifact Safety

Do not rewrite manifest paths unnecessarily: the loader derives local WAV paths from `sample_id`. Avoid committing new large audio files, virtual environments, or experimental checkpoints unless they are explicitly required deliverables.
