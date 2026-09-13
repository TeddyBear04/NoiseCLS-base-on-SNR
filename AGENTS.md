# Repository Guidelines

## Project Structure & Module Organization

The repository implements a 36-label single-label BEATs fine-tuning pipeline
with a mixture branch and a separated/amplified-noise branch fused before
classification. Its primary entrypoint is `main.py`; source is organized in
`config/`, `models/` (`separator.py` for the noise separator, `fusion.py` for
the shared-encoder fusion classifier, `beats/` for the vendored backbone),
`noise_pipeline/`, `tasks/`, and `utils/`. Edit `config/train_config.json` to
control training defaults. The dataset defaults to `../../36_labels/`, with
`manifest.csv`, `labels.txt`, and WAV files referenced by `mixture_path` and
`noise_path`; pretrained and fine-tuned weights live in `checkpoint/`;
generated embedding caches live in `artifacts/`.

## Build, Test, and Development Commands

Create and activate a local environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the four stages through `main.py`, in order — `separator36` must run
before `head36`, which must run before `finetune36`:

```bash
python main.py separator36 --config config/train_config.json
python main.py head36 --config config/train_config.json
python main.py finetune36 --config config/train_config.json
python main.py test36 --config config/train_config.json
```

Each stage script (`train_separator.py`, `train_fusion_head.py`,
`finetune_fusion.py`, `evaluate_fusion_test.py`) also accepts `--smoke-test`
for a bounded run with a handful of samples per class and one epoch.

## Coding Style & Naming Conventions

Follow standard Python conventions: four-space indentation, `snake_case` for
functions and variables, `PascalCase` for classes, and `UPPER_CASE` for
constants. Add type hints to public data-loading and model interfaces where
practical. Use `pathlib.Path` for paths and keep CLI behavior behind `main()`
and an `if __name__ == "__main__"` guard. No formatter or linter is
configured, so keep imports grouped and changes PEP 8-compatible.

## Testing Guidelines

There is currently no dedicated automated test suite or coverage threshold.
Before submitting changes, run each stage with `--smoke-test` and confirm it
completes without non-finite losses. For loader changes, exercise at least
one file from each split. For separator changes, confirm validation SI-SDR
improves over the raw mixture SI-SDR baseline reported by
`train_separator.py`. Add future tests under `tests/` using names such as
`test_data.py` and `test_model.py`.

## Commit & Pull Request Guidelines

Use concise, imperative commit subjects (for example, `Fix PCM16 padding for
short clips`). Keep commits focused. Pull requests should explain the
motivation, list validation commands and results, note dataset or checkpoint
impacts, and include metric comparisons (accuracy/macro-F1 and separator
SI-SDR) for model changes. Attach screenshots only when changing diagrams or
other visual artifacts; link related issues when applicable.

## Data & Artifact Safety

Do not rewrite manifest paths unnecessarily: the loader derives local WAV
paths from `sample_id`. Avoid committing new large audio files, virtual
environments, or experimental checkpoints unless they are explicitly required
deliverables.
