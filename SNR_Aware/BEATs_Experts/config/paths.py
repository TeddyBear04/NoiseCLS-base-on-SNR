"""Default project paths, all relative to this repository checkout."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset" / "21_labels_dataset"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoint"
PRETRAINED_DIR = CHECKPOINT_DIR / "pretrained"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
BEATS_CHECKPOINT = PRETRAINED_DIR / "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
